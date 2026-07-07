# -*- coding: utf-8 -*-
"""
MLP cross-sectional ranking to event-driven backtest.

Run from project root:
    python backtest/run.py
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
import warnings

import duckdb
import numpy as np
import pandas as pd
# backtesting.py no longer used

# -- project root --
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategies import MLPStrategy, walk_forward, rank_ic, ic_summary
from strategies.labels import compute_forward_returns
# MLPBacktestStrategy replaced by portfolio simulation
from backtest.signals import run_portfolio, compute_benchmark

# ============================================================================
# CONFIG
# ============================================================================
DB_PATH = ROOT / "data" / "ashare.duckdb"
TEST_START = pd.Timestamp("2025-05-01")
TEST_END = pd.Timestamp("2026-06-26")
WARMUP_DAYS = 100
TRAIN_WINDOW = 252
MIN_TRAIN = 252
FORWARD_HORIZON = 5
TOP_K = 3
REBAL_INTERVAL = 5
CASH_PER_STOCK = 10000
COMMISSION = 0.0003
RISK_FREE_RATE = 0.025  # 1-year China government bond yield
SELECTED_FACTORS = [
    # Momentum (4)
    "Return_1d", "Return_5d", "Return_20d", "Reversal_60d",
    # Volatility (5)
    "ATR", "Volatility", "Volatility_60d", "Bollinger_width", "alpha060",
    # Price position / other (6)
    "Price_position_252d", "Stochastic_K", "Return_skew_20d",
    "Trend_strength", "SMA", "MACD_signal",
    # Pattern / intraday (3)
    "Gap_pct", "Body_pct", "Intraday_range_pct",
    # Volume / liquidity (2)
    "Volume_ratio", "Amihud_illiquidity",
    # Alpha composite (13)
    "alpha001", "alpha002", "alpha003", "alpha006", "alpha009",
    "alpha012", "alpha013", "alpha014", "alpha019", "alpha020",
    "alpha050", "alpha101", "alpha191",
]

PRED_PATH = ROOT / "data" / "predictions.parquet"
PRED_META_PATH = ROOT / "data" / "predictions_meta.json"

warnings.filterwarnings("ignore")


# ============================================================================
# Data loading
# ============================================================================

def load_factors(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load factor panel from DuckDB, return MultiIndex (date, code)."""
    df = con.execute("SELECT * FROM factor_values").fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "code"]).sort_index()
    return df


def load_labels(con: duckdb.DuckDBPyConnection) -> pd.Series:
    """Compute 5-day forward returns as labels."""
    kline = con.execute(
        "SELECT code, date, close FROM daily_kline ORDER BY code, date"
    ).fetchdf()
    return compute_forward_returns(kline, horizon=FORWARD_HORIZON)


def load_ohlcv_map(con: duckdb.DuckDBPyConnection, codes: list[str]) -> dict[str, pd.DataFrame]:
    """Load OHLCV data for specified codes.

    Returns {code: DataFrame} with DatetimeIndex and columns Open/High/Low/Close/Volume.
    """
    placeholders = ",".join(["?"] * len(codes))
    df = con.execute(
        f"SELECT code, date, open, high, low, close, volume "
        f"FROM daily_kline WHERE code IN ({placeholders}) "
        f"ORDER BY code, date",
        codes,
    ).fetchdf()
    df["date"] = pd.to_datetime(df["date"])

    ohlcv_map: dict[str, pd.DataFrame] = {}
    for code, grp in df.groupby("code"):
        grp = grp.set_index("date")
        grp = grp.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })
        ohlcv_map[code] = grp[["Open", "High", "Low", "Close", "Volume"]]

    return ohlcv_map



def load_or_train_predictions(factors, labels, factor_cols):
    """Load cached predictions if valid; otherwise train and cache."""
    if PRED_PATH.exists() and PRED_META_PATH.exists():
        try:
            cached_meta = json.loads(PRED_META_PATH.read_text(encoding="utf-8"))
            cached_factors = set(cached_meta.get("factor_names", []))
            cached_start = cached_meta.get("test_start")
            cached_end = cached_meta.get("test_end")
            current_factors = set(factor_cols)
            current_start = str(TEST_START.date())
            current_end = str(TEST_END.date())
            if (cached_factors == current_factors and
                cached_start == current_start and
                cached_end == current_end):
                preds = pd.read_parquet(PRED_PATH)
                if "prediction" in preds.columns:
                    preds = preds["prediction"]
                print("  [CACHE HIT] Loaded predictions from cache.")
                return preds, True
            else:
                print("  [CACHE STALE] Metadata mismatch, retraining ...")
        except Exception as e:
            print(f"  [CACHE ERROR] {e}, retraining ...")

    print("  [TRAINING] Walk-forward MLP predictions ...")
    print(f"  Test period: {TEST_START.date()} ~ {TEST_END.date()}")

    strategy = MLPStrategy(
        factor_names=factor_cols,
        hidden_layer_sizes=(25, 12),
        alpha=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        learning_rate=0.001,
        random_state=42,
    )

    preds = walk_forward(
        strategy, factors, labels,
        train_window=TRAIN_WINDOW,
        min_train=MIN_TRAIN,
        warmup_days=WARMUP_DAYS,
        test_start=TEST_START,
        test_end=TEST_END,
    )

    if len(preds) > 0:
        import datetime as dt
        PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
        pred_df = preds.to_frame("prediction")
        pred_df.to_parquet(PRED_PATH)
        meta = {
            "factor_names": factor_cols,
            "test_start": str(TEST_START.date()),
            "test_end": str(TEST_END.date()),
            "horizon": FORWARD_HORIZON,
            "hidden_layer_sizes": [25, 12],
            "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        PRED_META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Predictions cached to {PRED_PATH}")

    return preds, False


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("  MLP Event-Driven Backtest")
    print("=" * 60)

    # ---- 1. Load ----
    print("\n[1/5] Loading data ...")
    con = duckdb.connect(str(DB_PATH), read_only=True)

    print("  Loading factors ...")
    factors = load_factors(con)

    print("  Computing labels (5-day forward return) ...")
    labels = load_labels(con)

    common = factors.index.intersection(labels.index)
    factors = factors.loc[common]
    labels = labels.loc[common]
    mask = labels.notna()
    factors, labels = factors.loc[mask], labels.loc[mask]

    # Restrict to curated factors (same as run_mlp.py)
    available = [f for f in SELECTED_FACTORS if f in factors.columns]
    missing  = [f for f in SELECTED_FACTORS if f not in factors.columns]
    if missing:
        print(f"  WARNING: {len(missing)} selected factors missing: {missing}")
    factor_cols = available
    factors = factors[factor_cols]
    print(f"  Factors: {len(factor_cols)}, Samples: {len(factors)}")
    print(f"  Date range: {factors.index.get_level_values('date').min().date()}"
          f" ~ {factors.index.get_level_values('date').max().date()}")

    # ---- 2. MLP predictions (cache-aware) ----
    print(f"\n[2/5] MLP predictions ...")
    preds, _from_cache = load_or_train_predictions(factors, labels, factor_cols)

    if len(preds) == 0:
        print("  ERROR: No predictions generated. Check data and parameters.")
        con.close()
        return

    n_preds = len(preds)
    n_dates = preds.index.get_level_values("date").nunique()
    n_codes = preds.index.get_level_values("code").nunique()
    print(f"  Predictions: {n_preds} rows, {n_dates} dates, {n_codes} stocks")

    ric = rank_ic(preds, labels)
    ic_s = ic_summary(ric)
    print(f"  Rank IC: mean={ic_s['mean_ic']:.4f}, IR={ic_s['ir']:.2f}, "
          f"hit_rate={ic_s['hit_rate']:.2%}")

    # ---- 3. Build OHLCV map ----
    print(f"\n[3/5] Loading OHLCV data ...")
    pred_codes = sorted(preds.index.get_level_values("code").unique())
    print(f"  {len(pred_codes)} stocks ...")
    ohlcv_map = load_ohlcv_map(con, pred_codes)
    con.close()

    # ---- 4. Portfolio backtest (long-short) ----
    print(f"\n[4/5] Running long-short portfolio backtest (top-{TOP_K}, rebal={REBAL_INTERVAL}d) ...")
    port_stats, equity_df, trade_df = run_portfolio(
        preds, ohlcv_map,
        test_start=str(TEST_START.date()),
        top_k=TOP_K,
        rebal_interval=REBAL_INTERVAL,
        initial_cash_per_side=CASH_PER_STOCK,
        commission=COMMISSION,
        risk_free_rate=RISK_FREE_RATE,
        gap_filter=0.015,
    )

    n_trades = len(trade_df)
    n_buys = int((trade_df["action"] == "BUY").sum()) if n_trades > 0 else 0
    n_sells = int((trade_df["action"] == "SELL").sum()) if n_trades > 0 else 0
    n_shorts = int((trade_df["action"] == "SHORT").sum()) if n_trades > 0 else 0
    n_covers = int((trade_df["action"] == "COVER").sum()) if n_trades > 0 else 0
    print(f"  Trades: {n_trades} total (BUY={n_buys}, SELL={n_sells}, SHORT={n_shorts}, COVER={n_covers})")

    test_dates = sorted(preds.index.get_level_values("date").unique())
    bench_df = compute_benchmark(ohlcv_map, test_dates)

    # ========================================================================
    # REPORT
    # ========================================================================
    print("\n" + "=" * 60)
    print("  PORTFOLIO RESULTS")
    print("=" * 60)

    print(f"\n  {'Total Return:':<22} {port_stats.get('total_return', 0):>+10.2%}")
    print(f"  {'CAGR:':<22} {port_stats.get('cagr', 0):>+10.2%}")
    print(f"  {'Sharpe Ratio:':<22} {port_stats.get('sharpe', 0):>10.2f}")
    print(f"  {'Sortino Ratio:':<22} {port_stats.get('sortino', 0):>10.2f}")
    print(f"  {'Max Drawdown:':<22} {port_stats.get('max_drawdown', 0):>10.2%}")
    print(f"  {'Calmar Ratio:':<22} {port_stats.get('calmar', 0):>10.2f}")
    print(f"  {'Win Rate:':<22} {port_stats.get('win_rate', 0):>10.2%}")
    print(f"  {'Trading Days:':<22} {port_stats.get('n_days', 0):>10}")
    print(f"  {'Total Trades:':<22} {n_trades:>10}")

    # Benchmark
    if not bench_df.empty and len(bench_df) > 0:
        bench_total = float(bench_df['equity'].iloc[-1] / bench_df['equity'].iloc[0] - 1)
        bench_ret = bench_df['daily_ret'].dropna()
        b_mean = float(bench_ret.mean())
        b_std = float(bench_ret.std())
        bench_annual_ret = b_mean * 252.0
        bench_annual_std = b_std * np.sqrt(252.0)
        bench_sharpe = float((bench_annual_ret - RISK_FREE_RATE) / bench_annual_std) if bench_annual_std > 0 else 0.0
        bench_peak = bench_df['equity'].cummax()
        bench_dd = (bench_df['equity'] - bench_peak) / bench_peak
        bench_maxdd = float(bench_dd.min())

        print(f"\n  --- Benchmark (Equal-Weight All {len(ohlcv_map)} stocks) ---")
        print(f"  {'Total Return:':<22} {bench_total:>+10.2%}")
        print(f"  {'Sharpe Ratio:':<22} {bench_sharpe:>10.2f}")
        print(f"  {'Max Drawdown:':<22} {bench_maxdd:>10.2%}")

        excess = port_stats.get('total_return', 0) - bench_total
        print(f"\n  {'Excess Return:':<22} {excess:>+10.2%}")

    # IC consistency
    print(f"\n  --- IC Consistency ---")
    print(f"  Rank IC mean:  {ic_s['mean_ic']:.4f}")
    print(f"  Rank IC IR:    {ic_s['ir']:.2f}")
    print(f"  Rank IC hit:   {ic_s['hit_rate']:.2%}")



    # Save equity curve
    eq_path = ROOT / "backtest" / "equity.csv"
    equity_df.to_csv(eq_path)
    if not bench_df.empty:
        bench_df.to_csv(ROOT / "backtest" / "benchmark.csv")
    print(f"\n  Equity curve saved to: {eq_path}")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
