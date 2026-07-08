# -*- coding: utf-8 -*-
"""
Multi-head MLP cross-sectional ranking to event-driven backtest.

Trains one multi-head MLP (1d/3d/5d/10d), uses the 5d predictions for
portfolio simulation.

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

# -- project root --
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_PATH, POOL_NAME, get_pool_codes, get_predictions_path, get_predictions_meta_path, get_backtest_dir

from strategies import MLPStrategy, walk_forward, rank_ic, ic_summary
from strategies.labels import compute_forward_returns
from backtest.signals import run_portfolio, compute_benchmark

# ============================================================================
# CONFIG
# ============================================================================
TEST_START = pd.Timestamp("2025-05-01")
TEST_END = pd.Timestamp("2026-06-26")
WARMUP_DAYS = 100
TRAIN_WINDOW = 252
MIN_TRAIN = 252

HORIZONS = [1, 3, 5, 10]
FORWARD_HORIZON = 5  # column used for portfolio simulation

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

warnings.filterwarnings("ignore")


# ============================================================================
# Data loading
# ============================================================================

def load_factors(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load factor panel from DuckDB, filtered to current pool's stock codes."""
    pool_codes = get_pool_codes()
    placeholders = ",".join(["?"] * len(pool_codes))
    query = f"SELECT * FROM factor_values WHERE code IN ({placeholders})"
    df = con.execute(query, pool_codes).fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "code"]).sort_index()
    return df


def load_labels(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Compute forward returns for all horizons, return DataFrame."""
    kline = con.execute(
        "SELECT code, date, close FROM daily_kline ORDER BY code, date"
    ).fetchdf()
    labels = {}
    for h in HORIZONS:
        labels[h] = compute_forward_returns(kline, horizon=h)
    return pd.DataFrame(labels)


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


def load_or_train_predictions(factors: pd.DataFrame, labels: pd.DataFrame,
                              factor_cols: list[str]) -> pd.DataFrame:
    """Load cached multi-head predictions if valid; otherwise train and cache.

    Returns a DataFrame with columns pred_1d, pred_3d, pred_5d, pred_10d
    indexed by (date, code).
    """
    pred_path = get_predictions_path()
    pred_meta_path = get_predictions_meta_path()
    if pred_path.exists() and pred_meta_path.exists():
        try:
            cached_meta = json.loads(pred_meta_path.read_text(encoding="utf-8"))
            cached_factors = set(cached_meta.get("factor_names", []))
            cached_start = cached_meta.get("test_start")
            cached_end = cached_meta.get("test_end")
            cached_horizons = cached_meta.get("horizons", [1, 3, 5, 10])
            current_factors = set(factor_cols)
            if (cached_factors == current_factors and
                cached_start == str(TEST_START.date()) and
                cached_end == str(TEST_END.date()) and
                cached_horizons == HORIZONS):
                pred_df = pd.read_parquet(pred_path)
                print("  [CACHE HIT] Loaded predictions from cache.")
                return pred_df
            else:
                print("  [CACHE STALE] Metadata mismatch, retraining ...")
        except Exception as e:
            print(f"  [CACHE ERROR] {e}, retraining ...")

    print("  [TRAINING] Walk-forward multi-head MLP predictions ...")
    print(f"  Horizons: {HORIZONS}")
    print(f"  Test period: {TEST_START.date()} ~ {TEST_END.date()}")

    strategy = MLPStrategy(
        factor_names=factor_cols,
        horizons=tuple(HORIZONS),
        hidden_layer_sizes=(25, 12),
        alpha=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        learning_rate=0.001,
        random_state=42,
    )

    pred_df = walk_forward(
        strategy, factors, labels,
        train_window=TRAIN_WINDOW,
        min_train=MIN_TRAIN,
        warmup_days=WARMUP_DAYS,
        test_start=TEST_START,
        test_end=TEST_END,
    )

    if isinstance(pred_df, pd.DataFrame) and len(pred_df) > 0:
        import datetime as dt
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_parquet(pred_path)
        meta = {
            "factor_names": factor_cols,
            "horizons": HORIZONS,
            "test_start": str(TEST_START.date()),
            "test_end": str(TEST_END.date()),
            "hidden_layer_sizes": [25, 12],
            "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        pred_meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Predictions cached to {pred_path}")

    return pred_df


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("  Multi-Head MLP Event-Driven Backtest")
    print("=" * 60)

    # ---- 1. Load ----
    print("\n[1/5] Loading data ...")
    con = duckdb.connect(str(DB_PATH), read_only=True)

    print("  Loading factors ...")
    factors = load_factors(con)

    print("  Computing labels (all horizons) ...")
    labels = load_labels(con)

    common = factors.index.intersection(labels.index)
    factors = factors.loc[common]
    labels = labels.loc[common]
    mask = labels.notna().all(axis=1)
    factors, labels = factors.loc[mask], labels.loc[mask]

    # Restrict to curated factors
    available = [f for f in SELECTED_FACTORS if f in factors.columns]
    missing = [f for f in SELECTED_FACTORS if f not in factors.columns]
    if missing:
        print(f"  WARNING: {len(missing)} selected factors missing: {missing}")
    factor_cols = available
    factors = factors[factor_cols]
    print(f"  Factors: {len(factor_cols)}, Samples: {len(factors)}")
    print(f"  Date range: {factors.index.get_level_values('date').min().date()}"
          f" ~ {factors.index.get_level_values('date').max().date()}")

    # ---- 2. Multi-head MLP predictions (cache-aware) ----
    print(f"\n[2/5] MLP predictions ...")
    pred_df = load_or_train_predictions(factors, labels, factor_cols)

    if not isinstance(pred_df, pd.DataFrame) or len(pred_df) == 0:
        print("  ERROR: No predictions generated. Check data and parameters.")
        con.close()
        return

    # Extract the 5d column for portfolio simulation
    preds_5d = pred_df["pred_5d"]

    n_preds = len(preds_5d)
    n_dates = preds_5d.index.get_level_values("date").nunique()
    n_codes = preds_5d.index.get_level_values("code").nunique()
    print(f"  Predictions (5d): {n_preds} rows, {n_dates} dates, {n_codes} stocks")

    # IC on the 5d horizon
    labels_5d = labels[FORWARD_HORIZON]
    ric = rank_ic(preds_5d, labels_5d)
    ic_s = ic_summary(ric)
    print(f"  Rank IC (5d): mean={ic_s['mean_ic']:.4f}, IR={ic_s['ir']:.2f}, "
          f"hit_rate={ic_s['hit_rate']:.2%}")

    # ---- 3. Build OHLCV map ----
    print(f"\n[3/5] Loading OHLCV data ...")
    pred_codes = sorted(preds_5d.index.get_level_values("code").unique())
    print(f"  {len(pred_codes)} stocks ...")
    ohlcv_map = load_ohlcv_map(con, pred_codes)
    con.close()

    # ---- 4. Portfolio backtest (long-short) ----
    print(f"\n[4/5] Running long-short portfolio backtest (top-{TOP_K}, rebal={REBAL_INTERVAL}d) ...")
    port_stats, equity_df, trade_df = run_portfolio(
        preds_5d, ohlcv_map,
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

    test_dates = sorted(preds_5d.index.get_level_values("date").unique())
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
    print(f"\n  --- IC Consistency (5d) ---")
    print(f"  Rank IC mean:  {ic_s['mean_ic']:.4f}")
    print(f"  Rank IC IR:    {ic_s['ir']:.2f}")
    print(f"  Rank IC hit:   {ic_s['hit_rate']:.2%}")

    # Save equity curve
    bt_dir = get_backtest_dir()
    bt_dir.mkdir(parents=True, exist_ok=True)
    eq_path = bt_dir / "equity.csv"
    equity_df.to_csv(eq_path)
    if not bench_df.empty:
        bench_df.to_csv(bt_dir / "benchmark.csv")
    print(f"\n  Equity curve saved to: {eq_path}")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()