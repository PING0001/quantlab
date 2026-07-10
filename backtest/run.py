# -*- coding: utf-8 -*-
"""
LightGBM long-only backtest with overnight limit orders.

Loads pre-computed LGB predictions from run_lgb.py output, then simulates
daily entry/exit screening with call-auction + intraday execution.

Run from project root:
    python -m backtest.run
"""
from __future__ import annotations

import sys
from pathlib import Path
import warnings

import duckdb
import numpy as np
import pandas as pd

# -- project root --
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_PATH, POOL_NAME, get_pool_codes, get_lgb_predictions_path, get_backtest_dir

from strategies import rank_ic, ic_summary
from strategies.labels import compute_forward_returns
from backtest.signals import run_portfolio, compute_benchmark

# ============================================================================
# CONFIG
# ============================================================================
TEST_START = pd.Timestamp("2025-06-01")

HORIZONS = [3, 5, 10, 20]
FORWARD_HORIZON = 5  # column used for portfolio simulation

# ---- Backtest execution params ----
MAX_POSITIONS = 10
ENTRY_THRESHOLD = 0.01
EXIT_THRESHOLD = -0.01
AUCTION_BUFFER = 0.01
SELL_MARKUP = 0.0005
CASH_PER_STOCK = 10000
COMMISSION = 0.0006  # 0.06% per side = 0.12% round-trip
RISK_FREE_RATE = 0.025

SELECTED_FACTORS = [
    # Momentum (3)
    "Return_5d", "Return_20d", "Reversal_60d",
    # Volatility (5)
    "ATR", "Volatility", "Volatility_60d", "Bollinger_width", "alpha060",
    # Price position / other (6)
    "Price_position_252d", "Stochastic_K", "Return_skew_20d",
    "Trend_strength", "SMA", "MACD_signal",
    # Pattern / intraday (3)
    "Gap_pct", "Body_pct", "Intraday_range_pct",
    # Volume / liquidity (2)
    "Volume_ratio", "Amihud_illiquidity",
    # Alpha composite (22)
    "alpha001", "alpha002", "alpha003", "alpha006", "alpha007",
    "alpha009", "alpha012", "alpha013", "alpha014", "alpha017",
    "alpha018", "alpha019", "alpha020", "alpha028", "alpha035",
    "alpha038", "alpha046", "alpha050", "alpha057",
    "alpha101", "alpha191",
    # Market cap / amount (3)
    "AvgAmount_90d", "LnMktCap", "LnFloatCap",
    # Turnover (2)
    "Turnover_3d", "Turnover_3d_ratio",
    # Intraday (1)
    "Intraday_return",
    # Market state (5)
    "CSI_return_1d", "CSI_return_20d", "CSI_volatility_20d",
    "HS300_return_1d", "HS300_return_20d",
    # Cross-sectional ranks (3)
    "Return_1d_rank", "Return_20d_rank", "Turnover_3d_rank",
    # ST status (1)
    "IsST",
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


def load_labels(con: duckdb.DuckDBPyConnection, codes: list[str]) -> pd.DataFrame:
    """Compute forward returns for all horizons, return DataFrame."""
    placeholders = ",".join(["?"] * len(codes))
    kline = con.execute(
        f"SELECT code, date, close FROM daily_kline WHERE code IN ({placeholders}) ORDER BY code, date",
        codes,
    ).fetchdf()
    labels = {}
    for h in HORIZONS:
        labels[h] = compute_forward_returns(kline, horizon=h)
    return pd.DataFrame(labels)


def load_ohlcv_map(con: duckdb.DuckDBPyConnection, codes: list[str]) -> dict[str, pd.DataFrame]:
    """Load OHLCV data + IsST flag for specified codes.

    Returns {code: DataFrame} with DatetimeIndex and columns
    Open/High/Low/Close/Volume/Pct_chg/IsST.
    """
    placeholders = ",".join(["?"] * len(codes))

    df = con.execute(
        f"SELECT code, date, open, high, low, close, volume, pct_chg "
        f"FROM daily_kline WHERE code IN ({placeholders}) "
        f"ORDER BY code, date",
        codes,
    ).fetchdf()
    df["date"] = pd.to_datetime(df["date"])

    # Merge IsST from factor_values
    try:
        isst = con.execute(
            f"SELECT code, date, IsST FROM factor_values WHERE code IN ({placeholders})",
            codes,
        ).fetchdf()
        if not isst.empty:
            isst["date"] = pd.to_datetime(isst["date"])
            df = df.merge(isst, on=["code", "date"], how="left")
        else:
            df["IsST"] = 0
    except Exception:
        df["IsST"] = 0
    df["IsST"] = df["IsST"].fillna(0).astype(int)

    ohlcv_map: dict[str, pd.DataFrame] = {}
    for code, grp in df.groupby("code"):
        grp = grp.set_index("date")
        grp = grp.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
            "pct_chg": "Pct_chg",
        })
        grp["IsST"] = grp.get("IsST", 0)
        ohlcv_map[code] = grp[["Open", "High", "Low", "Close", "Volume", "Pct_chg", "IsST"]]

    return ohlcv_map


def main():
    print("=" * 60)
    print("  LightGBM Long-Only Backtest")
    print(f"  Pool: {POOL_NAME}")
    print("=" * 60)

    # ---- 1. Load factors + labels (for IC reference) ----
    print("\n[1/5] Loading data ...")
    con = duckdb.connect(str(DB_PATH), read_only=True)

    print("  Loading factors ...")
    factors = load_factors(con)

    print("  Computing labels ...")
    pool_codes = get_pool_codes()
    labels = load_labels(con, pool_codes)

    common = factors.index.intersection(labels.index)
    factors = factors.loc[common]
    labels = labels.loc[common]
    mask = labels.notna().all(axis=1)
    factors, labels = factors.loc[mask], labels.loc[mask]

    available = [f for f in SELECTED_FACTORS if f in factors.columns]
    factor_cols = available
    factors = factors[factor_cols]
    print(f"  Factors: {len(factor_cols)}, Samples: {len(factors)}")
    print(f"  Date range: {factors.index.get_level_values('date').min().date()}"
          f" ~ {factors.index.get_level_values('date').max().date()}")

    # ---- 2. Load pre-computed LGB predictions ----
    print(f"\n[2/5] Loading LGB predictions ...")
    pred_path = get_lgb_predictions_path()
    if not pred_path.exists():
        print(f"  ERROR: Predictions not found at {pred_path}")
        print("  Run: python run_lgb.py")
        con.close()
        return

    pred_df = pd.read_parquet(pred_path)
    preds_5d = pred_df["pred_5d"]

    n_preds = len(preds_5d)
    n_dates = preds_5d.index.get_level_values("date").nunique()
    n_codes = preds_5d.index.get_level_values("code").nunique()
    print(f"  Predictions (5d): {n_preds} rows, {n_dates} dates, {n_codes} stocks")

    # IC reference
    labels_5d = labels[FORWARD_HORIZON]
    common = preds_5d.index.intersection(labels_5d.index)
    ric = rank_ic(preds_5d.loc[common], labels_5d.loc[common])
    ic_s = ic_summary(ric)
    print(f"  Rank IC (5d): mean={ic_s['mean_ic']:.4f}, IR={ic_s['ir']:.2f}, "
          f"hit_rate={ic_s['hit_rate']:.2%}")

    # ---- 3. Build OHLCV map ----
    print(f"\n[3/5] Loading OHLCV data ...")
    pred_codes = sorted(preds_5d.index.get_level_values("code").unique())
    print(f"  {len(pred_codes)} stocks ...")
    ohlcv_map = load_ohlcv_map(con, pred_codes)
    con.close()

    # ---- 4. Portfolio backtest ----
    print(f"\n[4/5] Running long-only backtest "
          f"(max_pos={MAX_POSITIONS}, entry>{ENTRY_THRESHOLD}, exit<{EXIT_THRESHOLD}) ...")
    port_stats, equity_df, trade_df = run_portfolio(
        preds_5d, ohlcv_map,
        test_start=str(TEST_START.date()),
        max_positions=MAX_POSITIONS,
        entry_threshold=ENTRY_THRESHOLD,
        exit_threshold=EXIT_THRESHOLD,
        auction_buffer=AUCTION_BUFFER,
        sell_markup=SELL_MARKUP,
        initial_cash_per_stock=CASH_PER_STOCK,
        commission=COMMISSION,
        risk_free_rate=RISK_FREE_RATE,
    )

    n_trades = len(trade_df)
    n_buys = int((trade_df["action"] == "BUY").sum()) if n_trades > 0 else 0
    n_sells = int((trade_df["action"] == "SELL").sum()) if n_trades > 0 else 0
    print(f"  Trades: {n_trades} total (BUY={n_buys}, SELL={n_sells})")

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