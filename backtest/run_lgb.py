# -*- coding: utf-8 -*-
"""
LightGBM long-only backtest — 5-day rebalancing with pred_20d.

Loads pre-computed LightGBM predictions from run_lgb.py output, then simulates
a periodic rebalancing strategy:
  - Every 5 trading days: rank stocks by pred_20d, sell positions that dropped
    out of top-N, buy top-N stocks not yet held.
  - All trades use overnight limit orders with auction + intraday execution.
  - Sell orders persist across non-rebalance days.

Run from project root:
    python -m backtest.run_lgb
"""
from __future__ import annotations

import sys
from pathlib import Path
import warnings

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_PATH, POOL_NAME, get_pool_codes, get_lgb_predictions_path, get_backtest_dir

from strategies import rank_ic, ic_summary
from strategies.labels import compute_median_close, compute_nextopen_limit_mask
from backtest.signals import run_portfolio_rebalance, compute_benchmark, run_long_short

# ============================================================================
# CONFIG
# ============================================================================
TEST_START = pd.Timestamp("2025-06-01")

PRED_COL = "pred_label"
LABEL_HORIZON = "median_16_20"  # median close return T+16~T+20

MAX_POSITIONS = 10
REBALANCE_FREQ = 5          # rebalance every 5 trading days
AUCTION_BUFFER = 0.02
SELL_MARKUP = 0.001
CASH_PER_STOCK = 10000
COMMISSION = 0.0006
STAMP_DUTY = 0.0005
BORROW_RATE = 0.08          # 融券年化费率
RISK_FREE_RATE = 0.025

warnings.filterwarnings("ignore")


# ============================================================================
# Data loading
# ============================================================================

def load_ohlcv_map(con: duckdb.DuckDBPyConnection, codes: list[str]) -> dict[str, pd.DataFrame]:
    placeholders = ",".join(["?"] * len(codes))

    df = con.execute(
        f"SELECT code, date, open, high, low, close, volume, pct_chg "
        f"FROM daily_kline WHERE code IN ({placeholders}) "
        f"ORDER BY code, date",
        codes,
    ).fetchdf()
    df["date"] = pd.to_datetime(df["date"])

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
    print("  LightGBM 20d 5-Day Rebalance Backtest")
    print(f"  Pool: {POOL_NAME}")
    print("=" * 60)

    # ---- 1. Load predictions ----
    print("\n[1/4] Loading LGBM predictions ...")
    pred_path = get_lgb_predictions_path()
    if not pred_path.exists():
        print(f"  ERROR: Predictions not found at {pred_path}")
        print("  Run: python run_lgb.py")
        return

    pred_df = pd.read_parquet(pred_path)

    if PRED_COL not in pred_df.columns:
        available = list(pred_df.columns)
        print(f"  ERROR: Column '{PRED_COL}' not found. Available: {available}")
        return

    preds = pred_df[PRED_COL]
    n_preds = len(preds)
    n_dates = preds.index.get_level_values("date").nunique()
    n_codes = preds.index.get_level_values("code").nunique()
    print(f"  Predictions ({PRED_COL}): {n_preds} rows, {n_dates} dates, {n_codes} stocks")

    # ---- 2. Load OHLCV + metadata ----
    print(f"\n[2/4] Loading OHLCV + metadata ...")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    pool_codes = get_pool_codes()

    pred_codes = sorted(preds.index.get_level_values("code").unique())
    ohlcv_map = load_ohlcv_map(con, pred_codes)

    full_ohlcv = load_ohlcv_map(con, pool_codes)
    print(f"  OHLCV: {len(ohlcv_map)} prediction stocks, {len(full_ohlcv)} pool stocks")

    # Excluded codes (ST/退 in name)
    excluded_codes = set()
    try:
        placeholders = ",".join(["?"] * len(pred_codes))
        name_df = con.execute(
            f"SELECT code, name FROM stock_info WHERE code IN ({placeholders})",
            pred_codes,
        ).fetchdf()
        for _, row in name_df.iterrows():
            n = row["name"]
            if isinstance(n, str) and ("ST" in n or "退" in n):
                excluded_codes.add(row["code"])
    except Exception:
        pass
    print(f"  Excluded (ST/退): {len(excluded_codes)} stocks")

    # Delist info
    delist_info = {}
    try:
        dl_df = con.execute("SELECT code, delist_date FROM delist_info").fetchdf()
        if not dl_df.empty:
            delist_info = {r["code"]: pd.Timestamp(r["delist_date"]) for _, r in dl_df.iterrows()}
    except Exception:
        pass
    print(f"  Delist info: {len(delist_info)} stocks")
    con.close()

    # ---- 3. IC reference check (with limit-hit + ST filter, matching train eval) ----
    print(f"\n[3/4] IC reference ({PRED_COL} vs median_16_20) ...")
    con_r = duckdb.connect(str(DB_PATH), read_only=True)
    placeholders = ",".join(["?"] * len(pool_codes))
    kline = con_r.execute(
        f"SELECT code, date, open, close FROM daily_kline WHERE code IN ({placeholders}) ORDER BY code, date",
        pool_codes,
    ).fetchdf()

    # Load ST series from factor_values
    try:
        st_df = con_r.execute(
            f"SELECT code, date, IsST FROM factor_values WHERE code IN ({placeholders})",
            pool_codes,
        ).fetchdf()
        if not st_df.empty:
            st_df["date"] = pd.to_datetime(st_df["date"])
            st_series = st_df.set_index(["date", "code"])["IsST"].astype(bool)
        else:
            st_series = None
    except Exception:
        st_series = None

    limit_mask = compute_nextopen_limit_mask(kline, st_series=st_series)

    labels = compute_median_close(kline, start_day=16, end_day=20, delist_info=delist_info)
    con_r.close()

    common = preds.index.intersection(labels.index)
    p_ic = preds.loc[common]
    l_ic = labels.loc[common]

    safe = ~limit_mask.reindex(p_ic.index, fill_value=False)
    if st_series is not None:
        safe = safe & ~st_series.reindex(p_ic.index, fill_value=False)
    ric = rank_ic(p_ic.loc[safe], l_ic.loc[safe])
    ic_s = ic_summary(ric)
    n_excl = (~safe).sum()
    print(f"  Rank IC (median_16_20): mean={ic_s['mean_ic']:.4f}, IR={ic_s['ir']:.2f}, "
          f"hit_rate={ic_s['hit_rate']:.2%}, {ic_s['n_periods']} dates, {n_excl} excluded")

    pred_dates = sorted(preds.index.get_level_values("date").unique())
    test_end_date = str(pred_dates[-1].date()) if len(pred_dates) > 0 else str(TEST_START.date())
    print(f"  Prediction period: {pred_dates[0].date()} ~ {pred_dates[-1].date()} ({len(pred_dates)} dates)")
    print(f"  Backtest will stop at prediction end: {test_end_date}")

    # ---- 4. Portfolio backtest ----
    print(f"\n[4/4] Running 5-day rebalance backtest "
          f"(max_pos={MAX_POSITIONS}, rebalance_freq={REBALANCE_FREQ}) ...")
    port_stats, equity_df, trade_df = run_portfolio_rebalance(
        preds, ohlcv_map,
        test_start=str(TEST_START.date()),
        max_positions=MAX_POSITIONS,
        rebalance_freq=REBALANCE_FREQ,
        auction_buffer=AUCTION_BUFFER,
        sell_markup=SELL_MARKUP,
        excluded_codes=excluded_codes,
        initial_cash_per_stock=CASH_PER_STOCK,
        commission=COMMISSION,
        risk_free_rate=RISK_FREE_RATE,
        delist_info=delist_info,
    )

    # Truncate equity to prediction end date
    if not equity_df.empty and test_end_date is not None:
        cutoff = pd.Timestamp(test_end_date)
        equity_df = equity_df[equity_df.index <= cutoff]
        # Recompute stats on truncated equity
        from backtest.signals import _compute_stats
        port_stats, equity_df = _compute_stats(equity_df["Equity"], risk_free_rate=RISK_FREE_RATE)

    n_trades = len(trade_df)
    n_buys = int((trade_df["action"] == "BUY").sum()) if n_trades > 0 else 0
    n_sells = int((trade_df["action"] == "SELL").sum()) if n_trades > 0 else 0
    print(f"  Trades: {n_trades} total (BUY={n_buys}, SELL={n_sells})")

    test_dates = sorted(preds.index.get_level_values("date").unique())
    # Truncate benchmark dates to prediction end
    if test_end_date is not None:
        cutoff = pd.Timestamp(test_end_date)
        test_dates = [d for d in test_dates if d <= cutoff]
    bench_df = compute_benchmark(full_ohlcv, test_dates, delist_info=delist_info)

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

        print(f"\n  --- Benchmark (Equal-Weight All {len(full_ohlcv)} stocks) ---")
        print(f"  {'Total Return:':<22} {bench_total:>+10.2%}")
        print(f"  {'Sharpe Ratio:':<22} {bench_sharpe:>10.2f}")
        print(f"  {'Max Drawdown:':<22} {bench_maxdd:>10.2%}")

        excess = port_stats.get('total_return', 0) - bench_total
        print(f"\n  {'Excess Return:':<22} {excess:>+10.2%}")

    # IC consistency
    print(f"\n  --- IC Consistency ({PRED_COL}) ---")
    print(f"  Rank IC mean:  {ic_s['mean_ic']:.4f}")
    print(f"  Rank IC IR:    {ic_s['ir']:.2f}")
    print(f"  Rank IC hit:   {ic_s['hit_rate']:.2%}")

    # Save equity curve
    bt_dir = get_backtest_dir()
    bt_dir.mkdir(parents=True, exist_ok=True)
    eq_path = bt_dir / "equity_lgb_20d_5d_rebalance.csv"
    equity_df.to_csv(eq_path)
    if not bench_df.empty:
        bench_df.to_csv(bt_dir / "benchmark.csv")
    print(f"\n  Equity curve saved to: {eq_path}")

    # ========================================================================
    # LONG-SHORT SIGNAL TEST
    # ========================================================================
    print(f"\n[5/5] Running long-short signal test (n_long={MAX_POSITIONS}, n_short={MAX_POSITIONS}) ...")
    ls_stats, ls_equity = run_long_short(
        preds, full_ohlcv,
        n_long=MAX_POSITIONS,
        n_short=MAX_POSITIONS,
        test_start=str(TEST_START.date()),
        excluded_codes=excluded_codes,
        risk_free_rate=RISK_FREE_RATE,
        commission=COMMISSION,
        stamp_duty=STAMP_DUTY,
        delist_info=delist_info,
        borrow_rate=BORROW_RATE,
    )

    if ls_stats:
        n_days = ls_stats.get("n_days", 0)
        n_traded = ls_stats.get("n_days_traded", 0)
        n_long_skip = ls_stats.get("n_long_limit_skipped", 0)
        n_short_skip = ls_stats.get("n_short_limit_skipped", 0)
        print(f"\n  --- Long-Short Results ({n_traded}/{n_days} days traded) ---")
        print(f"  {'Total Return:':<22} {ls_stats.get('total_return', 0):>+10.2%}")
        print(f"  {'Mean Daily Ret:':<22} {ls_stats.get('mean_daily_ret', 0):>+10.4%}")
        print(f"  {'Std Daily Ret:':<22} {ls_stats.get('std_daily_ret', 0):>10.4%}")
        print(f"  {'Sharpe Ratio:':<22} {ls_stats.get('sharpe', 0):>10.2f}")
        print(f"  {'Limit-Up Skipped:':<22} {n_long_skip:>10} (long leg)")
        print(f"  {'Limit-Down Skipped:':<22} {n_short_skip:>10} (short leg)")

        ls_path = bt_dir / "equity_lgb_20d_long_short.csv"
        ls_equity.to_csv(ls_path)
        print(f"\n  Long-short equity saved to: {ls_path}")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
