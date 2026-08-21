# -*- coding: utf-8 -*-
"""
Dual-regression combined backtest — DAILY rebalancing (bench 2026-08, v4).

Loads both model prediction parquets (20d / 6d), blends them into a single
score = 0.4*pred_20d + 0.6*pred_6d (label anchor close[T]; no percentile
layer, 2026-08-21 用户口径 v4), then simulates daily:
  - buy: top-k of cash slots (k = max_positions - held), bargain limit
    close*(1+score-3%)
  - sell: every held position at target price close*(1+score)

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

from config import (DB_PATH, POOL_NAME, get_pool_codes, get_backtest_dir,
                    get_lgb_predictions_path, MODEL_CONFIGS)

from strategies import rank_ic, ic_summary
from strategies.labels import compute_median_open, compute_nextopen_limit_mask
from strategies.combine import combine_scores
from backtest.signals import run_portfolio_rebalance, compute_benchmark, run_long_short

# ============================================================================
# CONFIG
# ============================================================================
TEST_START = pd.Timestamp("2025-06-01")

PRED_COLS = {"20d": "pred_label_20d", "6d": "pred_label_6d"}
W20, W6 = 0.4, 0.6             # v4：score = 0.4*p20 + 0.6*p6（2026-08-21 用户裁定）

MAX_POSITIONS = 10
REBALANCE_FREQ = 1          # 每日调仓（2026-08-21 用户裁定，spec §3.6）
AUCTION_BUFFER = 0.03        # 买入限价 = 收盘×(1+pred−3%)（2026-08-21 用户口径 v3）
SELL_MARKUP = 0.0            # 卖出限价 = 收盘×(1+pred) 目标价，无上浮（v3）
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


def load_predictions() -> dict[str, pd.Series]:
    preds = {}
    for m, col in PRED_COLS.items():
        path = get_lgb_predictions_path(m)
        if not path.exists():
            print(f"  ERROR: predictions for {m} not found at {path}")
            print("  Run: python run_lgb.py --model all")
            sys.exit(1)
        pdf = pd.read_parquet(path)
        if col not in pdf.columns:
            print(f"  ERROR: column '{col}' not found in {path}")
            sys.exit(1)
        preds[m] = pdf[col]
        n_dates = preds[m].index.get_level_values("date").nunique()
        print(f"  {m}: {len(preds[m])} rows, {n_dates} dates ({col})")
    return preds


def holding_days(trade_df: pd.DataFrame) -> pd.Series:
    """FIFO pairing of BUY/SELL per code -> holding days distribution."""
    holds: list[float] = []
    queues: dict[str, list] = {}
    for _, t in trade_df.sort_values("date").iterrows():
        q = queues.setdefault(t["code"], [])
        if t["action"] == "BUY":
            q.append((t["date"], t["shares"]))
        elif t["action"] == "SELL":
            remain = t["shares"]
            while remain > 0 and q:
                bdate, bshares = q[0]
                take = min(remain, bshares)
                holds.append((t["date"] - bdate).days)
                remain -= take
                if take == bshares:
                    q.pop(0)
                else:
                    q[0] = (bdate, bshares - take)
    return pd.Series(holds, dtype=float)


def main():
    print("=" * 60)
    print("  Dual-Regression Backtest — DAILY, v4: score = 0.4*p20 + 0.6*p6")
    print(f"  Pool: {POOL_NAME} | label anchor: close[T] | top-k cash slots + bargain limit")
    print("=" * 60)

    # ---- 1. Load + combine predictions ----
    print("\n[1/5] Loading predictions ...")
    preds = load_predictions()
    score = combine_scores(preds["20d"], preds["6d"], w20=W20, w6=W6)

    n_dates = score.index.get_level_values("date").nunique()
    print(f"  combined: {len(score)} rows, {n_dates} dates")
    print(f"  score quantiles: "
          f"1%={score.quantile(0.01):+.4f} 50%={score.quantile(0.5):+.4f} "
          f"99%={score.quantile(0.99):+.4f} |0.999|={abs(score).quantile(0.999):.4f}")
    # 量纲 sanity：score 必须是收益量纲，否则限价公式失真
    if abs(score).quantile(0.999) > 0.35:
        print("  ERROR: score magnitude exceeds plausible return range — "
              "check combine inputs")
        sys.exit(1)

    # ---- 2. Load OHLCV + metadata ----
    print(f"\n[2/5] Loading OHLCV + metadata ...")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    pool_codes = get_pool_codes()
    pred_codes = sorted(score.index.get_level_values("code").unique())
    ohlcv_map = load_ohlcv_map(con, pred_codes)
    full_ohlcv = load_ohlcv_map(con, pool_codes)
    print(f"  OHLCV: {len(ohlcv_map)} prediction stocks, {len(full_ohlcv)} pool stocks")

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

    delist_info = {}
    try:
        dl_df = con.execute("SELECT code, delist_date FROM delist_info").fetchdf()
        if not dl_df.empty:
            delist_info = {r["code"]: pd.Timestamp(r["delist_date"]) for _, r in dl_df.iterrows()}
    except Exception:
        pass
    print(f"  Delist info: {len(delist_info)} stocks")
    con.close()

    # ---- 3. IC reference (each model vs own label + score vs both) ----
    print(f"\n[3/5] IC reference ...")
    con_r = duckdb.connect(str(DB_PATH), read_only=True)
    placeholders = ",".join(["?"] * len(pool_codes))
    kline = con_r.execute(
        f"SELECT code, date, open, close FROM daily_kline WHERE code IN ({placeholders}) ORDER BY code, date",
        pool_codes,
    ).fetchdf()

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
    con_r.close()

    limit_mask = compute_nextopen_limit_mask(kline, st_series=st_series)
    # 预测帧在训练入口已排除 limit/ST/退市行，故此处计数为 0 属预期；
    # 掩码仍用于 IC 块的 safe 过滤（防未来数据变化）
    n_limit = int(limit_mask.loc[score.index].sum()) if not limit_mask.empty else 0
    n_st = int(st_series.loc[score.index].sum()) if st_series is not None else 0
    print(f"  pred rows with limit-hit/ST (expected 0, excluded upstream): "
          f"{n_limit} / {n_st}")

    def _safe_ic(p: pd.Series, lab: pd.Series) -> dict:
        common = p.index.intersection(lab.index)
        p_c, l_c = p.loc[common], lab.loc[common]
        safe = ~limit_mask.reindex(common, fill_value=False)
        if st_series is not None:
            safe = safe & ~st_series.reindex(common, fill_value=False)
        return ic_summary(rank_ic(p_c.loc[safe], l_c.loc[safe]))

    for m in PRED_COLS:
        cfg = MODEL_CONFIGS[m]
        s0, e0 = cfg["label_window"]
        lab = compute_median_open(kline, start_day=s0, end_day=e0, baseline=cfg["baseline"])
        s_own = _safe_ic(preds[m], lab)
        s_comb = _safe_ic(score, lab)
        print(f"  {m:>3} label: own IC={s_own['mean_ic']:+.4f} (IR {s_own['ir']:.2f}) | "
              f"score IC={s_comb['mean_ic']:+.4f} (IR {s_comb['ir']:.2f})")

    pred_dates = sorted(score.index.get_level_values("date").unique())
    test_end_date = str(pred_dates[-1].date())
    print(f"  Prediction period: {pred_dates[0].date()} ~ {pred_dates[-1].date()} ({len(pred_dates)} dates)")

    # ---- 4. Portfolio backtest (daily rebalance) ----
    print(f"\n[4/5] Running DAILY rebalance backtest "
          f"(max_pos={MAX_POSITIONS}, rebalance_freq={REBALANCE_FREQ}, v4 single score) ...")
    port_stats, equity_df, trade_df = run_portfolio_rebalance(
        score, ohlcv_map,
        test_start=str(TEST_START.date()),
        max_positions=MAX_POSITIONS,
        rebalance_freq=REBALANCE_FREQ,
        auction_buffer=AUCTION_BUFFER,
        sell_markup=SELL_MARKUP,
        excluded_codes=excluded_codes,
        initial_cash_per_stock=CASH_PER_STOCK,
        commission=COMMISSION,
        stamp_duty=STAMP_DUTY,
        risk_free_rate=RISK_FREE_RATE,
        delist_info=delist_info,
    )

    if not equity_df.empty and test_end_date is not None:
        cutoff = pd.Timestamp(test_end_date)
        pos_keep = {k: port_stats[k] for k in
                    ("avg_positions", "n_days_empty", "n_days_full", "n_days")
                    if k in port_stats}
        equity_df = equity_df[equity_df.index <= cutoff]
        from backtest.signals import _compute_stats
        port_stats, equity_df = _compute_stats(equity_df["Equity"], risk_free_rate=RISK_FREE_RATE)
        port_stats.update(pos_keep)

    n_trades = len(trade_df)
    n_buys = int((trade_df["action"] == "BUY").sum()) if n_trades > 0 else 0
    n_sells = int((trade_df["action"] == "SELL").sum()) if n_trades > 0 else 0
    hd = holding_days(trade_df) if n_trades > 0 else pd.Series(dtype=float)
    print(f"  Trades: {n_trades} total (BUY={n_buys}, SELL={n_sells}, "
          f"{n_trades / max(1, len(pred_dates)):.1f}/day)")
    if len(hd):
        print(f"  Holding days (FIFO): median={hd.median():.0f} mean={hd.mean():.1f} "
              f"max={hd.max():.0f}")

    test_dates = [d for d in pred_dates if d <= pd.Timestamp(test_end_date)]
    bench_df = compute_benchmark(full_ohlcv, test_dates, delist_info=delist_info,
                                 excluded_codes=excluded_codes)

    # ========================================================================
    # REPORT
    # ========================================================================
    print("\n" + "=" * 60)
    print("  PORTFOLIO RESULTS (v4 single score, daily rebalance)")
    print("=" * 60)

    print(f"\n  {'Total Return:':<22} {port_stats.get('total_return', 0):>+10.2%}")
    print(f"  {'CAGR:':<22} {port_stats.get('cagr', 0):>+10.2%}")
    print(f"  {'Sharpe Ratio:':<22} {port_stats.get('sharpe', 0):>10.2f}")
    print(f"  {'Sortino Ratio:':<22} {port_stats.get('sortino', 0):>10.2f}")
    print(f"  {'Max Drawdown:':<22} {port_stats.get('max_drawdown', 0):>+10.2%}")
    print(f"  {'Calmar Ratio:':<22} {port_stats.get('calmar', 0):>10.2f}")
    print(f"  {'Win Rate:':<22} {port_stats.get('win_rate', 0):>+10.2%}")
    print(f"  {'Avg Positions:':<22} {port_stats.get('avg_positions', float('nan')):>10.1f}")
    print(f"  {'Empty Days:':<22} {port_stats.get('n_days_empty', 0):>10}"
          f" / {port_stats.get('n_days', 0)}")
    print(f"  {'Trading Days:':<22} {port_stats.get('n_days', 0):>10}")
    print(f"  {'Total Trades:':<22} {n_trades:>10}")

    if not bench_df.empty and len(bench_df) > 0:
        bench_daily = bench_df['daily_ret'].dropna()
        bench_total = float(bench_df['equity'].iloc[-1] - 1.0)
        b_mean, b_std = float(bench_daily.mean()), float(bench_daily.std())
        bench_sharpe = float((b_mean * 252.0 - RISK_FREE_RATE) / (b_std * np.sqrt(252.0))) if b_std > 0 else 0.0
        bench_peak = bench_df['equity'].cummax()
        bench_maxdd = float(((bench_df['equity'] - bench_peak) / bench_peak).min())

        print(f"\n  --- Benchmark (Equal-Weight All {len(full_ohlcv)} stocks) ---")
        print(f"  {'Total Return:':<22} {bench_total:>+10.2%}")
        print(f"  {'Sharpe Ratio:':<22} {bench_sharpe:>10.2f}")
        print(f"  {'Max Drawdown:':<22} {bench_maxdd:>+10.2%}")
        print(f"\n  {'Excess Return:':<22} {port_stats.get('total_return', 0) - bench_total:>+10.2%}")

    # ---- save outputs（独立命名，勿覆写旧文件——report_strategy 还在消费旧对照）----
    bt_dir = get_backtest_dir()
    bt_dir.mkdir(parents=True, exist_ok=True)
    th_suffix = "_v4"
    eq_path = bt_dir / f"equity_lgb_combined_daily{th_suffix}_rebalance.csv"
    equity_df.to_csv(eq_path)
    if not bench_df.empty:
        bench_df.to_csv(bt_dir / "benchmark_combined.csv")
    if n_trades:
        trade_df.to_csv(bt_dir / f"trades_lgb_combined_daily{th_suffix}_rebalance.csv", index=False)
    print(f"\n  Equity curve saved to: {eq_path}")

    # ========================================================================
    # LONG-SHORT SIGNAL TEST
    # ========================================================================
    print(f"\n[5/5] Running long-short signal test (n={MAX_POSITIONS}/{MAX_POSITIONS}) ...")
    ls_stats, ls_equity = run_long_short(
        score, full_ohlcv,
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
        print(f"\n  --- Long-Short Results ({ls_stats.get('n_days_traded', 0)}/"
              f"{ls_stats.get('n_days', 0)} days traded) ---")
        print(f"  {'Total Return:':<22} {ls_stats.get('total_return', 0):>+10.2%}")
        print(f"  {'Sharpe Ratio:':<22} {ls_stats.get('sharpe', 0):>10.2f}")
        ls_equity.to_csv(bt_dir / "equity_lgb_combined_long_short.csv")
        print(f"\n  Long-short equity saved")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
