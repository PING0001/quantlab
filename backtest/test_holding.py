# -*- coding: utf-8 -*-
"""
Fixed 5-day holding test — validates model predictions against actual returns.

Buy: same overnight limit-order rules as main engine, NO entry threshold.
Sell: forced 5 trading days, at close price (no exit logic, no deferral).

Output: per-trade detail + pred-bucket aggregate statistics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_PATH, POOL_NAME, get_pool_codes, get_lgb_predictions_path

LOT_SIZE = 100

# ---- reused price-limit helpers ----
def _limit_pct(is_st):
    return 0.05 if is_st else 0.10

def _limit_up(prev_close, is_st):
    return prev_close * (1.0 + _limit_pct(is_st))

def _is_frozen_up(low, limit_up_px):
    return low >= limit_up_px

# ---- limit price ----
def _buy_limit(prev_close, pred_score, auction_buffer=0.01):
    return prev_close * (1.0 + pred_score - auction_buffer)


def run_holding_test(
    predictions,
    ohlcv_map,
    excluded_codes=None,
    max_positions=10,
    auction_buffer=0.01,
    holding_days=5,
    initial_cash_per_stock=10000,
    commission=0.0006,
):
    """Run fixed-holding test and return per-trade DataFrame + stats."""
    if excluded_codes is None:
        excluded_codes = set()

    all_dates = sorted(set().union(*(ohlcv.index for ohlcv in ohlcv_map.values())))
    n_dates = len(all_dates)
    cash = float(max_positions * initial_cash_per_stock)

    positions = {}       # code -> {shares, cost, buy_date, sell_date, pred}
    buy_orders = {}      # code -> limit_price  (for next day)
    trades = []

    for i, date in enumerate(all_dates):
        # ---- snapshot today's prices ----
        open_map, high_map, low_map, close_map, isst_map = {}, {}, {}, {}, {}
        for code, ohlcv in ohlcv_map.items():
            if date in ohlcv.index:
                row = ohlcv.loc[date]
                op = float(row["Open"])
                if not pd.isna(op) and op > 0:
                    open_map[code] = op
                    high_map[code] = float(row["High"])
                    low_map[code] = float(row["Low"])
                    close_map[code] = float(row["Close"])
                    isst_map[code] = int(row.get("IsST", 0)) if "IsST" in row else 0

        def _prev_close(code):
            ohlcv = ohlcv_map[code]
            idx = ohlcv.index.get_loc(date)
            if idx == 0:
                return None
            return float(ohlcv.iloc[idx - 1]["Close"])

        # ====================================================================
        # Phase 1: Sell matured positions at close
        # ====================================================================
        for code, pos in list(positions.items()):
            if pos["sell_date"] == date:
                cl = close_map.get(code)
                if cl is None:
                    continue
                shares = pos["shares"]
                cash += shares * cl * (1.0 - commission)
                trades.append({
                    "buy_date": pos["buy_date"],
                    "sell_date": date,
                    "code": code,
                    "buy_price": pos["cost"],
                    "sell_price": cl,
                    "shares": shares,
                    "pred_5d": pos["pred"],
                })
                del positions[code]

        # ====================================================================
        # Phase 2: Execute buy orders (placed last evening)
        # ====================================================================
        for code, (limit_price, pred_val) in list(buy_orders.items()):
            op = open_map.get(code)
            lo = low_map.get(code)
            if op is None:
                continue
            prev_cl = _prev_close(code)
            if prev_cl is None or prev_cl <= 0:
                continue

            is_st = isst_map.get(code, 0)
            limit_up_px = _limit_up(prev_cl, is_st)
            if _is_frozen_up(lo, limit_up_px):
                continue

            if op <= limit_price:
                fill_px = op
            elif lo <= limit_price:
                fill_px = limit_price
            else:
                continue

            cost_unit = fill_px * (1.0 + commission)
            shares = int(initial_cash_per_stock / cost_unit / LOT_SIZE) * LOT_SIZE
            if shares < LOT_SIZE:
                continue

            cost = shares * cost_unit
            if cost > cash:
                continue

            cash -= cost

            # sell date: buy_date + holding_days
            try:
                idx = all_dates.index(date)
                sell_idx = min(idx + holding_days, n_dates - 1)
                sell_date = all_dates[sell_idx]
            except (ValueError, IndexError):
                sell_date = all_dates[-1]

            positions[code] = {
                "shares": shares,
                "cost": fill_px,
                "buy_date": date,
                "sell_date": sell_date,
                "pred": pred_val,
            }

        buy_orders.clear()

        # ====================================================================
        # Phase 3: Evening — generate buy orders for tomorrow
        # ====================================================================
        next_date = all_dates[i + 1] if i + 1 < n_dates else None

        if next_date is not None and date in predictions.index.get_level_values("date"):
            try:
                today_pred = predictions.xs(date, level="date")
            except KeyError:
                today_pred = pd.Series(dtype=float)

            available = max(max_positions - len(positions) - len(buy_orders), 0)
            if available <= 0:
                continue

            held = set(positions.keys())
            st_set = {c for c in today_pred.index if isst_map.get(c, 0) == 1}

            candidates = today_pred[~today_pred.index.isin(held)]
            candidates = candidates[~candidates.index.isin(excluded_codes)]
            candidates = candidates[~candidates.index.isin(st_set)]
            candidates = candidates.sort_values(ascending=False)

            buy_orders = {}
            for code in candidates.head(available).index:
                prev_cl = close_map.get(code)
                if prev_cl is None or prev_cl <= 0:
                    continue
                pred_val = float(today_pred[code])
                buy_orders[code] = (_buy_limit(prev_cl, pred_val, auction_buffer), pred_val)

    # ========================================================================
    # Aggregate stats
    # ========================================================================
    if not trades:
        return pd.DataFrame(), pd.DataFrame()

    trade_df = pd.DataFrame(trades)
    trade_df["return"] = trade_df["sell_price"] / trade_df["buy_price"] - 1.0
    trade_df["return_net"] = trade_df["return"] - commission * 2  # approx round-trip

    # pred bucket stats
    bins = [-999, -0.02, -0.01, 0.0, 0.01, 0.02, 999]
    bucket_stats = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (trade_df["pred_5d"] >= lo) & (trade_df["pred_5d"] < hi)
        n = mask.sum()
        if n == 0:
            continue
        sub = trade_df[mask]
        bucket_stats.append({
            "pred_range": f"[{lo:.2f}, {hi:.2f})",
            "n_trades": n,
            "mean_ret_gross": float(sub["return"].mean()),
            "mean_ret_net": float(sub["return_net"].mean()),
            "win_rate": float((sub["return"] > 0).mean()),
        })
    bucket_df = pd.DataFrame(bucket_stats)

    trade_df["pnl"] = trade_df["shares"] * (trade_df["sell_price"] - trade_df["buy_price"])
    print_stats = {
        "n_trades": len(trade_df),
        "total_pnl": float(trade_df["pnl"].sum()),
        "mean_ret_gross": float(trade_df["return"].mean()),
        "mean_ret_net": float(trade_df["return_net"].mean()),
        "win_rate": float((trade_df["return"] > 0).mean()),
    }

    return trade_df, bucket_df, print_stats


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("  Fixed 5-Day Holding Test (no entry threshold)")
    print(f"  Pool: {POOL_NAME}")
    print("=" * 60)

    pred_path = get_lgb_predictions_path()
    if not pred_path.exists():
        print(f"ERROR: {pred_path} not found. Run: python run_lgb.py")
        return

    print("\n[1/3] Loading predictions ...")
    preds = pd.read_parquet(pred_path)
    preds_5d = preds["pred_5d"]
    pred_codes = sorted(preds_5d.index.get_level_values("code").unique())
    print(f"  {len(preds_5d)} rows, {preds_5d.index.get_level_values('date').nunique()} dates, {len(pred_codes)} stocks")

    print("\n[2/3] Loading OHLCV + ST info ...")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    pool_codes = get_pool_codes()
    placeholders = ",".join(["?"] * len(pool_codes))

    # OHLCV for all pool codes
    df = con.execute(
        f"SELECT code, date, open, high, low, close, volume, pct_chg "
        f"FROM daily_kline WHERE code IN ({placeholders}) "
        f"ORDER BY code, date",
        pool_codes,
    ).fetchdf()
    df["date"] = pd.to_datetime(df["date"])

    # Merge IsST
    try:
        isst = con.execute(
            f"SELECT code, date, IsST FROM factor_values WHERE code IN ({placeholders})",
            pool_codes,
        ).fetchdf()
        if not isst.empty:
            isst["date"] = pd.to_datetime(isst["date"])
            df = df.merge(isst, on=["code", "date"], how="left")
        else:
            df["IsST"] = 0
    except Exception:
        df["IsST"] = 0
    df["IsST"] = df["IsST"].fillna(0).astype(int)

    ohlcv_map = {}
    for code, grp in df.groupby("code"):
        grp = grp.set_index("date")
        grp = grp.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
            "pct_chg": "Pct_chg",
        })
        grp["IsST"] = grp.get("IsST", 0)
        ohlcv_map[code] = grp[["Open", "High", "Low", "Close", "Volume", "Pct_chg", "IsST"]]

    # Excluded codes (ST/退 in name)
    excluded = set()
    try:
        name_df = con.execute(
            f"SELECT code, name FROM stock_info WHERE code IN ({placeholders})",
            pool_codes,
        ).fetchdf()
        for _, r in name_df.iterrows():
            n = r["name"]
            if isinstance(n, str) and ("ST" in n or "退" in n):
                excluded.add(r["code"])
    except Exception:
        pass
    con.close()
    print(f"  OHLCV: {len(ohlcv_map)} stocks, Excluded: {len(excluded)}")

    print("\n[3/3] Running holding test ...")
    trade_df, bucket_df, stats = run_holding_test(
        preds_5d, ohlcv_map,
        excluded_codes=excluded,
        max_positions=10,
        auction_buffer=0.01,
        holding_days=5,
        initial_cash_per_stock=10000,
        commission=0.0006,
    )

    print("\n" + "=" * 60)
    print("  HOLDING TEST RESULTS")
    print("=" * 60)
    print(f"  Total Trades:      {stats['n_trades']:>10}")
    print(f"  Total PnL:         {stats['total_pnl']:>+10.0f}")
    print(f"  Mean Ret (gross):  {stats['mean_ret_gross']:>+10.4%}")
    print(f"  Mean Ret (net):    {stats['mean_ret_net']:>+10.4%}")
    print(f"  Win Rate:          {stats['win_rate']:>10.2%}")

    if not bucket_df.empty:
        print(f"\n  --- By Pred Bucket ---")
        print(f"  {'Range':<18} {'Trades':>7} {'Gross':>9} {'Net':>9} {'Win%':>7}")
        for _, r in bucket_df.iterrows():
            print(f"  {r['pred_range']:<18} {r['n_trades']:>7} {r['mean_ret_gross']:>+8.4%} {r['mean_ret_net']:>+8.4%} {r['win_rate']:>7.2%}")

    if not trade_df.empty:
        out_path = Path(__file__).parent / "holding_test_trades.csv"
        trade_df.to_csv(out_path, index=False)
        print(f"\n  Trade details saved to: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
