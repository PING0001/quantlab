# -*- coding: utf-8 -*-
"""
Dual-regression combined backtest - DAILY rebalancing, 开盘市价执行（v8）。

2026-08-25 简化合并：原 backtest/signals.py 模拟器并入本文件；limit 执行
语义（已判死，2026-08-21 七折 +3.9% 跑输基准）与长空诊断件（历史极不稳
定）物理删除，开盘市价为唯一语义。

Loads three model prediction parquets (open2d / 6d / 20d, 均 next_open 锚),
blends them into a single score = 0.4*pred_2d + 0.35*pred_6d + 0.25*pred_20d
(2026-08-22 用户口径 v8), then simulates daily:
  - buy : 次日开盘必成交（分数前 k 填空仓份数；开盘一字封板跳过）
  - sell: score 低于平移零点（Σwᵢ×calib_medianᵢ）时开盘市价卖出，
          否则持有（无目标价止盈）；开盘一字跌停顺延

T+1 constraint: stocks bought today cannot be sold tomorrow.
Price limits: regular ±10%, ST ±5% (from factor_values.IsST)。
退市持仓到达退市日强制清仓计零（与基准侧同口径）。

Run from project root:
    python -m backtest.run_lgb [--fold F*]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import warnings

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (DB_PATH, POOL_NAME, get_backtest_dir,
                    get_lgb_predictions_path, MODEL_CONFIGS, FOLDS, get_fold)

from strategies.lgb import rank_ic, ic_summary, combine_scores3
from strategies.labels import compute_median_open, compute_nextopen_limit_mask

# ============================================================================
# CONFIG
# ============================================================================
TEST_START = pd.Timestamp("2025-06-01")

PRED_COLS = {"open2d": "pred_label_open2d", "6d": "pred_label_6d", "20d": "pred_label_20d"}
W2D, W6D, W20D = 0.40, 0.35, 0.25  # v8：score = 0.4*p2d + 0.35*p6d + 0.25*p20d（三模型均 next_open 锚，2026-08-22 用户裁定）

MAX_POSITIONS = 10
REBALANCE_FREQ = 1          # 每日调仓（2026-08-21 用户裁定，spec §3.6）
CASH_PER_STOCK = 10000
COMMISSION = 0.0006
STAMP_DUTY = 0.0005
RISK_FREE_RATE = 0.025

warnings.filterwarnings("ignore")


# ============================================================================
# 组合模拟器（开盘市价语义；原 backtest/signals.py）
# ============================================================================
LOT_SIZE = 100


def _limit_pct(is_st) -> float:
    """Daily price limit ratio."""
    if isinstance(is_st, (np.floating, float)):
        is_st = float(is_st)
    return 0.05 if is_st else 0.10


# 2026-08-24 审计 #10：封板判定改纯比率口径（±0.05% 容差，与标签侧
# compute_nextopen_limit_mask 同源同参）。原实现 round(2) 价格网格打 qfq
# 价--qfq 不在原始价 0.01 网格上，adj≠1 的股票近板误判。
_FROZEN_TOL = 0.0005


def _frozen_up(low, prev_close, is_st) -> bool:
    return prev_close > 0 and low >= prev_close * (1.0 + _limit_pct(is_st) - _FROZEN_TOL)


def _frozen_down(high, prev_close, is_st) -> bool:
    return prev_close > 0 and high <= prev_close * (1.0 - _limit_pct(is_st) + _FROZEN_TOL)


def _compute_stats(equity_series, risk_free_rate=0.025):
    daily_ret = equity_series.pct_change().dropna()
    n_days = len(daily_ret)
    if n_days < 5:
        return {"n_days": n_days, "n_stocks_traded": 0}, pd.DataFrame()

    total_return = float(equity_series.iloc[-1] / equity_series.iloc[0] - 1.0)
    cagr = float((1.0 + total_return) ** (252.0 / n_days) - 1.0)

    mean_ret = float(daily_ret.mean())
    std_ret = float(daily_ret.std())
    annual_ret = mean_ret * 252.0
    annual_std = std_ret * np.sqrt(252.0)
    sharpe = float((annual_ret - risk_free_rate) / annual_std) if annual_std > 0 else 0.0

    downside = daily_ret[daily_ret < 0]
    d_std = float(downside.std()) if len(downside) > 0 else 0.0
    sortino = float(mean_ret / d_std * np.sqrt(252.0)) if len(downside) > 0 and d_std > 0 else 0.0

    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax.replace(0.0, np.nan)
    max_dd = float(drawdown.min()) if not drawdown.empty and not drawdown.isna().all() else 0.0

    calmar = float(cagr / abs(max_dd)) if max_dd != 0 else 0.0
    win_rate = float((daily_ret > 0).mean())

    stats = {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "n_days": n_days,
    }
    equity_df = pd.DataFrame({"Equity": equity_series, "Drawdown": drawdown})
    return stats, equity_df


def _prev_close(code: str, date, ohlcv_map: dict) -> float | None:
    """Get the previous trading day's close for a stock."""
    ohlcv = ohlcv_map[code]
    idx = ohlcv.index.get_loc(date)
    if idx == 0:
        return None
    return float(ohlcv.iloc[idx - 1]["Close"])


def run_portfolio_rebalance(
    predictions,
    ohlcv_map,
    test_start=None,
    max_positions=10,
    rebalance_freq=1,
    initial_cash_per_stock=10000,
    commission=0.0006,
    stamp_duty=0.0005,
    risk_free_rate=0.025,
    delist_info=None,
    sell_threshold=0.0,
):
    """Long-only backtest with periodic rebalancing, 开盘市价执行。

    On every *rebalance_freq* trading day:
      - Rank all eligible stocks by prediction score descending.
      - Buy top-k (k = 空仓份数) at next open（一字封板跳过，买单当日有效）。
      - Sell held positions whose score < sell_threshold at next open
        （一字跌停顺延；T+1 锁定除外）。

    predictions : pd.Series, MultiIndex (date, code), values = score
    ohlcv_map   : {code: DataFrame} with DatetimeIndex, columns Open/High/Low/Close/Volume/IsST
    delist_info : dict or None
        Mapping from code to delist_date (Timestamp). Stocks on or after
        their delist_date are excluded from buy candidates and forced to
        zero (audit #9, 与基准侧同口径).
    sell_threshold : float
        卖出零点（平移量）。L1 中位数输出的典型水平为负，绝对零点会让
        score<0 常态触发；阈值 = 融合权重加权的成分 calib_median。
    """
    all_dates = sorted(set().union(*(ohlcv.index for ohlcv in ohlcv_map.values())))
    total_cash = float(max_positions * initial_cash_per_stock)
    cash = total_cash

    positions = {}       # code -> {"shares", "cost", "entry_date"}
    sell_orders = set()  # codes with open-market sell orders for next day
    buy_orders = set()   # codes with open-market buy orders for next day (expire daily)
    buy_lock = set()     # T+1 sell prohibition

    nav_records = []
    pos_count_records = []   # (date, n_positions, cash)
    trades = []
    n_dates = len(all_dates)

    for i, date in enumerate(all_dates):
        # ---- open / high / low / close / isst snapshot ----
        open_map, high_map, low_map, close_map, isst_map = {}, {}, {}, {}, {}
        for code, ohlcv in ohlcv_map.items():
            if date in ohlcv.index:
                row = ohlcv.loc[date]
                op = float(row["Open"])
                hi = float(row["High"])
                lo = float(row["Low"])
                cl = float(row["Close"])
                st = int(row.get("IsST", 0)) if "IsST" in row else 0
                if not pd.isna(op) and op > 0:
                    open_map[code] = op
                    high_map[code] = hi
                    low_map[code] = lo
                    close_map[code] = cl
                    isst_map[code] = st

        # ================================================================
        # Phase 1: Execute sell orders at open（一字跌停顺延）
        # ================================================================
        deferred_sells = set()
        for code in list(sell_orders):
            ohlcv = ohlcv_map.get(code)
            if ohlcv is None or date not in ohlcv.index:
                deferred_sells.add(code)
                continue

            op = open_map.get(code)
            hi = high_map.get(code)
            if op is None:
                deferred_sells.add(code)
                continue

            prev_cl = _prev_close(code, date, ohlcv_map)
            if prev_cl is None or prev_cl <= 0:
                deferred_sells.add(code)
                continue

            is_st = isst_map.get(code, 0)
            if _frozen_down(hi, prev_cl, is_st):
                deferred_sells.add(code)
                continue

            pos = positions.get(code)
            if pos is None:
                continue

            shares = pos["shares"]
            cash += shares * op * (1.0 - commission - stamp_duty)
            trades.append({"date": date, "code": code, "action": "SELL",
                           "price": op, "shares": shares})
            del positions[code]

        sell_orders = {c for c in sell_orders if c in deferred_sells}

        # ================================================================
        # Phase 2: Execute buy orders at open（一字涨停跳过）
        # ================================================================
        buy_slots_remaining = max_positions - len(positions)

        for code in list(buy_orders):
            ohlcv = ohlcv_map.get(code)
            if ohlcv is None or date not in ohlcv.index:
                continue

            op = open_map.get(code)
            lo = low_map.get(code)
            if op is None:
                continue

            prev_cl = _prev_close(code, date, ohlcv_map)
            if prev_cl is None or prev_cl <= 0:
                continue

            is_st = isst_map.get(code, 0)
            if _frozen_up(lo, prev_cl, is_st):
                continue

            cost_unit = op * (1.0 + commission)
            target_cost = min(initial_cash_per_stock, cash / max(1, buy_slots_remaining))
            shares = int(target_cost / cost_unit / LOT_SIZE) * LOT_SIZE
            if shares < LOT_SIZE:
                continue

            cost = shares * cost_unit
            if cost > cash:
                continue

            cash -= cost
            positions[code] = {"shares": shares, "cost": op, "entry_date": date}
            trades.append({"date": date, "code": code, "action": "BUY",
                           "price": op, "shares": shares})
            buy_slots_remaining -= 1

        # Unfilled buy orders expire at end of day: fresh orders are only
        # placed on rebalance evenings.
        buy_orders.clear()

        # 2026-08-24 审计 #9：退市持仓强制清仓计零。原实现：预测行随退市消失
        # ->永不挂卖单->按最后收盘价永续估值；基准侧同股退市直接归零--两侧
        # 口径相反。统一为：到达退市日的持仓移除且无现金回流（保守计零）。
        if delist_info:
            for code in list(positions.keys()):
                if code in delist_info and date >= delist_info[code]:
                    del positions[code]

        buy_lock = {code for code, pos in positions.items() if pos["entry_date"] == date}

        # ================================================================
        # Phase 3: Evening - order generation (ONLY on rebalance days)
        # ================================================================
        is_rebalance = (i % rebalance_freq == 0)
        next_date = all_dates[i + 1] if i + 1 < n_dates else None

        if is_rebalance and next_date is not None and date in predictions.index.get_level_values("date"):
            try:
                today_pred = predictions.xs(date, level="date")
            except KeyError:
                today_pred = pd.Series(dtype=float)

            if not today_pred.empty:
                # --- 3a. SELL：score < 阈值的持仓挂次日开盘市价卖单 ---
                new_sells = set()
                for code, pos in positions.items():
                    if code in buy_lock:
                        continue
                    pred_val = today_pred.get(code)
                    if pred_val is None or pd.isna(pred_val):
                        continue
                    prev_cl = close_map.get(code)
                    if prev_cl is None or prev_cl <= 0:
                        continue
                    if float(pred_val) < sell_threshold:
                        new_sells.add(code)

                # --- 3b. BUY：k = 空仓份数，预测排名前 k（剔除已持有/ST/退市）---
                candidates = today_pred.dropna()
                st_codes = {c for c in candidates.index if isst_map.get(c, 0) == 1}
                candidates = candidates[~candidates.index.isin(st_codes)]
                if delist_info:
                    delisted = {c for c in candidates.index
                                if c in delist_info and date >= delist_info[c]}
                    candidates = candidates[~candidates.index.isin(delisted)]
                candidates = candidates.sort_values(ascending=False)

                held_codes = set(positions.keys())
                k = max_positions - len(positions)

                new_buys = set()
                for code in candidates.index:
                    if len(new_buys) >= k:
                        break
                    if code in held_codes:
                        continue
                    prev_cl = close_map.get(code)
                    if prev_cl is None or prev_cl <= 0:
                        continue
                    new_buys.add(code)

                sell_orders = new_sells
                buy_orders = new_buys

        # ================================================================
        # Phase 4: NAV at close
        # ================================================================
        nav = cash
        for code, pos in positions.items():
            if delist_info and code in delist_info and date >= delist_info[code]:
                continue          # 退市持仓计零（与基准侧口径一致）
            cl = close_map.get(code)
            if cl is None:
                # Suspended: use last known close price
                ohlcv = ohlcv_map.get(code)
                if ohlcv is not None and date in ohlcv.index:
                    pass  # should have been in close_map
                elif ohlcv is not None:
                    # Find last trading day with a close
                    prev_dates = ohlcv.index[ohlcv.index < date]
                    if len(prev_dates) > 0:
                        cl = float(ohlcv.loc[prev_dates[-1], "Close"])
                if cl is None:
                    continue
            nav += pos["shares"] * cl

        nav_records.append((date, nav))
        pos_count_records.append((date, len(positions), cash))

    # ---- stats from nav ----
    nav_series = pd.Series(
        [v for _, v in nav_records],
        index=pd.DatetimeIndex([d for d, _ in nav_records]),
    ).sort_index()
    if len(nav_series) < 2:
        return {}, pd.DataFrame(), pd.DataFrame()

    if test_start is not None:
        nav_series = nav_series[nav_series.index >= pd.Timestamp(test_start)]

    stats, equity_df = _compute_stats(nav_series, risk_free_rate=risk_free_rate)
    trade_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["date", "code", "action", "price", "shares"])
    stats["n_trades"] = len(trades)

    # 仓位/空仓统计（按 test_start 之后的交易日口径）
    pos_df = pd.DataFrame(pos_count_records, columns=["date", "n_pos", "cash"]).set_index("date")
    if test_start is not None and not pos_df.empty:
        pos_df = pos_df[pos_df.index >= pd.Timestamp(test_start)]
    if not pos_df.empty:
        stats["avg_positions"] = float(pos_df["n_pos"].mean())
        stats["n_days_empty"] = int((pos_df["n_pos"] == 0).sum())
        stats["n_days_full"] = int((pos_df["n_pos"] >= max_positions).sum())
        stats["n_days"] = int(len(pos_df))
    return stats, equity_df, trade_df


def compute_benchmark_pit(ohlcv_map, test_dates, reset_points, delist_info=None):
    """池时点化基准：半年重置等权（2026-08-24 用户裁定 A）。

    测试首日与各档生效日重置：等权买入当期池成员，段内买入持有（权重
    随价格漂移）；入场过滤 = 各股段内首个交易日 IsST=1 不入（时点）；
    退市日之后贡献归零；停牌按段内最后收盘价延续。段间 NAV 链乘。

    reset_points: [(date_str, members)] 升序（pools.membership.reset_points）。
    返回 DataFrame[daily_ret, equity]（index=test_dates）。
    """
    if not test_dates or not ohlcv_map or not reset_points:
        return pd.DataFrame()

    delist_dates = {c: pd.Timestamp(d) for c, d in (delist_info or {}).items()}
    pts = sorted(reset_points, key=lambda x: x[0])
    td = pd.DatetimeIndex(test_dates)

    equity_vals = []
    eq_cum = 1.0
    for i, (pt_date, members) in enumerate(pts):
        seg_start = pd.Timestamp(pt_date)
        seg_end = pd.Timestamp(pts[i + 1][0]) if i + 1 < len(pts) else td[-1] + pd.Timedelta(days=1)
        seg_dates = td[(td >= seg_start) & (td < seg_end)]
        if len(seg_dates) == 0:
            continue

        # 段内入选：首个交易日有行情且非 ST、未退市
        picks = {}
        for code in members:
            ohlcv = ohlcv_map.get(code)
            if ohlcv is None:
                continue
            if code in delist_dates and seg_dates[0] >= delist_dates[code]:
                continue
            valid = ohlcv[(ohlcv.index >= seg_dates[0]) & (ohlcv.index <= seg_dates[-1])]
            if valid.empty:
                continue
            first = valid.iloc[0]
            if int(first.get("IsST", 0) or 0) == 1:
                continue
            base = float(first["Close"])
            if base <= 0:
                continue
            picks[code] = (valid, base)
        if not picks:
            continue

        for dt in seg_dates:
            total = 0.0
            for code, (valid, base) in picks.items():
                upto = valid[valid.index <= dt]
                if upto.empty:
                    continue    # 段首前无行（不应发生，防御）
                if code in delist_dates and dt >= delist_dates[code]:
                    continue    # 退市后归零
                total += float(upto.iloc[-1]["Close"]) / base
            seg_nav = total / len(picks)
            equity_vals.append((dt, eq_cum * seg_nav))
        eq_cum = equity_vals[-1][1]

    nav = pd.Series(dict(equity_vals)).sort_index()
    if len(nav) < 2:
        return pd.DataFrame()
    daily_ret = nav.pct_change().dropna()
    equity = nav / nav.iloc[0]
    return pd.DataFrame({"daily_ret": daily_ret, "equity": equity})


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


def load_predictions(fold: str | None = None) -> dict[str, pd.Series]:
    preds = {}
    for m, col in PRED_COLS.items():
        path = get_lgb_predictions_path(m, fold=fold)
        if not path.exists():
            print(f"  ERROR: predictions for {m} not found at {path}")
            print(f"  Run: python run_lgb.py --model all"
                  f"{' --fold ' + fold if fold else ''}")
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
    import argparse
    parser = argparse.ArgumentParser(description="Dual-regression combined backtest")
    parser.add_argument("--fold", choices=sorted(FOLDS), default=None,
                        help="滚动折 CV：预测走 fold 路径，TEST_START=折 test_start，"
                             "输出 backtest/{pool}/folds/{fid}/")
    args = parser.parse_args()

    fold = args.fold
    test_start = pd.Timestamp(get_fold(fold)[0]) if fold else TEST_START

    print("=" * 60)
    print(f"  Dual-Regression Backtest - DAILY, v8 score = 0.4*p2d + 0.35*p6d + 0.25*p20d"
          f"{f' | fold={fold}' if fold else ''} | exec=market_open")
    print(f"  Pool: {POOL_NAME} | label anchor: next_open | "
          f"开盘市价（买=开盘必成交，卖=score<零点）")
    print("=" * 60)

    # ---- 1. Load + combine predictions ----
    print("\n[1/4] Loading predictions ...")
    preds = load_predictions(fold)

    # ---- 卖出零点平移（2026-08-22 审计 F8）----
    # L1 中位数输出下三成分典型水平为负（池内典型股票远期中位收益为负），
    # score<0 会常态触发卖出。融合排序不变，卖出阈值 = Σ w_i × calib_median_i
    # （训练尾段横截面中位数，来自各模型 meta；缺失时回退 0 = 旧行为）。
    # parquet 里的预测保持原始诚实涨幅，不平移。
    sell_zero = 0.0
    med_parts = []
    for m, w in (("open2d", W2D), ("6d", W6D), ("20d", W20D)):
        p = get_lgb_predictions_path(m, fold=fold)
        meta_path = p.with_name(p.name.replace(".parquet", "_meta.json"))
        if meta_path.exists():
            try:
                med = json.loads(meta_path.read_text())["results"].get("calib_median")
            except Exception:
                med = None
            if med is not None:
                sell_zero += w * float(med)
                med_parts.append(f"{m}={float(med):+.5f}")
    print(f"  sell zero-point: {sell_zero:+.5f} "
          f"({'; '.join(med_parts) if med_parts else 'meta 缺 calib_median，回退 0'})")

    score = combine_scores3(preds["open2d"], preds["6d"], preds["20d"],
                            w2d=W2D, w6=W6D, w20=W20D)

    n_dates = score.index.get_level_values("date").nunique()
    print(f"  combined: {len(score)} rows, {n_dates} dates")
    print(f"  score quantiles: "
          f"1%={score.quantile(0.01):+.4f} 50%={score.quantile(0.5):+.4f} "
          f"99%={score.quantile(0.99):+.4f} |0.999|={abs(score).quantile(0.999):.4f}")
    # 量纲 sanity：score 必须是收益量纲，否则执行逻辑失真
    if abs(score).quantile(0.999) > 0.35:
        print("  ERROR: score magnitude exceeds plausible return range - "
              "check combine inputs")
        sys.exit(1)

    # ---- 2. Load OHLCV + metadata ----
    print(f"\n[2/4] Loading OHLCV + metadata ...")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    from pools.membership import reset_points as pool_reset_points
    pred_end = str(score.index.get_level_values("date").max().date())
    bench_reset_pts = pool_reset_points(str(test_start.date()), pred_end)
    bench_codes = sorted(set().union(*[m for _, m in bench_reset_pts]))
    pred_codes = sorted(score.index.get_level_values("code").unique())
    ohlcv_map = load_ohlcv_map(con, pred_codes)
    full_ohlcv = load_ohlcv_map(con, bench_codes)
    print(f"  OHLCV: {len(ohlcv_map)} prediction stocks, {len(full_ohlcv)} "
          f"bench stocks（池时点化：窗口内成员并集，半年重置 {len(bench_reset_pts)} 段）")

    # 2026-08-24 用户裁定：名称快照层退役--ST/退市判定统一走时点口径
    # （日度 IsST 因子 + delist_info 日期），在模拟器候选过滤与退市清仓处执行
    print("  ST/退 defense: 日度 IsST + delist 日期（时点口径，模拟器内过滤）")

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
    print(f"\n[3/4] IC reference ...")
    con_r = duckdb.connect(str(DB_PATH), read_only=True)
    ic_codes = sorted(set(pred_codes) | set(bench_codes))   # 池时点化：预测 ∪ 基准成员
    placeholders = ",".join(["?"] * len(ic_codes))
    kline = con_r.execute(
        f"SELECT code, date, open, close FROM daily_kline WHERE code IN ({placeholders}) ORDER BY code, date",
        ic_codes,
    ).fetchdf()

    try:
        st_df = con_r.execute(
            f"SELECT code, date, IsST FROM factor_values WHERE code IN ({placeholders})",
            ic_codes,
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
    # 2026-08-24 语义变更：训练排斥仅作用于训练集，预测帧含全量行；
    # 此处计数非零属预期；掩码用于 IC 块的 safe 过滤
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
    print(f"\n[4/4] Running DAILY rebalance backtest "
          f"(max_pos={MAX_POSITIONS}, rebalance_freq={REBALANCE_FREQ}) ...")
    port_stats, equity_df, trade_df = run_portfolio_rebalance(
        score, ohlcv_map,
        test_start=str(test_start.date()),
        max_positions=MAX_POSITIONS,
        rebalance_freq=REBALANCE_FREQ,
        initial_cash_per_stock=CASH_PER_STOCK,
        commission=COMMISSION,
        stamp_duty=STAMP_DUTY,
        risk_free_rate=RISK_FREE_RATE,
        delist_info=delist_info,
        sell_threshold=sell_zero,
    )

    if not equity_df.empty and test_end_date is not None:
        cutoff = pd.Timestamp(test_end_date)
        pos_keep = {k: port_stats[k] for k in
                    ("avg_positions", "n_days_empty", "n_days_full", "n_days")
                    if k in port_stats}
        equity_df = equity_df[equity_df.index <= cutoff]
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
    bench_df = compute_benchmark_pit(full_ohlcv, test_dates,
                                     reset_points=bench_reset_pts,
                                     delist_info=delist_info)

    # ========================================================================
    # REPORT
    # ========================================================================
    print("\n" + "=" * 60)
    print("  PORTFOLIO RESULTS (v8 single score, daily rebalance, market open)")
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

    # ---- save outputs ----
    bt_dir = get_backtest_dir(fold=fold)
    bt_dir.mkdir(parents=True, exist_ok=True)
    if fold:
        th_suffix = "_market"   # folds/{fid}/equity_lgb_combined_daily_market_rebalance.csv
    else:
        th_suffix = "_v8mo"
    eq_path = bt_dir / f"equity_lgb_combined_daily{th_suffix}_rebalance.csv"
    equity_df.to_csv(eq_path)
    if not bench_df.empty:
        bench_df.to_csv(bt_dir / "benchmark_combined.csv")
    if n_trades:
        trade_df.to_csv(bt_dir / f"trades_lgb_combined_daily{th_suffix}_rebalance.csv", index=False)
    print(f"\n  Equity curve saved to: {eq_path}")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
