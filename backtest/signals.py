# -*- coding: utf-8 -*-
"""Daily long-only backtest with overnight limit orders.

T+1 constraint: stocks bought today cannot be sold tomorrow.

Entry (buy):
  trigger   pred > entry_threshold (threshold 模式) / top-N
  limit     prev_close * (1 + pred - auction_buffer)
            （2026-08-21 用户二次裁定：保留此加减法口径，弃目标价×(1-3%)
             乘法折让版——两者数值相近，此版更直观）
  fill      open <= limit -> fill at open (call auction)
            low  <= limit -> fill at limit (intraday)
            else         -> order expires (no chase)
            sealed limit-up -> skip

Exit (sell):
  EVERY evening, for EVERY held position (except T+1 locked):
    limit   prev_close * (1 + pred + sell_markup)
    fill    open >= limit -> fill at open (call auction)
            high >= limit -> fill at limit (intraday)
            otherwise      -> defer (recalculated next evening)
            sealed limit-down -> defer

Price limits:
  regular   +/-10%
  ST        +/-5%   (from factor_values.IsST)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOT_SIZE = 100


# ============================================================================
# Price limit helpers
# ============================================================================

def _limit_pct(is_st) -> float:
    """Daily price limit ratio."""
    if isinstance(is_st, (np.floating, float)):
        is_st = float(is_st)
    return 0.05 if is_st else 0.10


def _limit_up(prev_close, is_st) -> float:
    return round(prev_close * (1.0 + _limit_pct(is_st)), 2)


def _limit_down(prev_close, is_st) -> float:
    return round(prev_close * (1.0 - _limit_pct(is_st)), 2)


def _is_frozen_up(low, limit_up_price) -> bool:
    return low >= limit_up_price


def _is_frozen_down(high, limit_down_price) -> bool:
    return high <= limit_down_price


# ============================================================================
# Limit price formulas
# ============================================================================

def _buy_limit(prev_close, pred_score: float, auction_buffer: float = 0.025) -> float:
    """Overnight buy limit: prev_close * (1 + pred - buffer).

    期望次日集合竞价或盘中低触成交，跳空高于限价则不追。
    （2026-08-21 用户试过目标价×(1-3%) 乘法口径后选择保留本式）"""
    return prev_close * (1.0 + pred_score - auction_buffer)


def _sell_limit(prev_close, pred_score: float, sell_markup: float = 0.0005) -> float:
    """Overnight sell limit: prev_close * (1 + pred + markup)."""
    return prev_close * (1.0 + pred_score + sell_markup)


# ============================================================================
# Stats
# ============================================================================

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


# ============================================================================
# Main backtest loop
# ============================================================================

def _prev_close(code: str, date, ohlcv_map: dict) -> float | None:
    """Get the previous trading day's close for a stock.

    Extracted from the main loop to avoid re-defining a function object
    on every iteration and to eliminate closure over mutable loop variables.
    """
    ohlcv = ohlcv_map[code]
    idx = ohlcv.index.get_loc(date)
    if idx == 0:
        return None
    return float(ohlcv.iloc[idx - 1]["Close"])


def run_portfolio(
    predictions,
    ohlcv_map,
    test_start=None,
    max_positions=10,
    entry_threshold=0.03,
    auction_buffer=0.025,
    sell_markup=0.0005,
    excluded_codes=None,
    initial_cash_per_stock=10000,
    commission=0.0003,
    stamp_duty=0.0005,
    risk_free_rate=0.025,
    delist_info=None,
):
    """Long-only backtest with overnight limit orders and call auction execution.

    DEPRECATED（bench 2026-08）：entry_threshold 语义依赖收益量纲分数，
    双模型 rank/exec 架构下不适配，主路径为 run_portfolio_rebalance；
    保留仅供历史对照，勿在新代码中使用。

    predictions : pd.Series, MultiIndex (date, code), values = pred_5d scores
    ohlcv_map   : {code: DataFrame}, columns must include
                  Open, High, Low, Close, Volume, IsST
    delist_info : dict or None
        Mapping from code to delist_date (Timestamp). Stocks on or after
        their delist_date are excluded from buy candidates.
    """
    # ---- trading calendar ----
    all_dates = sorted(set().union(*(ohlcv.index for ohlcv in ohlcv_map.values())))
    total_cash = float(max_positions * initial_cash_per_stock)
    cash = total_cash
    if excluded_codes is None:
        excluded_codes = set()

    positions = {}       # code -> {"shares", "cost", "entry_date"}
    sell_orders = {}     # code -> limit_price  (for next trading day)
    buy_orders = {}      # code -> limit_price  (for next trading day)
    buy_lock = set()     # codes bought today (T+1 sell prohibition)

    nav_records = []
    trades = []

    n_dates = len(all_dates)

    for i, date in enumerate(all_dates):
        # --- open / high / low snapshot ---
        open_map = {}
        high_map = {}
        low_map = {}
        close_map = {}
        isst_map = {}
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

        # ====================================================================
        # Phase 1: Execute sell orders (placed last evening)
        # ====================================================================
        deferred_sells = set()
        for code, limit_price in list(sell_orders.items()):
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
            limit_dn = _limit_down(prev_cl, is_st)

            # sealed limit-down -> defer
            if _is_frozen_down(hi, limit_dn):
                deferred_sells.add(code)
                continue

            # auction fill
            if op >= limit_price:
                fill_px = op
            elif hi >= limit_price:
                fill_px = limit_price
            else:
                deferred_sells.add(code)
                continue

            # execute sell
            pos = positions.get(code)
            if pos is None:
                continue

            shares = pos["shares"]
            cash += shares * fill_px * (1.0 - commission - stamp_duty)
            trades.append({"date": date, "code": code, "action": "SELL",
                           "price": fill_px, "shares": shares})
            del positions[code]

        # remove executed sells from order book
        remaining = {}
        for code, limit_price in sell_orders.items():
            if code in deferred_sells:
                remaining[code] = limit_price
        sell_orders = remaining

        # ====================================================================
        # Phase 2: Execute buy orders (placed last evening)
        # ====================================================================
        n_buy_orders = len(buy_orders)
        buy_slots_remaining = max_positions - len(positions)

        for code, limit_price in list(buy_orders.items()):
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
            limit_up_px = _limit_up(prev_cl, is_st)

            # sealed limit-up -> skip
            if _is_frozen_up(lo, limit_up_px):
                continue

            # auction fill
            if op <= limit_price:
                fill_px = op
            elif lo <= limit_price:
                fill_px = limit_price
            else:
                continue

            # execute buy: target initial_cash_per_stock per position
            cost_unit = fill_px * (1.0 + commission)
            target_cost = min(initial_cash_per_stock, cash / max(1, buy_slots_remaining))
            shares = int(target_cost / cost_unit / LOT_SIZE) * LOT_SIZE
            if shares < LOT_SIZE:
                continue

            cost = shares * cost_unit
            if cost > cash:
                continue

            cash -= cost
            positions[code] = {"shares": shares, "cost": fill_px, "entry_date": date}
            trades.append({"date": date, "code": code, "action": "BUY",
                           "price": fill_px, "shares": shares})
            buy_slots_remaining -= 1

        buy_orders.clear()

        # Track stocks bought today (T+1 sell prohibition)
        buy_lock = {code for code, pos in positions.items() if pos["entry_date"] == date}

        # ====================================================================
        # Phase 3: Evening — generate orders for tomorrow
        # ====================================================================
        next_date = all_dates[i + 1] if i + 1 < n_dates else None

        if next_date is not None and date in predictions.index.get_level_values("date"):
            try:
                today_pred = predictions.xs(date, level="date")
            except KeyError:
                today_pred = pd.Series(dtype=float)

            # --- 3a. Build sell orders for ALL positions (except T+1 locked) ---
            # Every evening, every held stock gets a new sell limit order.
            # Previously deferred sells are automatically recalculated here.
            new_sells = {}
            for code, pos in positions.items():
                if code in buy_lock:
                    continue
                pred_val = today_pred.get(code)
                if pred_val is None or pd.isna(pred_val):
                    continue
                prev_cl = close_map.get(code)
                if prev_cl is None or prev_cl <= 0:
                    continue
                new_sells[code] = _sell_limit(prev_cl, float(pred_val), sell_markup)

            # --- 3b. Buy candidates ---
            effective_slots = max_positions - len(positions) + len(new_sells)
            held_codes = set(positions.keys())

            candidates = today_pred[today_pred > entry_threshold]
            candidates = candidates[~candidates.index.isin(held_codes)]
            # Exclude ST/delisting stocks (name-based)
            candidates = candidates[~candidates.index.isin(excluded_codes)]
            # Exclude stocks that are ST on today's date
            st_codes = {code for code in candidates.index if isst_map.get(code, 0) == 1}
            candidates = candidates[~candidates.index.isin(st_codes)]
            # Exclude stocks past their delist_date
            if delist_info:
                delisted = {c for c in candidates.index
                            if c in delist_info and date >= delist_info[c]}
                candidates = candidates[~candidates.index.isin(delisted)]
            candidates = candidates.sort_values(ascending=False)

            new_buys = {}
            for code in candidates.head(effective_slots).index:
                prev_cl = close_map.get(code)
                if prev_cl is None or prev_cl <= 0:
                    continue
                pred_val = float(today_pred[code])
                new_buys[code] = _buy_limit(prev_cl, pred_val, auction_buffer)

            sell_orders = new_sells
            buy_orders = new_buys

        # ====================================================================
        # Phase 4: NAV at close
        # ====================================================================
        nav = cash
        for code, pos in positions.items():
            cl = close_map.get(code)
            if cl is None:
                continue
            nav += pos["shares"] * cl

        nav_records.append((date, nav))

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
    return stats, equity_df, trade_df


# ============================================================================
# 5-day rebalancing long-only: Top-N ranking, periodic portfolio turnover
# ============================================================================

def run_portfolio_rebalance(
    predictions,
    ohlcv_map,
    test_start=None,
    max_positions=10,
    rebalance_freq=5,
    auction_buffer=0.02,
    sell_markup=0.001,
    excluded_codes=None,
    initial_cash_per_stock=10000,
    commission=0.0006,
    stamp_duty=0.0005,
    risk_free_rate=0.025,
    delist_info=None,
    rank_scores=None,
    rank_threshold=None,
):
    """Long-only backtest with periodic rebalancing and overnight limit orders.

    On every *rebalance_freq* trading day:
      - Rank all eligible stocks by prediction score descending.
      - If rank_threshold is None: classic top-N — target = top max_positions.
      - If rank_threshold is set (e.g. 0.90): threshold entry — only stocks
        with rank score >= rank_threshold qualify (score should be a
        per-date percentile in (0,1)); target = qualifying stocks capped
        at max_positions, bought in rank order. NOTHING qualifies -> all
        positions sold -> cash (空仓 allowed; max_positions is a cap,
        not a target).
      - Sell positions not in target; buy target stocks not yet held.
      - Sell orders persist across non-rebalance days; unfilled buy orders
        expire at end of their execution day.

    predictions : pd.Series, MultiIndex (date, code), values = pred_hd scores
    ohlcv_map   : {code: DataFrame} with DatetimeIndex, columns Open/High/Low/Close/Volume/IsST
    delist_info : dict or None
        Mapping from code to delist_date (Timestamp). Stocks on or after
        their delist_date are excluded from buy candidates.
    rank_scores : pd.Series or None（bench 2026-08，P0-2 解耦）
        排序专用通道（如 combine_scores 的 rank 百分位）。提供时：候选集
        仍由 predictions（exec 量纲，限价公式依赖）决定，但 top-N 排序用
        rank_scores。缺省 None 时行为与旧版完全一致。
    rank_threshold : float or None（bench 2026-08-21 用户裁定：阈值准入）
        候选准入阈值：rank score >= 该值才可买/可留（配合按日百分位
        score，0.90 即当日综合分前 ~10%）。
    """
    all_dates = sorted(set().union(*(ohlcv.index for ohlcv in ohlcv_map.values())))
    total_cash = float(max_positions * initial_cash_per_stock)
    cash = total_cash
    if excluded_codes is None:
        excluded_codes = set()

    positions = {}       # code -> {"shares", "cost", "entry_date"}
    sell_orders = {}     # code -> limit_price
    buy_orders = {}      # code -> limit_price
    buy_lock = set()     # T+1 sell prohibition

    nav_records = []
    pos_count_records = []   # (date, n_positions, cash) —— 空仓率/仓位统计
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
        # Phase 1: Execute sell orders
        # ================================================================
        deferred_sells = set()
        for code, limit_price in list(sell_orders.items()):
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
            limit_dn = _limit_down(prev_cl, is_st)

            if _is_frozen_down(hi, limit_dn):
                deferred_sells.add(code)
                continue

            if op >= limit_price:
                fill_px = op
            elif hi >= limit_price:
                fill_px = limit_price
            else:
                deferred_sells.add(code)
                continue

            pos = positions.get(code)
            if pos is None:
                continue

            shares = pos["shares"]
            cash += shares * fill_px * (1.0 - commission - stamp_duty)
            trades.append({"date": date, "code": code, "action": "SELL",
                           "price": fill_px, "shares": shares})
            del positions[code]

        remaining = {}
        for code, limit_price in sell_orders.items():
            if code in deferred_sells:
                remaining[code] = limit_price
        sell_orders = remaining

        # ================================================================
        # Phase 2: Execute buy orders
        # ================================================================
        buy_slots_remaining = max_positions - len(positions)

        for code, limit_price in list(buy_orders.items()):
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
            target_cost = min(initial_cash_per_stock, cash / max(1, buy_slots_remaining))
            shares = int(target_cost / cost_unit / LOT_SIZE) * LOT_SIZE
            if shares < LOT_SIZE:
                continue

            cost = shares * cost_unit
            if cost > cash:
                continue

            cash -= cost
            positions[code] = {"shares": shares, "cost": fill_px, "entry_date": date}
            trades.append({"date": date, "code": code, "action": "BUY",
                           "price": fill_px, "shares": shares})
            buy_slots_remaining -= 1

        # Unfilled buy orders expire at end of day (same as run_portfolio):
        # limit prices are based on tonight's close/pred and must not survive
        # to later days. Fresh orders are only placed on rebalance evenings.
        buy_orders.clear()

        buy_lock = {code for code, pos in positions.items() if pos["entry_date"] == date}

        # ================================================================
        # Phase 3: Evening — order generation (ONLY on rebalance days)
        # ================================================================
        is_rebalance = (i % rebalance_freq == 0)
        next_date = all_dates[i + 1] if i + 1 < n_dates else None

        if is_rebalance and next_date is not None and date in predictions.index.get_level_values("date"):
            try:
                today_pred = predictions.xs(date, level="date")
            except KeyError:
                today_pred = pd.Series(dtype=float)

            if not today_pred.empty:
                # --- 3a. Filter candidates ---
                # 候选集合 = 有 exec 预测（限价可定价）；top-N 排序用 rank 通道
                if rank_scores is not None:
                    try:
                        rank_vals = rank_scores.xs(date, level="date").reindex(today_pred.index)
                    except KeyError:
                        rank_vals = today_pred
                    candidates = pd.DataFrame(
                        {"rank": rank_vals, "exec": today_pred}
                    ).dropna(subset=["exec"])["rank"]
                else:
                    candidates = today_pred.dropna()
                candidates = candidates[~candidates.index.isin(excluded_codes)]
                st_codes = {c for c in candidates.index if isst_map.get(c, 0) == 1}
                candidates = candidates[~candidates.index.isin(st_codes)]
                # Exclude stocks past their delist_date
                if delist_info:
                    delisted = {c for c in candidates.index
                                if c in delist_info and date >= delist_info[c]}
                    candidates = candidates[~candidates.index.isin(delisted)]
                # 阈值准入（2026-08-21 用户裁定）：低于阈值的候选一律不买；
                # 无合格标的时 target 为空 -> 全部卖出 -> 允许空仓
                if rank_threshold is not None:
                    candidates = candidates[candidates >= rank_threshold]
                candidates = candidates.sort_values(ascending=False)

                target_codes = set(candidates.head(max_positions).index)

                # --- 3b. Sell orders: positions NOT in target (skip T+1 locked) ---
                new_sells = {}
                for code, pos in positions.items():
                    if code in buy_lock:
                        continue
                    if code in target_codes:
                        continue
                    pred_val = today_pred.get(code)
                    if pred_val is None or pd.isna(pred_val):
                        continue
                    prev_cl = close_map.get(code)
                    if prev_cl is None or prev_cl <= 0:
                        continue
                    new_sells[code] = _sell_limit(prev_cl, float(pred_val), sell_markup)

                # --- 3c. Buy orders: target stocks not yet held ---
                held_codes = set(positions.keys())
                available = max_positions - len(positions) + len(new_sells)

                new_buys = {}
                for code in candidates.index:
                    if code not in target_codes or code in held_codes:
                        continue
                    if len(new_buys) >= available:
                        break
                    prev_cl = close_map.get(code)
                    if prev_cl is None or prev_cl <= 0:
                        continue
                    new_buys[code] = _buy_limit(prev_cl, float(today_pred[code]), auction_buffer)

                sell_orders = new_sells
                buy_orders = new_buys

        # ================================================================
        # Phase 4: NAV at close
        # ================================================================
        nav = cash
        for code, pos in positions.items():
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


# ============================================================================
# Long-Short cross-sectional signal test
# ============================================================================

def run_long_short(
    predictions,
    ohlcv_map,
    n_long=10,
    n_short=10,
    test_start=None,
    excluded_codes=None,
    risk_free_rate=0.025,
    commission=0.0003,
    stamp_duty=0.0005,
    delist_info=None,
    borrow_rate=0.08,
    rank_scores=None,
):
    """Daily-rebalanced long-short portfolio: long top N, short bottom N.

    Applies price-limit filtering AND transaction costs:
      - Long leg: skip stocks sealed at limit-UP   (can't buy)
      - Short leg: skip stocks sealed at limit-DOWN (can't short)
      - Both legs: subtract commission × 2 + stamp duty on exit each day

    Returns are T+1 close-to-close, net of costs.
    If the IC is real, long(top) − short(bottom) should be consistently positive.

    predictions : pd.Series, MultiIndex (date, code), values = pred scores
    ohlcv_map   : {code: DataFrame} with DatetimeIndex, columns Close + IsST
    """
    if excluded_codes is None:
        excluded_codes = set()

    all_dates = sorted(predictions.index.get_level_values("date").unique())
    daily_rets: dict[pd.Timestamp, float] = {}
    long_rets: dict[pd.Timestamp, float] = {}
    short_rets: dict[pd.Timestamp, float] = {}

    n_long_skipped = 0
    n_short_skipped = 0
    n_days_traded = 0

    # round-trip cost: commission × 2 + stamp duty on exit
    roundtrip_cost = commission * 2 + stamp_duty
    # daily borrowing cost for short leg
    daily_borrow = borrow_rate / 252.0

    for i, date in enumerate(all_dates):
        # --- today's predictions ---
        try:
            today_pred = predictions.xs(date, level="date")
        except KeyError:
            continue

        today_pred = today_pred[~today_pred.index.isin(excluded_codes)]
        today_pred = today_pred.dropna()

        # Filter out ST stocks on this date
        st_codes = set()
        for c in today_pred.index:
            ohlcv = ohlcv_map.get(c)
            if ohlcv is not None and date in ohlcv.index:
                try:
                    if int(ohlcv.loc[date].get("IsST", 0)) == 1:
                        st_codes.add(c)
                except (ValueError, TypeError):
                    pass
        if st_codes:
            today_pred = today_pred[~today_pred.index.isin(st_codes)]

        # Filter out stocks past their delist_date
        if delist_info:
            delisted = {c for c in today_pred.index
                        if c in delist_info and date >= delist_info[c]}
            if delisted:
                today_pred = today_pred[~today_pred.index.isin(delisted)]

        if len(today_pred) < n_long + n_short:
            continue

        # --- next trading day ---
        next_date = None
        for nd in all_dates[i + 1:]:
            next_date = nd
            break
        if next_date is None:
            continue

        # --- collect close / prev_close / limits in one pass ---
        class _Rec:
            __slots__ = ("close_t", "close_n", "limit_up", "limit_dn")

        records: dict[str, _Rec] = {}
        for code in today_pred.index:
            ohlcv = ohlcv_map.get(code)
            if ohlcv is None or date not in ohlcv.index:
                continue
            pos = ohlcv.index.get_loc(date)
            if pos == 0:
                continue
            row_t = ohlcv.iloc[pos]
            row_p = ohlcv.iloc[pos - 1]

            close_t = float(row_t["Close"])
            prev_c = float(row_p["Close"])
            if prev_c <= 0:
                continue

            if next_date not in ohlcv.index:
                continue
            pos_n = ohlcv.index.get_loc(next_date)
            close_n = float(ohlcv.iloc[pos_n]["Close"])

            # price-limit band: 10% normal, 5% ST
            try:
                is_st = int(row_t.get("IsST", 0))
            except (ValueError, TypeError):
                is_st = 0
            limit_pct = 0.05 if is_st else 0.10

            rec = _Rec()
            rec.close_t = close_t
            rec.close_n = close_n
            rec.limit_up = prev_c * (1.0 + limit_pct)
            rec.limit_dn = prev_c * (1.0 - limit_pct)
            records[code] = rec

        if not records:
            continue

        valid = set(records) & set(today_pred.index)
        valid_pred = today_pred[today_pred.index.isin(valid)]
        # 排序用 rank 通道（bench 2026-08 P0-2 解耦；缺省同旧版）
        if rank_scores is not None:
            try:
                day_rank = rank_scores.xs(date, level="date").reindex(valid_pred.index)
                valid_pred = day_rank.sort_values(ascending=False, na_position="last")
            except KeyError:
                pass
        ranked = valid_pred.sort_values(ascending=False)

        if len(ranked) < n_long + n_short:
            continue

        # --- select long leg: top N, skip limit-up ---
        long_codes = []
        for c in ranked.index:
            if len(long_codes) >= n_long:
                break
            r = records[c]
            if r.close_t >= r.limit_up * 0.999:
                n_long_skipped += 1
                continue
            long_codes.append(c)

        # --- select short leg: bottom N, skip limit-down ---
        short_codes = []
        for c in ranked.index[::-1]:  # worst first
            if len(short_codes) >= n_short:
                break
            r = records[c]
            if r.close_t <= r.limit_dn * 1.001:
                n_short_skipped += 1
                continue
            short_codes.append(c)

        if len(long_codes) < n_long or len(short_codes) < n_short:
            continue

        n_days_traded += 1

        gross_lr = np.mean([records[c].close_n / records[c].close_t - 1.0 for c in long_codes])
        gross_sr = np.mean([records[c].close_n / records[c].close_t - 1.0 for c in short_codes])

        # costs: each leg pays roundtrip cost independently → 2× total
        # short leg also pays daily borrow cost
        lr = gross_lr
        sr_ = gross_sr
        daily_ret = gross_lr - gross_sr - daily_borrow - roundtrip_cost * 2

        long_rets[date] = float(lr)
        short_rets[date] = float(sr_)
        daily_rets[date] = float(daily_ret)

    if not daily_rets:
        return {}, pd.DataFrame()

    ls_series = pd.Series(daily_rets, name="long_short_ret").sort_index()
    ls_series.index = pd.to_datetime(ls_series.index)

    if test_start is not None:
        ls_series = ls_series[ls_series.index >= pd.Timestamp(test_start)]

    equity = (1.0 + ls_series).cumprod()
    equity_df = pd.DataFrame({"daily_ret": ls_series, "equity": equity})

    stats = {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 0 else 0.0,
        "mean_daily_ret": float(ls_series.mean()),
        "std_daily_ret": float(ls_series.std()),
        "sharpe": float(ls_series.mean() / ls_series.std() * np.sqrt(252)) if ls_series.std() > 0 else 0.0,
        "n_days": len(ls_series),
        "n_days_traded": n_days_traded,
        "n_long_limit_skipped": n_long_skipped,
        "n_short_limit_skipped": n_short_skipped,
    }
    return stats, equity_df


def compute_benchmark(ohlcv_map, test_dates, delist_info=None, excluded_codes=None):
    """Buy-and-hold equal-weight benchmark.

    Start with 1 unit per stock at test_dates[0], never rebalance.
    Suspended stocks hold at last known close.
    Delisted stocks become worthless after delist_date.
    ST stocks excluded.
    """
    if not test_dates or not ohlcv_map:
        return pd.DataFrame()

    excluded = excluded_codes or set()

    # Build delist set: code → delist_date
    delist_dates = {}
    if delist_info:
        for code, dl_date in delist_info.items():
            delist_dates[code] = pd.Timestamp(dl_date)

    # For each stock, find base close (first date in test period with data)
    base_dt = test_dates[0]
    base_close: dict[str, float] = {}
    for code, ohlcv in ohlcv_map.items():
        valid = ohlcv[ohlcv.index >= base_dt]
        if not valid.empty:
            base_close[code] = float(valid.iloc[0]["Close"])

    # Track last known close for each stock (for suspension handling)
    last_close: dict[str, float] = dict(base_close)

    nav_records = []
    for dt in test_dates:
        active = 0
        nav = 0.0
        for code, base_cl in base_close.items():
            if code in excluded:
                continue
            # Delisted after delist_date → value = 0
            if code in delist_dates and dt >= delist_dates[code]:
                continue
            ohlcv = ohlcv_map.get(code)
            if ohlcv is None:
                continue
            if dt in ohlcv.index:
                cl = float(ohlcv.loc[dt, "Close"])
                last_close[code] = cl
            else:
                cl = last_close.get(code, base_cl)
            if pd.isna(cl) or cl <= 0:
                continue
            active += 1
            nav += cl / base_cl

        if active > 0 and nav > 0:
            nav_records.append((dt, nav))

    if len(nav_records) < 2:
        return pd.DataFrame()

    nav_series = pd.Series(
        [v for _, v in nav_records],
        index=pd.DatetimeIndex([d for d, _ in nav_records]),
    ).sort_index()

    daily_ret = nav_series.pct_change().dropna()
    equity = nav_series / nav_series.iloc[0]
    return pd.DataFrame({"daily_ret": daily_ret, "equity": equity})
