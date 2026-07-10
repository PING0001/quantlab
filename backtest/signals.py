# -*- coding: utf-8 -*-
"""Daily single-entry long-only backtest with call auction + intraday execution.

T+1 constraint: stocks bought today cannot be sold tomorrow.

Entry (buy):
  trigger   pred_5d > entry_threshold
  limit     prev_close * (1 + pred - auction_buffer)
  fill      open <= limit -> fill at open (call auction)
            low  <= limit -> fill at limit (intraday pullback)
            frozen-up    -> skip (sealed limit-up)

Exit (sell):
  trigger   pred_5d < exit_threshold (all held stocks, except T+1 locked)
  limit     prev_close * (1 + pred + sell_markup)
  fill      open >= limit -> fill at open (call auction)
            high >= limit -> fill at limit (intraday spike)
            frozen-down   -> defer to next day
            otherwise      -> defer to next day
  cancel    deferred sell cancelled if pred recovers >= exit_threshold

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
    return prev_close * (1.0 + _limit_pct(is_st))


def _limit_down(prev_close, is_st) -> float:
    return prev_close * (1.0 - _limit_pct(is_st))


def _is_frozen_up(low, limit_up_price) -> bool:
    return low >= limit_up_price


def _is_frozen_down(high, limit_down_price) -> bool:
    return high <= limit_down_price


# ============================================================================
# Limit price formulas
# ============================================================================

def _buy_limit(prev_close, pred_score: float, auction_buffer: float = 0.025) -> float:
    """Overnight buy limit: prev_close * (1 + pred - buffer)."""
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

def run_portfolio(
    predictions,
    ohlcv_map,
    test_start=None,
    max_positions=10,
    entry_threshold=0.03,
    exit_threshold=0.0,
    auction_buffer=0.025,
    sell_markup=0.0005,
    excluded_codes=None,
    initial_cash_per_stock=10000,
    commission=0.0003,
    risk_free_rate=0.025,
):
    """Long-only backtest with overnight limit orders and call auction execution.

    predictions : pd.Series, MultiIndex (date, code), values = pred_5d scores
    ohlcv_map   : {code: DataFrame}, columns must include
                  Open, High, Low, Close, Volume, IsST
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

        # ---- helper: get prev_close for a code on this date ----
        def _prev_close(code):
            ohlcv = ohlcv_map[code]
            idx = ohlcv.index.get_loc(date)
            if idx == 0:
                return None
            return float(ohlcv.iloc[idx - 1]["Close"])

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

            prev_cl = _prev_close(code)
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
            cash += shares * fill_px * (1.0 - commission)
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

            prev_cl = _prev_close(code)
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

            # --- 3a. Recalculate / cancel deferred sells ---
            new_sells = {}
            for code in list(sell_orders.keys()):
                pred_val = today_pred.get(code)
                if pred_val is None or pd.isna(pred_val) or pred_val >= exit_threshold:
                    continue
                if code not in positions:
                    continue
                prev_cl = close_map.get(code)
                if prev_cl is None or prev_cl <= 0:
                    continue
                new_sells[code] = _sell_limit(prev_cl, float(pred_val), sell_markup)

            # --- 3b. New sell candidates from positions ---
            for code, pos in positions.items():
                if code in new_sells:
                    continue
                if code in buy_lock:
                    continue
                pred_val = today_pred.get(code)
                if pred_val is None or pd.isna(pred_val) or pred_val >= exit_threshold:
                    continue
                prev_cl = close_map.get(code)
                if prev_cl is None or prev_cl <= 0:
                    continue
                new_sells[code] = _sell_limit(prev_cl, float(pred_val), sell_markup)

            # --- 3c. Buy candidates ---
            effective_slots = max_positions - len(positions) + len(new_sells)
            held_codes = set(positions.keys())

            candidates = today_pred[today_pred > entry_threshold]
            candidates = candidates[~candidates.index.isin(held_codes)]
            # Exclude ST/delisting stocks (name-based)
            candidates = candidates[~candidates.index.isin(excluded_codes)]
            # Exclude stocks that are ST on today's date
            st_codes = {code for code in candidates.index if isst_map.get(code, 0) == 1}
            candidates = candidates[~candidates.index.isin(st_codes)]
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


def compute_benchmark(ohlcv_map, test_dates):
    """Compute equal-weight benchmark daily returns."""
    if not test_dates:
        return pd.DataFrame()
    daily_rets = {}
    for dt in test_dates:
        rets = []
        for ohlcv in ohlcv_map.values():
            if dt not in ohlcv.index:
                continue
            pos = ohlcv.index.get_loc(dt)
            if pos == 0:
                continue
            prev_close = ohlcv.iloc[pos - 1]["Close"]
            curr_close = ohlcv.loc[dt, "Close"]
            if prev_close > 0:
                rets.append(float(curr_close / prev_close - 1))
        if rets:
            daily_rets[dt] = float(np.mean(rets))
    sr = pd.Series(daily_rets, name="benchmark_ret")
    sr.index = pd.to_datetime(sr.index)
    sr = sr.sort_index()
    equity = (1.0 + sr).cumprod()
    return pd.DataFrame({"daily_ret": sr, "equity": equity})
