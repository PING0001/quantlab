# -*- coding: utf-8 -*-
"""Portfolio-level long-short backtest with next-day open execution."""
from __future__ import annotations

import numpy as np
import pandas as pd

LOT_SIZE = 100


def _is_gap_up(open_price, prev_close, gap_filter=0.015):
    if pd.isna(open_price) or pd.isna(prev_close) or prev_close <= 0:
        return True
    return open_price > prev_close * (1.0 + gap_filter)


def _is_gap_down(open_price, prev_close, gap_filter=0.015):
    if pd.isna(open_price) or pd.isna(prev_close) or prev_close <= 0:
        return True
    return open_price < prev_close * (1.0 - gap_filter)


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


def run_portfolio(predictions, ohlcv_map, test_start=None, top_k=3, rebal_interval=5,
                  risk_free_rate=0.025,
                  initial_cash_per_side=10000, commission=0.0003,
                  gap_filter=0.015):
    """Long-short portfolio backtest with next-day open execution.

    Each rebalancing day:
      1. Take top_k longs and bottom_k shorts by MLP score.
      2. Execute at the *next* trading day open price.
      3. Long entry filtered if open gaps up > gap_filter.
      4. Short entry filtered if open gaps down > gap_filter.
      5. Exit always executes at open (no filter).

    Returns (stats_dict, equity_df, trade_df).
    """
    # ---- trading calendar ----
    all_dates = sorted(set().union(*(ohlcv.index for ohlcv in ohlcv_map.values())))

    # ---- rebalancing schedule ----
    pred_dates = sorted(predictions.index.get_level_values("date").unique())
    rebal_dates = pred_dates[::rebal_interval]

    rebal_plan = {}  # {exec_date: {"L": [...], "S": [...]}}
    for reb in rebal_dates:
        try:
            pred = predictions.xs(reb, level="date")
        except KeyError:
            continue
        pred = pred.dropna().sort_values()
        if len(pred) < top_k * 2:
            continue
        # next trading day
        try:
            idx = all_dates.index(reb)
            exec_date = all_dates[idx + 1]
        except (ValueError, IndexError):
            continue
        rebal_plan[exec_date] = {
            "L": list(pred.nlargest(top_k).index),
            "S": list(pred.nsmallest(top_k).index),
        }

    # ---- portfolio state ----
    total_cash = float(top_k * initial_cash_per_side * 2)
    cash = total_cash
    positions = {}  # {code: {"side": "L"|"S", "shares": int, "cost": float}}

    nav_records = []  # [(date, nav)]
    trades = []       # [{date, code, action, price, shares}]

    for date in all_dates:
        # -- open snapshot --
        open_map = {}
        for code, ohlcv in ohlcv_map.items():
            if date in ohlcv.index:
                px = float(ohlcv.loc[date, "Open"])
                if not pd.isna(px) and px > 0:
                    open_map[code] = px

        # -- execute rebal -- #
        if date in rebal_plan:
            plan = rebal_plan[date]

            # 1. close all existing positions
            for code, pos in list(positions.items()):
                px = open_map.get(code)
                if px is None:
                    continue
                if pos["side"] == "L":
                    cash += pos["shares"] * px * (1.0 - commission)
                    trades.append({"date": date, "code": code, "action": "SELL",
                                   "price": px, "shares": pos["shares"]})
                else:  # short
                    cash -= pos["shares"] * px * (1.0 + commission)
                    trades.append({"date": date, "code": code, "action": "COVER",
                                   "price": px, "shares": pos["shares"]})
                del positions[code]

            # 2. filter new entries by gap
            valid_longs = []
            for code in plan["L"]:
                px = open_map.get(code)
                if px is None:
                    continue
                ohlcv = ohlcv_map[code]
                idx_pos = ohlcv.index.get_loc(date)
                if idx_pos == 0:
                    continue
                prev_close = float(ohlcv.iloc[idx_pos - 1]["Close"])
                if _is_gap_up(px, prev_close, gap_filter):
                    continue
                valid_longs.append((code, px))

            valid_shorts = []
            for code in plan["S"]:
                px = open_map.get(code)
                if px is None:
                    continue
                ohlcv = ohlcv_map[code]
                idx_pos = ohlcv.index.get_loc(date)
                if idx_pos == 0:
                    continue
                prev_close = float(ohlcv.iloc[idx_pos - 1]["Close"])
                if _is_gap_down(px, prev_close, gap_filter):
                    continue
                valid_shorts.append((code, px))

            n_slots = len(valid_longs) + len(valid_shorts)
            if n_slots > 0:
                cash_per_slot = cash / n_slots

                for code, px in valid_longs:
                    cost_unit = px * (1.0 + commission)
                    shares = int(cash_per_slot / cost_unit / LOT_SIZE) * LOT_SIZE
                    if shares < LOT_SIZE:
                        continue
                    cost = shares * cost_unit
                    if cost > cash:
                        continue
                    cash -= cost
                    positions[code] = {"side": "L", "shares": shares, "cost": px}
                    trades.append({"date": date, "code": code, "action": "BUY",
                                   "price": px, "shares": shares})

                # re-spread remaining cash after longs consumed
                remaining_slots = len(valid_shorts)
                if remaining_slots > 0:
                    cash_per_short = cash / remaining_slots
                    for code, px in valid_shorts:
                        cost_unit = px * (1.0 + commission)
                        shares = int(cash_per_short / cost_unit / LOT_SIZE) * LOT_SIZE
                        if shares < LOT_SIZE:
                            continue
                        credit = shares * px * (1.0 - commission)
                        cash += credit
                        positions[code] = {"side": "S", "shares": shares, "cost": px}
                        trades.append({"date": date, "code": code, "action": "SHORT",
                                       "price": px, "shares": shares})

        # -- NAV at close -- #
        nav = cash
        for code, pos in positions.items():
            ohlcv = ohlcv_map.get(code)
            if ohlcv is None or date not in ohlcv.index:
                continue
            close_px = float(ohlcv.loc[date, "Close"])
            if pd.isna(close_px):
                continue
            if pos["side"] == "L":
                nav += pos["shares"] * close_px
            else:
                nav -= pos["shares"] * close_px

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
    trade_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=["date", "code", "action", "price", "shares"])
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
