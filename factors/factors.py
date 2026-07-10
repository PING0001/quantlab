# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
from .ops import (
    rank, scale, signed_power,
    ts_sum, ts_mean, ts_std, ts_min, ts_max,
    delay, delta, correlation, covariance,
    decay_linear, ts_rank, ts_argmax, ts_argmin,
    reg_beta,
)

def SMA(data, period=20):
    return ts_mean(data["close"], period)


def EMA(data, period=20):
    return data["close"].ewm(span=period, adjust=False).mean()


def RSI(data, period=14):
    close = data["close"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    loss_mask = avg_loss == 0
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - 100 / (1 + rs)
    result[loss_mask] = 100.0
    return result


def MACD(data, ):
    close = data["close"]
    return close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()


def MACD_signal(data, period=9):
    return MACD(data).ewm(span=period, adjust=False).mean()


def Bollinger_upper(data, period=20, n_std=2.0):
    m = ts_mean(data["close"], period)
    s = ts_std(data["close"], period)
    return m + n_std * s


def Bollinger_lower(data, period=20, n_std=2.0):
    m = ts_mean(data["close"], period)
    s = ts_std(data["close"], period)
    return m - n_std * s


def Bollinger_width(data, period=20, n_std=2.0):
    s = ts_std(data["close"], period)
    m = ts_mean(data["close"], period).replace(0, np.nan)
    return 2 * n_std * s / m


def ATR(data, period=14):
    h, l, c = data["high"], data["low"], data["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def CCI(data, period=20):
    tp = (data["high"] + data["low"] + data["close"]) / 3
    sma = ts_mean(tp, period)
    mad = (tp - sma).abs().rolling(period, min_periods=1).mean()
    cci = (tp - sma) / (0.015 * mad.replace(0, np.nan))
    cci[mad == 0] = 0.0
    return cci


def Stochastic_K(data, period=14):
    ll = ts_min(data["low"], period)
    hh = ts_max(data["high"], period)
    rng = (hh - ll).replace(0, np.nan)
    stoch = (data["close"] - ll) / rng * 100
    stoch[hh == ll] = 50.0
    return stoch


def OBV(data, ):
    d = np.sign(data["close"].diff()).fillna(0)
    return (d * data["volume"]).cumsum()


def Volume_ratio(data, period=5):
    avg_vol = ts_mean(data["volume"], period)
    result = data["volume"] / avg_vol.replace(0, np.nan)
    zero_mask = avg_vol == 0
    result[zero_mask & (data["volume"] > 0)] = 5.0
    result[zero_mask & (data["volume"] == 0)] = 0.0
    return result


def Return_1d(data, ):
    return data["close"].pct_change()


def Return_5d(data, ):
    return data["close"].pct_change(5)


def Return_20d(data, ):
    return data["close"].pct_change(20)


def Volatility(data, period=20):
    return data["close"].pct_change().rolling(period, min_periods=1).std(ddof=1)


def Price_position(data, period=20):
    l = ts_min(data["low"], period)
    h = ts_max(data["high"], period)
    rng = (h - l).replace(0, np.nan)
    return (data["close"] - l) / rng


def Price_position_252d(data):
    """Annual (252-day) price position: where close sits in 1-year range.
    
    min_periods=20: factor value only available after 20 trading days.
    Warm-up (125d) handled at strategy/training level.
    """
    l = data["low"].rolling(252, min_periods=20).min()
    h = data["high"].rolling(252, min_periods=20).max()
    rng = (h - l).replace(0, np.nan)
    return (data["close"] - l) / rng


def Zscore_close(data, period=20):
    std_val = ts_std(data["close"], period)
    zscore = (data["close"] - ts_mean(data["close"], period)) / std_val.replace(0, np.nan)
    zscore[std_val == 0] = 0.0
    return zscore


def Reversal_60d(data):
    """Medium-term (60-day) reversal: negative of cumulative return."""
    return -ts_sum(data["close"].pct_change(), 60)


def Volatility_60d(data):
    """60-day rolling volatility of daily returns."""
    return data["close"].pct_change().rolling(60, min_periods=20).std(ddof=1)


def Turnover_20d(data):
    """20-day average turnover rate."""
    return ts_mean(data["turn"], 20)


def WinnerRate(data):
    """Chip win rate: fraction of positions in profit [0, 1]."""
    wr = data["winner_rate"]
    wr = wr.replace([np.inf, -np.inf], np.nan)
    return wr / 100.0


def CostPosition(data):
    """Current price relative to weighted average holding cost."""
    wa = data["weight_avg"].replace(0, np.nan)
    return (data["close"] - wa) / wa


def ChipDispersion(data):
    """Width of cost distribution: (P95 - P5) / weighted average cost."""
    wa = data["weight_avg"].replace(0, np.nan)
    return (data["cost_95pct"] - data["cost_5pct"]) / wa


def ChipSkew(data):
    """Skew of cost distribution: (weight_avg - median) / inter-percentile range."""
    rng = (data["cost_95pct"] - data["cost_5pct"]).replace(0, np.nan)
    return (data["weight_avg"] - data["cost_50pct"]) / rng


def Turnover_3d(data):
    """3-day average turnover rate."""
    return ts_mean(data["turn"], 3)


def Turnover_3d_ratio(data):
    """Ratio of 3-day to 20-day average turnover."""
    t3 = Turnover_3d(data)
    t20 = Turnover_20d(data)
    result = t3 / t20.replace(0, np.nan)
    zero_mask = (t20 == 0) & t3.notna()
    result[zero_mask & (t3 > 0)] = 5.0
    result[zero_mask & (t3 == 0)] = 0.0
    return result


def Intraday_return(data):
    """Intraday return: (close - open) / open."""
    return (data["close"] - data["open"]) / data["open"].replace(0, np.nan)


def Return_60d(data):
    """60-day cumulative return (medium-term momentum)."""
    return data["close"].pct_change(60)


def RSI_60d(data):
    """60-day RSI (medium-term overbought/oversold)."""
    close = data["close"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=60, adjust=False).mean()
    avg_loss = loss.ewm(span=60, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float('nan'))
    return 100 - 100 / (1 + rs)


def alpha001(data):
    c, v = data["close"], data["volume"]
    return ts_argmax(signed_power(correlation(c, v, 5), 2), 5) - 0.5


def alpha002(data):
    return (-correlation(data["dc1_cs"], data["dv1_cs"], 6)).replace([np.inf, -np.inf], np.nan)


def alpha003(data):
    return (-correlation(data["close_cs"], data["volume_cs"], 10)).replace([np.inf, -np.inf], np.nan)


def alpha004(data):
    l_cs = data["low_cs"]
    return -ts_rank(l_cs, 9)


def alpha006(data):
    o, v = data["open"], data["volume"]
    return -correlation(o, v, 10)


def alpha009(data):
    c = data["close"]
    dc = delta(c, 1)
    return np.where(ts_min(dc, 5) > 0, dc, np.where(ts_max(dc, 5) < 0, dc, -dc))


def alpha012(data):
    v = data["volume"]
    return np.sign(delta(v, 1)) * (-delta(v, 1))


def alpha013(data):
    return -covariance(data["close_cs"], data["volume_cs"], 5)


def alpha014(data):
    o = data["open"]
    v_cs = data["volume_cs"]
    return -delta(v_cs, 1) * correlation(o, data["volume"], 10)


def alpha019(data):
    c = data["close"]
    ret_cs = data["ret1d_cs"]
    return -delta(ret_cs, 3) * correlation(c, data["close"].pct_change().fillna(0), 5)


def alpha020(data):
    c_cs = data["close_cs"]
    v_cs = data["volume_cs"]
    return -correlation(c_cs, v_cs, 3) * covariance(c_cs, v_cs, 5)


def alpha050(data):
    c, v = data["close"], data["volume"]
    return -correlation(ts_rank(c, 10), ts_rank(v, 10), 10)


def alpha060(data):
    c = data["close"]
    return -(ts_rank(ts_std(c, 20), 10) - ts_rank(ts_std(c, 5), 10))


def alpha101(data):
    c, v = data["close"], data["volume"]
    return correlation(c, v, 10) * correlation(ts_rank(c, 10), ts_rank(v, 10), 10)


def alpha191(data):
    c, v = data["close"], data["volume"]
    num = correlation(ts_rank(c, 10), ts_rank(v, 10), 10)
    den = ts_rank(correlation(c, v, 10), 10)
    return num / den.replace(0, np.nan)


def alpha007(data):
    c, v, amt = data["close"], data["volume"], data["amount"]
    adv20 = ts_mean(amt, 20)
    dc7 = delta(c, 7)
    return np.where(adv20 < v, (-ts_rank(np.abs(dc7), 60)) * np.sign(dc7), -1.0)


def alpha017(data):
    c, v, amt = data["close"], data["volume"], data["amount"]
    adv20 = ts_mean(amt, 20)
    return {
        "_d17_a": ts_rank(c, 10),
        "_d17_b": delta(delta(c, 1), 1),
        "_d17_c": ts_rank(v / adv20.replace(0, np.nan), 5),
    }


def alpha018(data):
    c, o = data["close"], data["open"]
    intra = c - o
    return -(ts_std(np.abs(intra), 5) + intra + correlation(c, o, 10))


def alpha028(data):
    h, l, c, amt = data["high"], data["low"], data["close"], data["amount"]
    adv20 = ts_mean(amt, 20)
    return scale(correlation(adv20, l, 5) + (h + l) / 2 - c)


def alpha035(data):
    c, h, l, v = data["close"], data["high"], data["low"], data["volume"]
    ret = c.pct_change().fillna(0)
    return ts_rank(v, 32) * (1 - ts_rank(c + h - l, 16)) * (1 - ts_rank(ret, 32))


def alpha038(data):
    c, o = data["close"], data["open"]
    return {
        "_d38_a": ts_rank(c, 10),
        "_d38_b": c / o.replace(0, np.nan),
    }


def alpha046(data):
    c = data["close"]
    d20 = delay(c, 20)
    d10 = delay(c, 10)
    accel = (d20 - d10) / 10 - (d10 - c) / 10
    return np.where(accel > 0.25, -1.0, np.where(accel < 0, 1.0, -(c - delay(c, 1))))


def alpha057(data):
    c, amt, vol = data["close"], data["amount"], data["volume"]
    vwap = amt / vol.replace(0, np.nan)
    vwap[vol == 0] = c[vol == 0]
    return {
        "_d57_a": ts_argmax(c, 30),
        "_d57_b": c - vwap,
    }




def Gap_pct(data):
    """Overnight gap: (open - prev_close) / prev_close."""
    return data["close"].shift(1).replace(0, np.nan).rdiv(data["open"]) - 1


def Body_pct(data):
    """Candlestick body as fraction of daily range."""
    rng = (data["high"] - data["low"]).replace(0, np.nan)
    body = (data["close"] - data["open"]).abs() / rng
    body[(data["high"] == data["low"])] = 0.0
    return body


def Trend_strength(data, period=20):
    """Absolute strength of recent trend: |20d return| / 20d vol."""
    ret = data["close"].pct_change()
    ret_20d = data["close"].pct_change(period)
    vol_20d = ret.rolling(period, min_periods=10).std(ddof=1).replace(0, np.nan)
    ts = ret_20d.abs() / vol_20d
    undef = (vol_20d == 0) | vol_20d.isna() | ret_20d.isna()
    ts[undef & (ret_20d.abs().fillna(0) > 0)] = 5.0
    ts[undef & (ret_20d.abs().fillna(0) == 0)] = 0.0
    return ts


def Intraday_position(data):
    """Where close sits within the day's range: (close - low) / (high - low)."""
    rng = (data["high"] - data["low"]).replace(0, np.nan)
    pos = (data["close"] - data["low"]) / rng
    flat = data["high"] == data["low"]
    prev_close = data["close"].shift(1).fillna(data["close"])
    pos[flat & (data["close"] > prev_close)] = 1.0
    pos[flat & (data["close"] < prev_close)] = 0.0
    pos[flat & (data["close"] == prev_close)] = 0.5
    return pos


def Amihud_illiquidity(data, period=20):
    """Amihud illiquidity ratio: rolling mean of |return| / dollar volume."""
    dollar_vol = (data["volume"] * data["close"]).replace(0, np.nan)
    ret = data["close"].pct_change().abs()
    raw = ret / dollar_vol
    raw[(data["volume"] * data["close"] == 0)] = 1e6
    return raw.rolling(period, min_periods=10).mean()



def Return_skew_20d(data):
    """Skewness of daily returns over 20 days."""
    return data["close"].pct_change().rolling(20, min_periods=10).skew()



def Intraday_range_pct(data):
    """Intraday range normalized by open: (high - low) / open."""
    return (data["high"] - data["low"]) / data["open"].replace(0, np.nan)


def LnMktCap(data):
    """Natural log of total market capitalisation (万元 -> 元 then ln)."""
    mv = data["total_mv"] * 1e4
    mv = mv.replace(0, np.nan)
    return np.log(mv)


def LnAge(data):
    """Natural log of calendar days since IPO listing date."""
    if "list_date" not in data.columns:
        return pd.Series(np.nan, index=data.index)
    ld = data["list_date"].iloc[0]
    if pd.isna(ld):
        return pd.Series(np.nan, index=data.index)
    days = (pd.to_datetime(data["date"]) - pd.Timestamp(ld)).dt.days.astype(float)
    days = np.clip(days, 1, None)
    return np.log(days)


def LnFloatCap(data):
    """Natural log of circulating market capitalisation (万元 -> 元 then ln)."""
    mv = data["circ_mv"] * 1e4
    mv = mv.replace(0, np.nan)
    return np.log(mv)


def AvgAmount_90d(data):
    """90-day rolling mean of daily turnover amount."""
    return data["amount"].rolling(90, min_periods=1).mean()

""
""
# 鍥犲瓙娉ㄥ唽琛?--- 涓€涓?dict
FACTOR_HUB = {
    'SMA': SMA,
    'EMA': EMA,
    'RSI': RSI,
    'MACD': MACD,
    'MACD_signal': MACD_signal,
    'Bollinger_upper': Bollinger_upper,
    'Bollinger_lower': Bollinger_lower,
    'Bollinger_width': Bollinger_width,
    'ATR': ATR,
    'CCI': CCI,
    'Stochastic_K': Stochastic_K,
    'OBV': OBV,
    'Volume_ratio': Volume_ratio,
    'Return_1d': Return_1d,
    'Return_5d': Return_5d,
    'Return_20d': Return_20d,
    'Volatility': Volatility,
    'Price_position': Price_position,
    'Price_position_252d': Price_position_252d,
    'Zscore_close': Zscore_close,
    'Reversal_60d': Reversal_60d,
    'Volatility_60d': Volatility_60d,
    'Turnover_20d': Turnover_20d,
    'Return_60d': Return_60d,
    'RSI_60d': RSI_60d,
    'alpha001': alpha001,
    'alpha002': alpha002,
    'alpha003': alpha003,
    'alpha004': alpha004,
    'alpha006': alpha006,
    'alpha009': alpha009,
    'alpha012': alpha012,
    'alpha013': alpha013,
    'alpha014': alpha014,
    'alpha019': alpha019,
    'alpha020': alpha020,
    'alpha050': alpha050,
    'alpha060': alpha060,
    'alpha101': alpha101,
    'alpha191': alpha191,
    'alpha007': alpha007,
    'alpha017': alpha017,
    'alpha018': alpha018,
    'alpha028': alpha028,
    'alpha035': alpha035,
    'alpha038': alpha038,
    'alpha046': alpha046,
    'alpha057': alpha057,
    'Gap_pct': Gap_pct,
    'Body_pct': Body_pct,
    'Trend_strength': Trend_strength,
    'Intraday_position': Intraday_position,
    'Amihud_illiquidity': Amihud_illiquidity,
    'Return_skew_20d': Return_skew_20d,
    'Intraday_range_pct': Intraday_range_pct,
    'LnMktCap': LnMktCap,
    'LnFloatCap': LnFloatCap,
    'AvgAmount_90d': AvgAmount_90d,
    'Turnover_3d': Turnover_3d,
    'Turnover_3d_ratio': Turnover_3d_ratio,
    'Intraday_return': Intraday_return,
    'WinnerRate': WinnerRate,
    'CostPosition': CostPosition,
    'ChipDispersion': ChipDispersion,
    'ChipSkew': ChipSkew,
    'LnAge': LnAge,
}

__all__ = list(FACTOR_HUB.keys())
