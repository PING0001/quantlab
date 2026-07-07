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
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


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
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def Stochastic_K(data, period=14):
    ll = ts_min(data["low"], period)
    hh = ts_max(data["high"], period)
    rng = (hh - ll).replace(0, np.nan)
    return (data["close"] - ll) / rng * 100


def OBV(data, ):
    d = np.sign(data["close"].diff()).fillna(0)
    return (d * data["volume"]).cumsum()


def Volume_ratio(data, period=5):
    return data["volume"] / ts_mean(data["volume"], period).replace(0, np.nan)


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
    return (data["close"] - ts_mean(data["close"], period)) / ts_std(data["close"], period).replace(0, np.nan)


def Reversal_60d(data):
    """Medium-term (60-day) reversal: negative of cumulative return."""
    return -ts_sum(data["close"].pct_change(), 60)


def Volatility_60d(data):
    """60-day rolling volatility of daily returns."""
    return data["close"].pct_change().rolling(60, min_periods=20).std(ddof=1)


def Turnover_20d(data):
    """20-day average turnover rate."""
    return ts_mean(data["turn"], 20)


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
    return rank(ts_argmax(signed_power(correlation(c, v, 5), 2), 5)) - 0.5


def alpha002(data):
    c, v = data["close"], data["volume"]
    return -correlation(rank(delta(c, 1)), rank(delta(v/100, 1)), 6)


def alpha003(data):
    c, v = data["close"], data["volume"]
    return -correlation(rank(c), rank(v/100), 10)


def alpha004(data):
    l = data["low"]
    return -ts_rank(rank(l), 9)


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
    c, v = data["close"], data["volume"]
    return -rank(covariance(rank(c), rank(v), 5))


def alpha014(data):
    o, v = data["open"], data["volume"]
    return -rank(delta(rank(v), 1)) * correlation(o, v, 10)


def alpha019(data):
    c = data["close"]
    ret = c.pct_change().fillna(0)
    return -rank(delta(rank(ret), 3) * correlation(c, ret, 5))


def alpha020(data):
    c, v = data["close"], data["volume"]
    return -rank(correlation(rank(c), rank(v), 3)) * rank(covariance(rank(c), rank(v), 5))


def alpha050(data):
    c, v = data["close"], data["volume"]
    return -rank(correlation(ts_rank(c, 10), ts_rank(v, 10), 10))


def alpha060(data):
    c = data["close"]
    return -(ts_rank(ts_std(c, 20), 10) - ts_rank(ts_std(c, 5), 10))


def alpha101(data):
    c, v = data["close"], data["volume"]
    return rank(correlation(c, v, 10)) * rank(correlation(ts_rank(c, 10), ts_rank(v, 10), 10))


def alpha191(data):
    c, v = data["close"], data["volume"]
    return rank(correlation(ts_rank(c, 10), ts_rank(v, 10), 10)) / rank(ts_rank(correlation(c, v, 10), 10)).replace(0, np.nan)




def Gap_pct(data):
    """Overnight gap: (open - prev_close) / prev_close."""
    return data["close"].shift(1).replace(0, np.nan).rdiv(data["open"]) - 1


def Body_pct(data):
    """Candlestick body as fraction of daily range."""
    rng = (data["high"] - data["low"]).replace(0, np.nan)
    return (data["close"] - data["open"]).abs() / rng


def Trend_strength(data, period=20):
    """Absolute strength of recent trend: |20d return| / 20d vol."""
    ret = data["close"].pct_change()
    ret_20d = data["close"].pct_change(period)
    vol_20d = ret.rolling(period, min_periods=10).std(ddof=1).replace(0, np.nan)
    return ret_20d.abs() / vol_20d


def Intraday_position(data):
    """Where close sits within the day's range: (close - low) / (high - low)."""
    rng = (data["high"] - data["low"]).replace(0, np.nan)
    return (data["close"] - data["low"]) / rng


def Amihud_illiquidity(data, period=20):
    """Amihud illiquidity ratio: rolling mean of |return| / dollar volume."""
    dollar_vol = (data["volume"] * data["close"]).replace(0, np.nan)
    ret = data["close"].pct_change().abs()
    return (ret / dollar_vol).rolling(period, min_periods=10).mean()



def Return_skew_20d(data):
    """Skewness of daily returns over 20 days."""
    return data["close"].pct_change().rolling(20, min_periods=10).skew()



def Intraday_range_pct(data):
    """Intraday range normalized by open: (high - low) / open."""
    return (data["high"] - data["low"]) / data["open"].replace(0, np.nan)

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
    'Gap_pct': Gap_pct,
    'Body_pct': Body_pct,
    'Trend_strength': Trend_strength,
    'Intraday_position': Intraday_position,
    'Amihud_illiquidity': Amihud_illiquidity,
    'Return_skew_20d': Return_skew_20d,
    'Intraday_range_pct': Intraday_range_pct
}

__all__ = list(FACTOR_HUB.keys())
