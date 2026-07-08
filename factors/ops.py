# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd

def rank(x):
    if isinstance(x, pd.Series):
        return x.rank(pct=True) - 0.5
    r = pd.Series(x).rank(pct=True)
    return (r - 0.5).values

def scale(x, a=1.0):
    s = np.abs(x).sum()
    if s < 1e-12:
        return x * 0
    return x * a / s

def signed_power(x, a):
    return np.sign(x) * (np.abs(x) ** a)

def ts_sum(x, d):
    return x.rolling(d, min_periods=1).sum()

def ts_mean(x, d):
    return x.rolling(d, min_periods=1).mean()

def ts_std(x, d):
    return x.rolling(d, min_periods=1).std(ddof=1)

def ts_min(x, d):
    return x.rolling(d, min_periods=1).min()

def ts_max(x, d):
    return x.rolling(d, min_periods=1).max()

def delay(x, d):
    return x.shift(d)

def delta(x, d):
    return x - x.shift(d)

def correlation(x, y, d):
    return x.rolling(d, min_periods=1).corr(y)

def covariance(x, y, d):
    return x.rolling(d, min_periods=1).cov(y)

def decay_linear(x, d):
    weights = np.arange(1, d + 1, dtype=float)
    wsum = weights.sum()
    def _decay(arr):
        n = len(arr)
        w = weights[-n:] / weights[-n:].sum() if n < d else weights / wsum
        return float(np.nansum(arr * w))
    return x.rolling(d, min_periods=1).apply(_decay, raw=True)

def ts_rank(x, d):
    def helper(arr):
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return np.nan
        return float(np.searchsorted(np.sort(valid), valid[-1], side="right")) / len(valid)
    return x.rolling(d, min_periods=1).apply(helper, raw=True)

def ts_argmax(x, d):
    def helper(arr):
        return float(np.nanargmax(arr)) if len(arr) > 0 else np.nan
    return x.rolling(d, min_periods=1).apply(helper, raw=True)

def ts_argmin(x, d):
    def helper(arr):
        return float(np.nanargmin(arr)) if len(arr) > 0 else np.nan
    return x.rolling(d, min_periods=1).apply(helper, raw=True)

def reg_beta(y, x, d):
    cov = y.rolling(d, min_periods=1).cov(x)
    var = x.rolling(d, min_periods=1).var(ddof=0)
    return (cov / var.replace(0, np.nan)).fillna(0)

def winsorize(x, limits=(0.01, 0.01)):
    return x.clip(lower=x.quantile(limits[0]), upper=x.quantile(1 - limits[1]))

def standardize(x):
    return (x - x.mean()) / x.std(ddof=0)

