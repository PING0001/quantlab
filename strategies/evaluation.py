"""
IC (Information Coefficient) computation for cross-sectional strategies.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def rank_ic(predictions: pd.Series, returns: pd.Series) -> pd.Series:
    """
    Cross-sectional Rank IC (Spearman) per date.

    Both arguments are indexed by (date, code).  Returns a Series indexed by date.
    """
    combined = pd.DataFrame({"pred": predictions, "ret": returns}).dropna()
    if combined.empty:
        return pd.Series(dtype=float)

    def _spearman(g: pd.DataFrame) -> float:
        if len(g) < 5:
            return np.nan
        return g["pred"].rank().corr(g["ret"].rank())

    return combined.groupby("date").apply(_spearman).dropna()


def pearson_ic(predictions: pd.Series, returns: pd.Series) -> pd.Series:
    """
    Cross-sectional Pearson IC per date.

    Both arguments are indexed by (date, code).  Returns a Series indexed by date.
    """
    combined = pd.DataFrame({"pred": predictions, "ret": returns}).dropna()
    if combined.empty:
        return pd.Series(dtype=float)

    return combined.groupby("date").apply(lambda g: g["pred"].corr(g["ret"]))


def ic_summary(ic_series: pd.Series) -> dict:
    """
    Summarise an IC time-series.

    Returns a dict with mean_ic, std_ic, ir (information ratio),
    hit_rate, n_periods, min_ic, max_ic.
    """
    ic = ic_series.dropna()
    if len(ic) == 0:
        return {"n_periods": 0}
    std = ic.std()
    return {
        "mean_ic": float(ic.mean()),
        "std_ic": float(std),
        "ir": float(ic.mean() / std) if std > 0 else 0.0,
        "hit_rate": float((ic > 0).mean()),
        "n_periods": len(ic),
        "min_ic": float(ic.min()),
        "max_ic": float(ic.max()),
    }
