"""
Forward returns computation for supervised learning labels.
"""
from __future__ import annotations

import pandas as pd


def compute_forward_returns(kline_df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Compute forward returns from a kline DataFrame.

    Parameters
    ----------
    kline_df : DataFrame
        Must contain columns: date, code, close.
        Sorted by (code, date).
    horizon : int
        Number of days forward.

    Returns
    -------
    Series with (date, code) MultiIndex.
    """
    df = kline_df[["date", "code", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])
    df = df.set_index(["date", "code"])

    fwd = df.groupby("code")["close"].transform(
        lambda x: x.shift(-horizon) / x - 1.0
    )
    fwd.name = "forward_ret"
    return fwd
