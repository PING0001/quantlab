"""
Forward returns computation for supervised learning labels.

Supports delisting-aware returns: for stocks in the delisting process,
forward returns beyond the last trading date are set to -1.0 to reflect
that delisted stock value goes to approximately zero.
"""
from __future__ import annotations

import pandas as pd


def compute_forward_returns(kline_df: pd.DataFrame, horizon: int = 5,
                            delist_info: dict[str, pd.Timestamp] | None = None) -> pd.Series:
    """Compute forward returns from a kline DataFrame.

    Parameters
    ----------
    kline_df : DataFrame
        Must contain columns: date, code, close.
        Sorted by (code, date).
    horizon : int
        Number of days forward.
    delist_info : dict or None
        Mapping from code to delist_date (Timestamp). For codes in this dict,
        NaN forward returns (caused by lookahead beyond the last trading day)
        are filled with -1.0 to reflect that delisted stocks go to zero.

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

    if delist_info:
        for code, delist_date in delist_info.items():
            if code not in fwd.index.get_level_values("code"):
                continue
            delist_date = pd.Timestamp(delist_date)
            code_mask = fwd.index.get_level_values("code") == code
            date_mask = fwd.index.get_level_values("date") >= delist_date
            mask = code_mask & date_mask
            fwd.loc[mask] = fwd.loc[mask].fillna(-1.0)

    return fwd
