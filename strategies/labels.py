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


def compute_nextopen_limit_mask(kline_df: pd.DataFrame,
                                st_series: pd.Series | None = None) -> pd.Series:
    """Detect (date, code) pairs where the NEXT day's open is at a price limit.

    A position entered at T+1's open cannot execute if T+1 opens at:
      - limit-up   (cannot buy)
      - limit-down (cannot sell)

    Limit reference is T's close:
      - regular stocks: ±10%
      - ST stocks:      ±5%   (identified via st_series parameter)

    Parameters
    ----------
    kline_df : DataFrame
        Must contain columns: date, code, open, close.
    st_series : Series or None
        Boolean Series indexed by (date, code), True if stock is ST on that date.

    Returns
    -------
    Boolean Series with (date, code) MultiIndex.
    True means the observation on date T should be excluded from test IC
    because T+1's open is at a price limit.
    """
    df = kline_df[["date", "code", "open", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])

    # T+1's open aligned to T's row
    df["next_open"] = df.groupby("code")["open"].shift(-1)

    # ±10% limit check (all stocks)
    limit_up_10 = (df["close"] * 1.10).round(2)
    limit_down_10 = (df["close"] * 0.90).round(2)
    is_limit = (
        (df["next_open"] >= limit_up_10 - 0.005)
        | (df["next_open"] <= limit_down_10 + 0.005)
    )

    # ±5% ST limit check
    if st_series is not None:
        df = df.set_index(["date", "code"])
        df["is_st"] = st_series.reindex(df.index, fill_value=False)

        limit_up_5 = (df["close"] * 1.05).round(2)
        limit_down_5 = (df["close"] * 0.95).round(2)
        st_limit = (
            (df["next_open"] >= limit_up_5 - 0.005)
            | (df["next_open"] <= limit_down_5 + 0.005)
        ) & df["is_st"]
        is_limit.index = df.index
        is_limit = is_limit | st_limit
    else:
        df = df.set_index(["date", "code"])
        is_limit.index = df.index

    is_limit = is_limit.fillna(False)
    is_limit.name = "nextopen_limit"
    return is_limit
