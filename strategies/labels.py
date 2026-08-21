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


def compute_peak_high(
    kline_df: pd.DataFrame,
    start_day: int = 11,
    end_day: int = 20,
    delist_info: dict[str, pd.Timestamp] | None = None,
) -> pd.Series:
    """Max of daily highs over a forward window [T+start_day, T+end_day].

    Relative return: max_high / close[T] - 1.

    Parameters
    ----------
    kline_df : DataFrame
        Must contain columns: date, code, high, close.
    start_day, end_day : int
        Forward window (inclusive).
    delist_info : dict or None
        Mapping from code to delist_date.

    Returns
    -------
    Series with (date, code) MultiIndex.
    """
    df = kline_df[["date", "code", "close", "high"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])
    df = df.set_index(["date", "code"])

    def _peak(group):
        c = group["close"]
        h = group["high"]
        peaks = pd.concat(
            [h.shift(-d) for d in range(start_day, end_day + 1)],
            axis=1,
        )
        return peaks.max(axis=1) / c - 1.0

    fwd = df.groupby("code", group_keys=False).apply(_peak)
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


def compute_peak_close(
    kline_df: pd.DataFrame,
    start_day: int = 11,
    end_day: int = 20,
    delist_info: dict[str, pd.Timestamp] | None = None,
) -> pd.Series:
    """Max of daily closes over a forward window [T+start_day, T+end_day].

    Relative return: max_close / close[T] - 1.

    Parameters
    ----------
    kline_df : DataFrame
        Must contain columns: date, code, close.
    start_day, end_day : int
        Forward window (inclusive).
    delist_info : dict or None
        Mapping from code to delist_date.

    Returns
    -------
    Series with (date, code) MultiIndex.
    """
    df = kline_df[["date", "code", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])
    df = df.set_index(["date", "code"])

    def _peak(group):
        c = group["close"]
        peaks = pd.concat(
            [c.shift(-d) for d in range(start_day, end_day + 1)],
            axis=1,
        )
        return peaks.max(axis=1) / c - 1.0

    fwd = df.groupby("code", group_keys=False).apply(_peak)
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


def compute_median_close(
    kline_df: pd.DataFrame,
    start_day: int = 16,
    end_day: int = 20,
    delist_info: dict[str, pd.Timestamp] | None = None,
) -> pd.Series:
    """Median of daily closes over a forward window [T+start_day, T+end_day].

    Relative return: median_close / close[T] - 1.

    Parameters
    ----------
    kline_df : DataFrame
        Must contain columns: date, code, close.
    start_day, end_day : int
        Forward window (inclusive).
    delist_info : dict or None
        Mapping from code to delist_date.

    Returns
    -------
    Series with (date, code) MultiIndex.
    """
    df = kline_df[["date", "code", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])
    df = df.set_index(["date", "code"])

    def _median(group):
        c = group["close"]
        vals = pd.concat(
            [c.shift(-d) for d in range(start_day, end_day + 1)],
            axis=1,
        )
        return vals.median(axis=1) / c - 1.0

    fwd = df.groupby("code", group_keys=False).apply(_median)
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


def compute_smoothed_forward_returns(kline_df: pd.DataFrame, horizon: int = 20,
                                     delist_info: dict[str, pd.Timestamp] | None = None) -> pd.Series:
    """Compute smoothed forward returns using 6 price points around T+horizon.

    For horizon h, uses:
        avg(close[T+h-1], open[T+h-1], close[T+h], open[T+h], close[T+h+1], open[T+h+1])
        / close[T] - 1

    This reduces label noise by averaging over a 3-day window with both
    open and close prices.

    Parameters
    ----------
    kline_df : DataFrame
        Must contain columns: date, code, open, close.
        Sorted by (code, date).
    horizon : int
        Center day of the averaging window.
    delist_info : dict or None
        Mapping from code to delist_date (Timestamp).

    Returns
    -------
    Series with (date, code) MultiIndex.
    """
    df = kline_df[["date", "code", "open", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])
    df = df.set_index(["date", "code"])

    def _smoothed_return(group):
        c = group["close"]
        o = group["open"]
        # 6 price points: close[T+h-1], open[T+h-1], close[T+h], open[T+h], close[T+h+1], open[T+h+1]
        avg_future = (
            c.shift(-(horizon - 1)) +
            o.shift(-(horizon - 1)) +
            c.shift(-horizon) +
            o.shift(-horizon) +
            c.shift(-(horizon + 1)) +
            o.shift(-(horizon + 1))
        ) / 6.0
        return avg_future / c - 1.0

    fwd = df.groupby("code", group_keys=False).apply(_smoothed_return)
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


def compute_median_open(
    kline_df: pd.DataFrame,
    start_day: int = 4,
    end_day: int = 6,
    baseline: str = "next_open",
) -> pd.Series:
    """Median of daily opens over a forward window [T+start_day, T+end_day].

    Relative return: median_open / baseline - 1, where baseline is one of:
      - "next_open": open[T+1] — earliest executable entry after the T-close
        signal; aligns with the backtest's next-day-open fill
      - "open":      open[T]
      - "close":     close[T]

    Delisting semantics (spec docs/superpowers/specs/2026-08-20-... §3.2):
    partial windows keep the median over the opens that exist (they are real
    tradable exit prices and carry the pre-delist crash signal); a fully
    missing window yields NaN and is dropped by the caller's notna filter.
    No -1.0 fill is implemented — the legacy fill in compute_median_close
    never fired (its date >= delist_date mask matches no kline rows) and
    delisting risk is handled at the portfolio layer instead.

    Parameters
    ----------
    kline_df : DataFrame
        Must contain columns: date, code, open, close.
        Sorted by (code, date).
    start_day, end_day : int
        Forward window (inclusive).
    baseline : str
        "next_open" | "open" | "close".

    Returns
    -------
    Series with (date, code) MultiIndex.
    """
    if baseline not in ("next_open", "open", "close"):
        raise ValueError(f"baseline must be next_open|open|close, got {baseline!r}")

    df = kline_df[["date", "code", "open", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])
    df = df.set_index(["date", "code"])

    def _median(group):
        o = group["open"]
        vals = pd.concat(
            [o.shift(-d) for d in range(start_day, end_day + 1)],
            axis=1,
        )
        med = vals.median(axis=1)
        if baseline == "next_open":
            base = o.shift(-1)
        elif baseline == "open":
            base = o
        else:
            base = group["close"]
        return med / base - 1.0

    fwd = df.groupby("code", group_keys=False).apply(_median)
    fwd.name = "forward_ret"
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
        # 按列 merge 对齐（2026-08-21 修复）：此前用 MultiIndex reindex，
        # st_series 的 ns 级日期与 kline 的 µs 级日期哈希失配 → is_st 全
        # False → ±5% ST 子带静默失效（主库遗留缺陷）
        st_frame = st_series.rename("_st").reset_index()
        st_frame["date"] = pd.to_datetime(st_frame["date"])
        df = df.merge(st_frame, on=["date", "code"], how="left")
        df["_st"] = df["_st"].fillna(False).astype(bool)

        limit_up_5 = (df["close"] * 1.05).round(2)
        limit_down_5 = (df["close"] * 0.95).round(2)
        st_limit = (
            (df["next_open"] >= limit_up_5 - 0.005)
            | (df["next_open"] <= limit_down_5 + 0.005)
        ) & df["_st"]

        # merge 保序（键唯一），两侧同为 RangeIndex 时先 OR 再挂 MultiIndex
        is_limit = is_limit | st_limit
        df = df.drop(columns=["_st"]).set_index(["date", "code"])
        is_limit.index = df.index
    else:
        df = df.set_index(["date", "code"])
        is_limit.index = df.index

    is_limit = is_limit.fillna(False)
    is_limit.name = "nextopen_limit"
    return is_limit
