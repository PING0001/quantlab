"""
Forward returns computation for supervised learning labels.
"""
from __future__ import annotations

import pandas as pd


def compute_median_open(
    kline_df: pd.DataFrame,
    start_day: int = 4,
    end_day: int = 6,
    baseline: str = "next_open",
) -> pd.Series:
    """Median of daily opens over a forward window [T+start_day, T+end_day].

    Relative return: median_open / baseline - 1, where baseline is one of:
      - "next_open": open[T+1] - earliest executable entry after the T-close
        signal; aligns with the backtest's next-day-open fill
      - "open":      open[T]
      - "close":     close[T]

    Delisting semantics (spec docs/superpowers/specs/2026-08-20-... §3.2):
    partial windows keep the median over the opens that exist (they are real
    tradable exit prices and carry the pre-delist crash signal); a fully
    missing window yields NaN and is dropped by the caller's notna filter.
    No -1.0 fill is implemented - the legacy fill in compute_median_close
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
    # 2026-08-22 审计 F1 修复：改纯比率判断（0.05% 容差）。原实现对 qfq 价做
    # round(2)±0.005 的价格网格判断--qfq 不在原始价 0.01 网格上，adj≠1 的
    # 股票近板开盘会误分类。比率口径下 adj 因子分子分母相消，与真实涨跌幅
    # 一致（除权日为复权收益，仍是最接近"真实可交易回报"的口径）
    _tol = 0.0005
    _close_ok = df["close"].notna() & (df["close"] > 0)
    _ratio = df["next_open"] / df["close"]
    is_limit = (
        (_ratio >= 1.10 - _tol) | (_ratio <= 0.90 + _tol)
    ) & _close_ok & df["next_open"].notna()

    # ±5% ST limit check
    if st_series is not None:
        # 按列 merge 对齐（2026-08-21 修复）：此前用 MultiIndex reindex，
        # st_series 的 ns 级日期与 kline 的 µs 级日期哈希失配 -> is_st 全
        # False -> ±5% ST 子带静默失效（主库遗留缺陷）
        st_frame = st_series.rename("_st").reset_index()
        st_frame["date"] = pd.to_datetime(st_frame["date"])
        df = df.merge(st_frame, on=["date", "code"], how="left")
        df["_st"] = df["_st"].fillna(False).astype(bool)

        st_limit = (
            (_ratio >= 1.05 - _tol) | (_ratio <= 0.95 + _tol)
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
