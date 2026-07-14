# -*- coding: utf-8 -*-
"""
Non-alpha factors computed from raw OHLCV data and supplementary tables.
Also provides IndNeutralize post-processing for alpha factors.
"""

import numpy as np
import polars as pl

from . import alpha101


def apply_ind_neutralize(
    alpha_df: pl.DataFrame,
    industry_df: pl.DataFrame,
) -> pl.DataFrame:
    """Apply industry neutralization to the 18 alpha factors that require it."""
    # Join on vt_symbol only (industry classification is time-invariant snapshot)
    df = alpha_df.join(industry_df, on="vt_symbol", how="left")

    for col in alpha101.IND_NEUTRALIZE_ALPHAS:
        if col not in df.columns:
            continue
        df = df.with_columns(
            (pl.col(col) - pl.col(col).mean().over(["datetime", "sw_l3_code"])).alias(col)
        )

    return df.select(alpha_df.columns)


def compute_non_alpha_factors(df_long: pl.DataFrame) -> pl.DataFrame:
    """
    Compute non-alpha factors from OHLCV data.

    Parameters
    ----------
    df_long : pl.DataFrame
        Long-format with columns:
        [datetime, vt_symbol, open, high, low, close, volume,
         total_mv, circ_mv, list_date, vwap]

    Returns
    -------
    pl.DataFrame with columns [datetime, vt_symbol, factor1, factor2, ...]
    """
    cs = pl.col("close")
    os = pl.col("open")
    hs = pl.col("high")
    ls = pl.col("low")
    vs = pl.col("volume")
    sym = pl.col("vt_symbol")

    # Keep all source columns for computation, will select factor columns at end
    result = df_long.sort(["datetime", "vt_symbol"])

    # ---- momentum ----
    result = result.with_columns([
        (cs / cs.shift(5).over("vt_symbol") - 1).alias("Return_5d"),
        (cs / cs.shift(20).over("vt_symbol") - 1).alias("Return_20d"),
        (-(cs / cs.shift(60).over("vt_symbol") - 1)).alias("Reversal_60d"),
    ])

    # ---- intraday ----
    result = result.with_columns([
        ((os - cs.shift(1).over("vt_symbol")) / cs.shift(1).over("vt_symbol")).alias("Gap_pct"),
        ((hs - ls) / os).alias("Intraday_range_pct"),
        ((cs - os) / os).alias("Intraday_return"),
    ])

    # ---- returns for volatility ----
    ret1d = (cs / cs.shift(1).over("vt_symbol") - 1)
    result = result.with_columns(ret1d.alias("_ret1d"))

    # ---- volatility ----
    result = result.with_columns([
        (pl.col("_ret1d").rolling_std(20, min_samples=1).over("vt_symbol")).alias("Volatility"),
        (pl.col("_ret1d").rolling_std(60, min_samples=1).over("vt_symbol")).alias("Volatility_60d"),
    ])

    # ---- ATR (Average True Range, 14-day) ----
    tr = pl.max_horizontal(
        hs - ls,
        (hs - cs.shift(1).over("vt_symbol")).abs(),
        (ls - cs.shift(1).over("vt_symbol")).abs(),
    )
    result = result.with_columns(tr.alias("_tr"))
    # Exponential smoothing for ATR (Wilder's method approximation)
    atr = pl.col("_tr").rolling_mean(14, min_samples=1).over("vt_symbol")
    result = result.with_columns(atr.alias("ATR"))

    # ---- Bollinger Band width ----
    ma20 = cs.rolling_mean(20, min_samples=1).over("vt_symbol")
    std20 = cs.rolling_std(20, min_samples=1).over("vt_symbol")
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    result = result.with_columns(((upper - lower) / ma20).alias("Bollinger_width"))

    # ---- Price Position 252d ----
    c_min_252 = cs.rolling_min(252, min_samples=1).over("vt_symbol")
    c_max_252 = cs.rolling_max(252, min_samples=1).over("vt_symbol")
    result = result.with_columns(
        ((cs - c_min_252) / (c_max_252 - c_min_252 + 1e-10)).alias("Price_position_252d")
    )

    # ---- Stochastic K (14d) ----
    min_low_14 = ls.rolling_min(14, min_samples=1).over("vt_symbol")
    max_high_14 = hs.rolling_max(14, min_samples=1).over("vt_symbol")
    result = result.with_columns(
        ((cs - min_low_14) / (max_high_14 - min_low_14 + 1e-10)).alias("Stochastic_K")
    )

    # ---- SMA (20d) ----
    result = result.with_columns(ma20.alias("SMA"))

    # ---- MACD Signal (12, 26, 9) ----
    ema12 = cs.ewm_mean(span=12, min_periods=1).over("vt_symbol")
    ema26 = cs.ewm_mean(span=26, min_periods=1).over("vt_symbol")
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm_mean(span=9, min_periods=1).over("vt_symbol")
    result = result.with_columns((macd_line - signal_line).alias("MACD_signal"))

    # ---- Return Skew (20d) ----
    result = result.with_columns(
        pl.col("_ret1d").rolling_skew(20, min_samples=5).over("vt_symbol").alias("Return_skew_20d")
    )

    # ---- Trend Strength (correlation of close with linear ramp, 20d) ----
    # Approximate as ts_rsquare of close
    n = 20
    sum_x2 = (n - 1) * n * (2 * n - 1) / 6
    mean_x = (n - 1) / 2
    var_x = sum_x2 / n - mean_x * mean_x
    sum_xy_expr = pl.sum_horizontal([
        (n - 1 - j) * cs.shift(j) for j in range(n)
    ])
    df_tmp = result.with_columns([
        cs.rolling_sum(n, min_samples=n).over("vt_symbol").alias("sum_y"),
        cs.rolling_var(n, min_samples=n, ddof=0).over("vt_symbol").alias("var_y"),
        sum_xy_expr.over("vt_symbol").alias("sum_xy"),
    ])
    df_tmp = df_tmp.with_columns((pl.col("sum_y") / n).alias("mean_y"))
    df_tmp = df_tmp.with_columns(
        (pl.col("sum_xy") / n - mean_x * pl.col("mean_y")).alias("cov_xy")
    )
    df_tmp = df_tmp.select([
        pl.col("datetime"), pl.col("vt_symbol"),
        (pl.col("cov_xy").pow(2) / (var_x * pl.col("var_y"))).alias("Trend_strength")
    ])
    df_tmp = df_tmp.with_columns(
        pl.when(pl.col("Trend_strength").is_infinite() | pl.col("Trend_strength").is_nan())
        .then(None)
        .otherwise(pl.col("Trend_strength"))
        .alias("Trend_strength")
    )
    result = result.join(df_tmp, on=["datetime", "vt_symbol"], how="left")

    # ---- Volume Ratio ----
    result = result.with_columns(
        (vs / vs.rolling_mean(20, min_samples=1).over("vt_symbol")).alias("Volume_ratio")
    )

    # ---- Amihud Illiquidity ----
    result = result.with_columns(
        (pl.col("_ret1d").abs() / (vs + 1e-10)).alias("Amihud_illiquidity")
    )

    # ---- Avg Amount 90d (approximate: volume * close) ----
    result = result.with_columns(
        (vs * cs).rolling_mean(90, min_samples=1).over("vt_symbol").alias("AvgAmount_90d")
    )

    # ---- LnMktCap / LnFloatCap ----
    result = result.with_columns([
        pl.col("total_mv").log().alias("LnMktCap"),
        pl.col("circ_mv").log().alias("LnFloatCap"),
    ])

    # ---- Turnover ----
    turnover = vs * cs / pl.col("circ_mv")
    result = result.with_columns(turnover.alias("_turnover_1d"))
    result = result.with_columns([
        pl.col("_turnover_1d").rolling_mean(3, min_samples=1).over("vt_symbol").alias("Turnover_3d"),
    ])
    result = result.with_columns([
        (pl.col("Turnover_3d") / pl.col("Turnover_3d").rolling_mean(20, min_samples=1).over("vt_symbol")).alias("Turnover_3d_ratio"),
    ])

    # ---- Cross-sectional Rank factors ----
    result = result.with_columns([
        pl.col("_ret1d").rank().over("datetime").alias("Return_1d_rank"),
        pl.col("Return_20d").rank().over("datetime").alias("Return_20d_rank"),
        pl.col("Turnover_3d").rank().over("datetime").alias("Turnover_3d_rank"),
    ])

    # ---- LnAge (trading days since list_date) ----
    result = result.with_columns(
        pl.col("list_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("_list_dt")
    )
    result = result.with_columns(
        (pl.col("datetime").cast(pl.Date) - pl.col("_list_dt")).dt.total_days().alias("_age_days")
    )
    result = result.with_columns(
        pl.when(pl.col("_age_days") > 0)
        .then(pl.col("_age_days").cast(pl.Float64).log())
        .otherwise(None)
        .alias("LnAge")
    )

    # ---- drop intermediate columns and keep only factor columns ----
    intermediate_cols = ["_ret1d", "_tr", "_turnover_1d", "_list_dt", "_age_days"]
    source_cols = ["open", "high", "low", "close", "volume", "vwap",
                   "total_mv", "circ_mv", "cap", "list_date", "pct_chg"]

    factor_cols = [c for c in result.columns
                   if c not in ("datetime", "vt_symbol") and
                   c not in source_cols and
                   c not in intermediate_cols]

    result = result.select(["datetime", "vt_symbol"] + factor_cols)
    result = result.drop([c for c in intermediate_cols if c in result.columns])

    return result
