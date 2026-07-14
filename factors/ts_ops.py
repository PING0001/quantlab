# -*- coding: utf-8 -*-
"""
Time-series operators for the expression DSL.
Ported from vnpy alpha dataset ts_function.py.
All functions accept and return DataProxy objects.
"""

from typing import cast

import numpy as np
import polars as pl
from scipy import stats

from .ops import DataProxy


def ts_delay(feature: DataProxy, window: int) -> DataProxy:
    df: pl.DataFrame = feature.df.select([
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").shift(window).over("vt_symbol")
    ])
    return DataProxy(df)


def ts_min(feature: DataProxy, window: int) -> DataProxy:
    df: pl.DataFrame = feature.df.select([
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_min(window, min_samples=1).over("vt_symbol")
    ])
    return DataProxy(df)


def ts_max(feature: DataProxy, window: int) -> DataProxy:
    df: pl.DataFrame = feature.df.select([
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_max(window, min_samples=1).over("vt_symbol")
    ])
    return DataProxy(df)


def ts_argmax(feature: DataProxy, window: int) -> DataProxy:
    df: pl.DataFrame = feature.df.select([
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_map(
            lambda s: int(np.argmax(s.to_numpy())) + 1, window
        ).over("vt_symbol")
    ])
    return DataProxy(df)


def ts_argmin(feature: DataProxy, window: int) -> DataProxy:
    df: pl.DataFrame = feature.df.select([
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_map(
            lambda s: int(np.argmin(s.to_numpy())) + 1, window
        ).over("vt_symbol")
    ])
    return DataProxy(df)


def ts_rank(feature: DataProxy, window: int) -> DataProxy:
    """Time-series percentile rank: fraction of last N values < current value.
    Implemented with vectorized shift+compare instead of rolling_map for speed."""
    if window <= 1:
        return feature.result(pl.Series("data", [0.5] * len(feature.df)))

    # Build: count(first_N-1_shifted < current) / count(non-null)
    lt_accum = pl.lit(0).cast(pl.Int32)
    cnt_accum = pl.lit(0).cast(pl.Int32)
    for i in range(1, window):
        shifted = pl.col("data").shift(i)
        lt_accum = lt_accum + (shifted < pl.col("data")).cast(pl.Int32)
        cnt_accum = cnt_accum + shifted.is_not_null().cast(pl.Int32)

    rank_col = lt_accum / pl.when(cnt_accum > 0).then(cnt_accum).otherwise(1)

    df = feature.df.select([
        pl.col("datetime"),
        pl.col("vt_symbol"),
        rank_col.over("vt_symbol").alias("data")
    ])

    df = df.with_columns(
        pl.when(pl.col("data").is_infinite() | pl.col("data").is_nan())
        .then(None)
        .otherwise(pl.col("data"))
        .alias("data")
    )

    return DataProxy(df)


def ts_sum(feature: DataProxy, window: int) -> DataProxy:
    df: pl.DataFrame = feature.df.select([
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_sum(window).over("vt_symbol")
    ])
    return DataProxy(df)


def ts_mean(feature: DataProxy, window: int) -> DataProxy:
    df: pl.DataFrame = feature.df.select([
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_mean(window, min_samples=1).over("vt_symbol")
    ])
    return DataProxy(df)


def ts_std(feature: DataProxy, window: int) -> DataProxy:
    df: pl.DataFrame = feature.df.select([
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_std(window, min_samples=1, ddof=0).over("vt_symbol")
    ])
    return DataProxy(df)


def ts_slope(feature: DataProxy, window: int) -> DataProxy:
    n = window
    sum_x = n * (n - 1) / 2
    sum_x2 = (n - 1) * n * (2 * n - 1) / 6
    denominator = n * sum_x2 - sum_x * sum_x

    sum_xy_expr: pl.Expr = pl.sum_horizontal([
        (window - 1 - j) * pl.col("data").shift(j)
        for j in range(window)
    ])

    df: pl.DataFrame = feature.df.with_columns([
        pl.col("data").rolling_sum(window, min_samples=window).over("vt_symbol").alias("sum_y"),
        sum_xy_expr.over("vt_symbol").alias("sum_xy")
    ])
    df = df.select([
        pl.col("datetime"),
        pl.col("vt_symbol"),
        ((n * pl.col("sum_xy") - sum_x * pl.col("sum_y")) / denominator).alias("data")
    ])
    return DataProxy(df)


def ts_quantile(feature: DataProxy, window: int, quantile: float) -> DataProxy:
    parts = []
    for i in range(window):
        parts.append(pl.col("data").shift(i))
    df: pl.DataFrame = feature.df.with_columns(
        pl.concat_list(parts).over("vt_symbol").alias("_w")
    )
    df = df.with_columns(
        pl.col("_w").list.eval(
            pl.element().sort().get(int(quantile * (pl.element().count() - 1)))
        ).alias("data")
    )
    df = df.select(["datetime", "vt_symbol", "data"])
    return DataProxy(df)


def ts_rsquare(feature: DataProxy, window: int) -> DataProxy:
    n = window
    sum_x2 = (n - 1) * n * (2 * n - 1) / 6
    mean_x = (n - 1) / 2
    var_x = sum_x2 / n - mean_x * mean_x

    sum_xy_expr: pl.Expr = pl.sum_horizontal([
        (window - 1 - j) * pl.col("data").shift(j)
        for j in range(window)
    ])

    df: pl.DataFrame = feature.df.with_columns([
        pl.col("data").rolling_sum(window, min_samples=window).over("vt_symbol").alias("sum_y"),
        pl.col("data").rolling_var(window, min_samples=window, ddof=0).over("vt_symbol").alias("var_y"),
        sum_xy_expr.over("vt_symbol").alias("sum_xy")
    ])
    df = df.with_columns((pl.col("sum_y") / n).alias("mean_y"))
    df = df.with_columns((pl.col("sum_xy") / n - mean_x * pl.col("mean_y")).alias("cov_xy"))
    df = df.select([
        pl.col("datetime"),
        pl.col("vt_symbol"),
        (pl.col("cov_xy").pow(2) / (var_x * pl.col("var_y"))).alias("data")
    ])
    df = df.with_columns(
        pl.when(pl.col("data").is_infinite() | pl.col("data").is_nan())
        .then(None)
        .otherwise(pl.col("data"))
        .alias("data")
    )
    return DataProxy(df)


def ts_resi(feature: DataProxy, window: int) -> DataProxy:
    n = window
    sum_x = n * (n - 1) / 2
    sum_x2 = (n - 1) * n * (2 * n - 1) / 6
    mean_x = (n - 1) / 2
    denominator = n * sum_x2 - sum_x * sum_x

    sum_xy_expr: pl.Expr = pl.sum_horizontal([
        (window - 1 - j) * pl.col("data").shift(j)
        for j in range(window)
    ])

    df: pl.DataFrame = feature.df.with_columns([
        pl.col("data").rolling_sum(window, min_samples=window).over("vt_symbol").alias("sum_y"),
        sum_xy_expr.over("vt_symbol").alias("sum_xy")
    ])
    df = df.with_columns([
        ((n * pl.col("sum_xy") - sum_x * pl.col("sum_y")) / denominator).alias("slope"),
        (pl.col("sum_y") / n).alias("mean_y"),
    ])
    df = df.with_columns(
        (pl.col("mean_y") - pl.col("slope") * mean_x).alias("intercept")
    )
    df = df.select([
        pl.col("datetime"),
        pl.col("vt_symbol"),
        (pl.col("data") - (pl.col("slope") * (n - 1) + pl.col("intercept"))).alias("data")
    ])
    return DataProxy(df)


def ts_corr(feature1: DataProxy, feature2: DataProxy, window: int) -> DataProxy:
    df_merged: pl.DataFrame = feature1.df.join(feature2.df, on=["datetime", "vt_symbol"])
    df: pl.DataFrame = df_merged.select([
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.rolling_corr("data", "data_right", window_size=window, min_samples=1)
        .over("vt_symbol").alias("data")
    ])
    df = df.with_columns(
        pl.when(pl.col("data").is_infinite()).then(None).otherwise(pl.col("data")).alias("data")
    )
    return DataProxy(df)


def ts_less(feature1: DataProxy, feature2: DataProxy | float) -> DataProxy:
    if isinstance(feature2, DataProxy):
        df_merged: pl.DataFrame = feature1.df.join(feature2.df, on=["datetime", "vt_symbol"])
    else:
        df_merged = feature1.df.with_columns(pl.lit(feature2).alias("data_right"))
    df: pl.DataFrame = df_merged.select([
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.min_horizontal("data", "data_right").over("vt_symbol").alias("data")
    ])
    return DataProxy(df)


def ts_greater(feature1: DataProxy, feature2: DataProxy | float) -> DataProxy:
    if isinstance(feature2, DataProxy):
        df_merged: pl.DataFrame = feature1.df.join(feature2.df, on=["datetime", "vt_symbol"])
    else:
        df_merged = feature1.df.with_columns(pl.lit(feature2).alias("data_right"))
    df: pl.DataFrame = df_merged.select([
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.max_horizontal("data", "data_right").over("vt_symbol").alias("data")
    ])
    return DataProxy(df)


def ts_log(feature: DataProxy) -> DataProxy:
    df: pl.DataFrame = feature.df.select([
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").log().over("vt_symbol")
    ])
    return DataProxy(df)


def ts_abs(feature: DataProxy) -> DataProxy:
    df: pl.DataFrame = feature.df.select([
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").abs().over("vt_symbol")
    ])
    return DataProxy(df)


def ts_delta(feature: DataProxy, window: int) -> DataProxy:
    return feature - ts_delay(feature, window)


def ts_cov(feature1: DataProxy, feature2: DataProxy, window: int) -> DataProxy:
    return ts_corr(feature1, feature2, window) * ts_std(feature1, window) * ts_std(feature2, window)


def ts_decay_linear(feature: DataProxy, window: int) -> DataProxy:
    denom = window * (window + 1) // 2
    parts = []
    for i in range(window):
        w = window - i
        parts.append(pl.col("data").shift(i).fill_null(0) * (w / denom))
    total = parts[0]
    for p in parts[1:]:
        total = total + p
    df: pl.DataFrame = feature.df.select([
        pl.col("datetime"),
        pl.col("vt_symbol"),
        total.over("vt_symbol").alias("data")
    ])
    return DataProxy(df)


def ts_product(feature: DataProxy, window: int) -> DataProxy:
    parts = []
    for i in range(window):
        parts.append(pl.col("data").shift(i).fill_null(1))
    total = parts[0]
    for p in parts[1:]:
        total = total * p
    df: pl.DataFrame = feature.df.select([
        pl.col("datetime"),
        pl.col("vt_symbol"),
        total.over("vt_symbol").alias("data")
    ])
    return DataProxy(df)
