# -*- coding: utf-8 -*-
"""
Cross-sectional operators for the expression DSL.
Ported from vnpy alpha dataset cs_function.py.
"""

import polars as pl

from .ops import DataProxy


def cs_rank(feature: DataProxy) -> DataProxy:
    """Cross-sectional percentile rank (0~1)."""
    df: pl.DataFrame = feature.df.select([
        pl.col("datetime"),
        pl.col("vt_symbol"),
        (pl.col("data").rank().over("datetime")
         / pl.col("data").count().over("datetime")).alias("data")
    ])
    return DataProxy(df)


def cs_mean(feature: DataProxy) -> DataProxy:
    df: pl.DataFrame = feature.df.select([
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").mean().over("datetime")
    ])
    return DataProxy(df)


def cs_std(feature: DataProxy) -> DataProxy:
    df: pl.DataFrame = feature.df.select([
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").std().over("datetime")
    ])
    return DataProxy(df)


def cs_sum(feature: DataProxy) -> DataProxy:
    df: pl.DataFrame = feature.df.select([
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").sum().over("datetime")
    ])
    return DataProxy(df)


def cs_scale(feature: DataProxy) -> DataProxy:
    abs_feature = abs(feature)
    sum_abs = cs_sum(abs_feature)
    df_merged: pl.DataFrame = feature.df.join(sum_abs.df, on=["datetime", "vt_symbol"], suffix="_sum")
    df: pl.DataFrame = df_merged.with_columns(
        pl.when(pl.col("data_sum") != 0)
        .then(pl.col("data") / pl.col("data_sum"))
        .otherwise(0)
        .alias("data")
    ).select(["datetime", "vt_symbol", "data"])
    return DataProxy(df)
