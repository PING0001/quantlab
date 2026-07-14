# -*- coding: utf-8 -*-
"""
Expression evaluator: executes string expressions using the DSL operators.
Ported from vnpy alpha dataset utility.py.
"""

import polars as pl

from .ops import DataProxy, EXPRESSION_FUNCTIONS


def calculate_by_expression(df: pl.DataFrame, expression: str) -> pl.DataFrame:
    from .ts_ops import (  # noqa: F401
        ts_delay, ts_min, ts_max, ts_argmax, ts_argmin,
        ts_rank, ts_sum, ts_mean, ts_std, ts_slope, ts_quantile,
        ts_rsquare, ts_resi, ts_corr,
        ts_less, ts_greater, ts_log, ts_abs,
        ts_delta, ts_cov, ts_decay_linear, ts_product
    )
    from .cs_ops import (  # noqa: F401
        cs_rank, cs_mean, cs_std, cs_sum, cs_scale
    )
    from .math_ops import (  # noqa: F401
        less, greater, log, abs,
        sign, pow1, pow2, quesval, quesval2
    )

    d: dict = locals()
    d.update(EXPRESSION_FUNCTIONS)

    for column in df.columns:
        if column in {"datetime", "vt_symbol"}:
            continue
        column_df = df[["datetime", "vt_symbol", column]]
        d[column] = DataProxy(column_df)

    other: DataProxy = eval(expression, {}, d)
    return other.df
