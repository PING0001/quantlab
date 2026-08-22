# -*- coding: utf-8 -*-
"""
评估器回归基准因子：8 个经典 alpha 表达式的原生 Polars 重实现。

背景（2026-08-22 决策）：alpha101 全系及其 vnpy DSL 引擎已从项目移除，
不再入模；仅保留本模块 8 个经典表达式作为**评估器回归基准**——评估
口径（IC 计算 / select_factors / 未来 factor foundry 准入）变更时，跑
`python -m factors.baseline_check` 验证基准 IC 未漂移。

公式出处：Kakushadze (2016)《101 Formulaic Alphas》（document/alpha101.md）
及已删除的 vnpy 移植版 alpha101.py，表达式原文存档于 BASELINE_EXPRESSIONS。
实现与原 DSL 算子语义逐一对齐（2026-08-22 全池交叉验证 |Pearson|>0.999）：

- ts_std(x, w)   -> rolling_std(w, min_samples=1, ddof=0)
- ts_corr(a,b,w) -> pl.rolling_corr(w, min_samples=1)；inf -> null
- ts_rank(x, w)  -> 过去 w-1 个值中严格小于当前值的占比（分母为非空
                    shifted 计数；为 0 时置 0）；inf/nan -> null
- ts_argmax(x,w) -> 窗口内最大值位置（1-based，np.argmax 首现）
- cs_rank(x)     -> 同日截面 rank()/count() 百分位（average 法，null 不参与）

输入约定：与 compute._load_ohlcv 相同的宽表，至少含
[datetime, vt_symbol, open, high, low, close, volume, vwap]，
本模块内部自排 (vt_symbol, datetime)。
"""

from __future__ import annotations

import numpy as np
import polars as pl

BASELINE_FACTORS = [
    "alpha1_v0", "alpha18_v0", "alpha50_v0", "alpha60_v0",
    "alpha6", "alpha40", "alpha42", "alpha101",
]

# 原表达式存档（自 alpha101.py 抄录，DSL 已删除）
BASELINE_EXPRESSIONS = {
    "alpha1_v0": "cs_rank(ts_argmax(pow1(ts_corr(close, volume, 5), 2.0), 5) - 0.5)",
    "alpha18_v0": "-1 * ((ts_std(abs(close - open), 5) + (close - open)) + ts_corr(close, open, 10))",
    "alpha50_v0": "cs_rank(-1 * ts_corr(ts_rank(close, 10), ts_rank(volume, 10), 10))",
    "alpha60_v0": "-1 * (ts_rank(ts_std(close, 20), 10) - ts_rank(ts_std(close, 5), 10))",
    "alpha6": "(-1) * ts_corr(open, volume, 10)",
    "alpha40": "((-1) * cs_rank(ts_std(high, 10))) * ts_corr(high, volume, 10)",
    "alpha42": "cs_rank((vwap - close)) / cs_rank((vwap + close))",
    "alpha101": "((close - open) / ((high - low) + 0.001))",
}


# ---- DSL 算子等价实现（返回 pl.Expr，over("vt_symbol") 在表达式整体外层）----

def _ts_std(col: str, window: int) -> pl.Expr:
    return pl.col(col).rolling_std(window, min_samples=1, ddof=0).over("vt_symbol")


def _ts_rank(col: str, window: int) -> pl.Expr:
    """ts_ops.ts_rank 等价：count(shift_i < cur, i=1..w-1) / count(非空 shift)。"""
    lt = pl.lit(0, dtype=pl.Int32)
    cnt = pl.lit(0, dtype=pl.Int32)
    for i in range(1, window):
        shifted = pl.col(col).shift(i)
        lt = lt + (shifted < pl.col(col)).cast(pl.Int32)
        cnt = cnt + shifted.is_not_null().cast(pl.Int32)
    rank_expr = lt / pl.when(cnt > 0).then(cnt).otherwise(1)
    rank_expr = pl.when(rank_expr.is_infinite() | rank_expr.is_nan()) \
        .then(None).otherwise(rank_expr)
    return rank_expr.over("vt_symbol")


def _ts_argmax(col: str, window: int) -> pl.Expr:
    """ts_ops.ts_argmax 等价：窗口内最大值 1-based 位次（首现）。"""
    return pl.col(col).rolling_map(
        lambda s: int(np.argmax(s.to_numpy())) + 1, window
    ).over("vt_symbol")


def _cs_rank(col: str) -> pl.Expr:
    """cs_ops.cs_rank 等价：同日截面百分位 (0,1]。"""
    return pl.col(col).rank().over("datetime") / pl.col(col).count().over("datetime")


def _rolling_corr(df: pl.DataFrame, a: str, b: str, window: int, out: str) -> pl.DataFrame:
    """ts_ops.ts_corr 等价（需物化输入列后按名计算）。"""
    df = df.with_columns(
        pl.rolling_corr(a, b, window_size=window, min_samples=1)
        .over("vt_symbol").alias(out)
    )
    return df.with_columns(
        pl.when(pl.col(out).is_infinite()).then(None)
        .otherwise(pl.col(out)).alias(out)
    )


def _clean(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    return df.with_columns([
        pl.when(pl.col(c).is_infinite() | pl.col(c).is_nan())
        .then(None).otherwise(pl.col(c)).alias(c)
        for c in cols
    ])


# ---- 因子实现 ----

def compute_baseline_factors(ohlcv: pl.DataFrame) -> pl.DataFrame:
    """在 OHLCV 宽表上追加 8 个基准因子列，返回原列 + 基准列。"""
    df = ohlcv.sort(["vt_symbol", "datetime"])

    # alpha1_v0
    df = _rolling_corr(df, "close", "volume", 5, "_a1_c")
    df = df.with_columns((pl.col("_a1_c") ** 2.0).alias("_a1_x"))
    df = df.with_columns(_ts_argmax("_a1_x", 5).alias("_a1_arg"))
    df = df.with_columns((pl.col("_a1_arg") - 0.5).alias("_a1_arg2"))
    df = df.with_columns(_cs_rank("_a1_arg2").alias("alpha1_v0"))

    # alpha18_v0
    df = df.with_columns((pl.col("close") - pl.col("open")).abs().alias("_a18_do"))
    df = df.with_columns(_ts_std("_a18_do", 5).alias("_a18_std"))
    df = _rolling_corr(df, "close", "open", 10, "_a18_c")
    df = df.with_columns(
        (-1 * ((pl.col("_a18_std") + (pl.col("close") - pl.col("open")))
               + pl.col("_a18_c"))).alias("alpha18_v0")
    )

    # alpha50_v0
    df = df.with_columns(_ts_rank("close", 10).alias("_a50_rc"))
    df = df.with_columns(_ts_rank("volume", 10).alias("_a50_rv"))
    df = _rolling_corr(df, "_a50_rc", "_a50_rv", 10, "_a50_c")
    df = df.with_columns((-1 * pl.col("_a50_c")).alias("_a50_nc"))
    df = df.with_columns(_cs_rank("_a50_nc").alias("alpha50_v0"))

    # alpha60_v0
    df = df.with_columns(_ts_std("close", 20).alias("_a60_s20"))
    df = df.with_columns(_ts_std("close", 5).alias("_a60_s5"))
    df = df.with_columns(_ts_rank("_a60_s20", 10).alias("_a60_r20"))
    df = df.with_columns(_ts_rank("_a60_s5", 10).alias("_a60_r5"))
    df = df.with_columns(
        (-1 * (pl.col("_a60_r20") - pl.col("_a60_r5"))).alias("alpha60_v0")
    )

    # alpha6
    df = _rolling_corr(df, "open", "volume", 10, "_a6_c")
    df = df.with_columns((-1 * pl.col("_a6_c")).alias("alpha6"))

    # alpha40
    df = df.with_columns(_ts_std("high", 10).alias("_a40_s"))
    df = df.with_columns(_cs_rank("_a40_s").alias("_a40_r"))
    df = _rolling_corr(df, "high", "volume", 10, "_a40_c")
    df = df.with_columns(
        (((-1) * pl.col("_a40_r")) * pl.col("_a40_c")).alias("alpha40")
    )

    # alpha42
    df = df.with_columns((pl.col("vwap") - pl.col("close")).alias("_a42_d"))
    df = df.with_columns((pl.col("vwap") + pl.col("close")).alias("_a42_s"))
    df = df.with_columns(_cs_rank("_a42_d").alias("_a42_rd"))
    df = df.with_columns(_cs_rank("_a42_s").alias("_a42_rs"))
    df = df.with_columns((pl.col("_a42_rd") / pl.col("_a42_rs")).alias("alpha42"))

    # alpha101
    df = df.with_columns(
        ((pl.col("close") - pl.col("open"))
         / ((pl.col("high") - pl.col("low")) + 0.001)).alias("alpha101")
    )

    df = _clean(df, BASELINE_FACTORS)
    drop_cols = [c for c in df.columns if c.startswith(("_a1_", "_a18_", "_a50_",
                                                         "_a60_", "_a6_", "_a40_",
                                                         "_a42_"))]
    return df.drop(drop_cols)
