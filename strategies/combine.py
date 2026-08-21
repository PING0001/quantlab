# -*- coding: utf-8 -*-
"""
Combined score for the dual-regression models（v4 口径 + v8 三模型版）.

单一分数，无百分位层。v8（2026-08-22 用户裁定）三模型融合：

    score = 0.4 * pred_2d + 0.35 * pred_6d + 0.25 * pred_20d

- 三个模型统一 next_open 锚（收益均自 open[T+1] 起算），口径一致可融合
- 权重重短端（2d 最近买点 > 6d > 20d）；某侧缺失时按可用权重重归一
- 排序（取前 k）与挂价共用同一分数
- 二模型版 combine_scores（0.4*p20+0.6*p6）保留供旧口径对照/回退
"""
from __future__ import annotations

import pandas as pd


def _blend(pairs: list[tuple[pd.Series, float]]) -> pd.Series:
    """Generic weighted blend of prediction Series on the union index.

    pairs: [(pred_series, weight), ...]; weights should sum to 1.
    Missing-side renormalization: rows where a model has no prediction use the
    remaining models' weights renormalized (fillna(0) trick avoids NaN*0).
    """
    w_sum = sum(w for _, w in pairs)
    if abs(w_sum - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1, got {w_sum}")

    idx = None
    for s, _ in pairs:
        idx = s.index.union(idx) if idx is not None else s.index
    num = None
    tot = None
    for s, w in pairs:
        p = s.reindex(idx).astype(float)
        w_eff = w * p.notna()
        term = p.fillna(0.0) * w_eff
        num = term if num is None else num + term
        tot = w_eff if tot is None else tot + w_eff
    score = num / tot.where(tot > 0)
    score.name = "score"
    return score


def combine_scores3(pred_2d: pd.Series, pred_6d: pd.Series, pred_20d: pd.Series,
                    w2d: float = 0.40, w6: float = 0.35, w20: float = 0.25) -> pd.Series:
    """v8 三模型融合分（均 next_open 锚）。缺失侧按可用权重重归一。"""
    return _blend([(pred_2d, w2d), (pred_6d, w6), (pred_20d, w20)])


def combine_scores(pred_20d: pd.Series, pred_6d: pd.Series,
                   w20: float = 0.4, w6: float = 0.6) -> pd.Series:
    """Blend two model prediction Series into one score (return units).

    Parameters
    ----------
    pred_20d, pred_6d : pd.Series
        Predictions indexed by (date, code). Missing entries allowed.
    w20, w6 : float
        Weights (should sum to 1; enforced).

    Returns
    -------
    pd.Series named "score" on the union index. Rows where neither model
    has a prediction are NaN. Rows with only one model present use that
    model's prediction renormalized (w_i / w_available).
    """
    return _blend([(pred_20d, w20), (pred_6d, w6)])
