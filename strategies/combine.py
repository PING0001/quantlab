# -*- coding: utf-8 -*-
"""
Combined score for the dual-regression models (2026-08-21 用户口径 v4 + 权重回改).

单一分数，无百分位层：

    score = 0.6 * pred_20d + 0.4 * pred_6d

- 预测锚 = T 日收盘价（label = median(open[T+s..T+e]) / close[T] - 1），
  与挂单公式同锚：买入限价 = 收盘×(1+score−3%)，卖出目标价 = 收盘×(1+score)
- 排序（取前 k）与挂价共用同一分数
- 权重重 20d（2026-08-21 用户二次裁定改回，原 v4 曾为 0.4/0.6 重 6d）；某侧缺失时按可用权重重归一
- 弃用的 rank_score/exec_score 双通道与 percentile 层见 git 历史（v3）
"""
from __future__ import annotations

import pandas as pd


def combine_scores(pred_20d: pd.Series, pred_6d: pd.Series,
                   w20: float = 0.6, w6: float = 0.4) -> pd.Series:
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
    if abs(w20 + w6 - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1, got w20={w20}, w6={w6}")

    idx = pred_20d.index.union(pred_6d.index)
    p20 = pred_20d.reindex(idx).astype(float)
    p6 = pred_6d.reindex(idx).astype(float)

    # 缺失侧重归一（先 fillna(0) 再乘：NaN*0=NaN 会污染可用侧行）
    w20_eff = w20 * p20.notna()
    w6_eff = w6 * p6.notna()
    tot = w20_eff + w6_eff

    score = (p20.fillna(0.0) * w20_eff + p6.fillna(0.0) * w6_eff) / tot.where(tot > 0)
    score.name = "score"
    return score
