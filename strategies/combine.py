# -*- coding: utf-8 -*-
"""
Combined score for the dual-regression models (spec §3.5, plan Task 6).

Two channels, both required by the backtest split (P0-2):
- rank_score: per-date cross-sectional percentile in [0,1], weighted sum.
  For ranking / HTML display / IC evaluation. Rank-ensemble is lossless for
  the backtest because run_portfolio_rebalance only consumes ordering; it
  also neutralizes the horizon scale mismatch (20d returns disperse ~2x
  more than 6d). All-tied days collapse to 0.5 (no fabricated dispersion).
- exec_score: raw-prediction weighted sum (expected-return magnitude).
  For the simulator's limit-price formulas (_buy_limit/_sell_limit), which
  treat the score as an expected return — a percentile there would make
  sell limits unfillable.

When only one model has a prediction for a (date, code), renormalize with
the available weight — required, not a fallback: the 6d label's shorter NaN
tail gives its predictions ~14 extra test-period dates over the 20d model.
"""
from __future__ import annotations

import pandas as pd


def percentile_per_date(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile in (0, 1) per date; NaN stays NaN.

    Project convention (extra_factors._pct_rank): (rank - 0.5) / n with
    average-rank ties — an all-tied date maps every row to 0.5 (neutral),
    a single-stock date also maps to 0.5. Public entry point so the
    backtest can build interpretable threshold channels (P90/P95 entry).
    """
    g = s.groupby(level="date")
    return (g.rank() - 0.5) / g.transform("count")


_percentile_per_date = percentile_per_date


def combine_scores(pred_20d: pd.Series, pred_6d: pd.Series,
                   w20: float = 0.6, w6: float = 0.4) -> pd.DataFrame:
    """Combine two model prediction Series into rank/exec score channels.

    Parameters
    ----------
    pred_20d, pred_6d : pd.Series
        Predictions indexed by (date, code). Missing entries allowed.
    w20, w6 : float
        Combination weights (should sum to 1; enforced).

    Returns
    -------
    pd.DataFrame with columns ["rank_score", "exec_score"], union index.
    Rows where neither model has a prediction are NaN.
    """
    if abs(w20 + w6 - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1, got w20={w20}, w6={w6}")

    idx = pred_20d.index.union(pred_6d.index)
    p20 = pred_20d.reindex(idx).astype(float)
    p6 = pred_6d.reindex(idx).astype(float)

    # per-row renormalized weights: only available models count
    # （先 fillna(0) 再乘：NaN*0=NaN 会污染重归一行）
    w20_eff = w20 * p20.notna()
    w6_eff = w6 * p6.notna()
    tot = w20_eff + w6_eff

    r20 = _percentile_per_date(p20)
    r6 = _percentile_per_date(p6)

    rank_score = (r20.fillna(0.0) * w20_eff + r6.fillna(0.0) * w6_eff) / tot.where(tot > 0)
    exec_score = (p20.fillna(0.0) * w20_eff + p6.fillna(0.0) * w6_eff) / tot.where(tot > 0)

    return pd.DataFrame({"rank_score": rank_score, "exec_score": exec_score})
