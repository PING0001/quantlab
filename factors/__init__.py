# -*- coding: utf-8 -*-
"""
Factor computation module for quantlab.

Provides:
- Non-alpha factors (momentum, volatility, chip, market state, etc.；
  alpha101 全系已于 2026-08-22 移除)
- Evaluator regression baselines (baseline_alphas / baseline_check)
- Full and incremental computation pipelines
"""

from .compute import compute_panel, store_factor_values
