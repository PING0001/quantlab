# -*- coding: utf-8 -*-
"""
Factor computation module for quantlab.

Provides:
- Alpha101 factor definitions (101 WorldQuant alpha factors)
- Non-alpha factors (momentum, volatility, chip, market state, etc.)
- Full and incremental computation pipelines
"""

from .alpha101 import ALPHA_EXPRESSIONS, IND_NEUTRALIZE_ALPHAS
from .compute import compute_panel, store_factor_values
from .utility import calculate_by_expression
from .ops import DataProxy, register_functions
