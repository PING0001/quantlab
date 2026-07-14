"""
Central configuration for quantlab.
All pool-specific paths are derived from POOL_NAME.

Set QUANTLAB_POOL env var to switch between stock pools:
    set QUANTLAB_POOL=smallcap_on_mainboard && python run_lgb.py
"""
from __future__ import annotations

import os
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POOL_NAME = os.environ.get("QUANTLAB_POOL", "mainboard_microcap")

log = logging.getLogger("config")

# ---- Database ----
DB_PATH = ROOT / "data" / "ashare.duckdb"

# ---- Stock pool definitions ----
POOLS_DIR = ROOT / "pools"


def get_pool_path(name: str = None) -> Path:
    return POOLS_DIR / f"{name or POOL_NAME}.json"


def load_stock_pool(name: str = None) -> tuple[str, list[dict]]:
    """Load a pool JSON. Returns (block_name, stocks_list)."""
    path = get_pool_path(name)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stocks = data["stocks"]
    log.info("Stock pool %s: %d stocks", path.stem, len(stocks))
    return data.get("block_name", ""), stocks


def load_all_pool_stocks() -> list[dict]:
    """Load union of stock dicts across ALL pool JSONs, deduplicated by code."""
    seen = {}
    if not POOLS_DIR.exists():
        return []
    for pool_file in sorted(POOLS_DIR.glob("*.json")):
        with open(pool_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for s in data.get("stocks", []):
            code = s["code"]
            if code not in seen:
                seen[code] = s
    stocks = list(seen.values())
    log.info("All pools union: %d unique stocks", len(stocks))
    return stocks


def get_pool_codes(name: str = None) -> list[str]:
    """Get sorted list of stock codes for a specific pool."""
    _, stocks = load_stock_pool(name)
    return sorted(s["code"] for s in stocks)


# ---- Selected factor set (single source of truth) ----
SELECTED_FACTORS = (
    # Alpha101 (101 factors, from vnpy / WorldQuant formulaic alphas)
    [f"alpha{i}" for i in range(1, 102)]
    +
    # Momentum (3)
    ["Return_5d", "Return_20d", "Reversal_60d"]
    +
    # Volatility (4)
    ["ATR", "Volatility", "Volatility_60d", "Bollinger_width"]
    +
    # Price position / technical (6)
    ["Price_position_252d", "Stochastic_K", "Return_skew_20d",
     "Trend_strength", "SMA", "MACD_signal"]
    +
    # Intraday pattern (3) -> (2) after removing Body_pct (dup of Intraday_return)
    ["Gap_pct", "Intraday_range_pct"]
    +
    # Volume / liquidity (2)
    ["Volume_ratio", "Amihud_illiquidity"]
    +
    # Market cap / amount (3)
    ["AvgAmount_90d", "LnMktCap", "LnFloatCap"]
    +
    # Turnover (2)
    ["Turnover_3d", "Turnover_3d_ratio"]
    +
    # Intraday (1)
    ["Intraday_return"]
    +
    # Market state (5)
    ["CSI_return_1d", "CSI_return_20d", "CSI_volatility_20d",
     "HS300_return_1d", "HS300_return_20d"]
    +
    # Cross-sectional ranks (3)
    ["Return_1d_rank", "Return_20d_rank", "Turnover_3d_rank"]
    +
    # Firm age (1)
    ["LnAge"]
    +
    # ST status (1)
    ["IsST"]
    +
    # Chip distribution (4)
    ["WinnerRate", "CostPosition", "ChipDispersion", "ChipSkew"]
    +
    # Alternative versions (old raw formulas)
    ["alpha1_v0", "alpha18_v0", "alpha50_v0", "alpha60_v0"]
)


# ---- Model ----
def get_model_dir(name: str = None) -> Path:
    return ROOT / "models" / (name or POOL_NAME)


def get_lgb_model_path(name: str = None) -> Path:
    return get_model_dir(name) / "lgb_multi.joblib"


# ---- Predictions cache ----
def get_lgb_predictions_path(name: str = None) -> Path:
    p = name or POOL_NAME
    return ROOT / "data" / f"predictions__{p}_lgb.parquet"


def get_lgb_predictions_meta_path(name: str = None) -> Path:
    p = name or POOL_NAME
    return ROOT / "data" / f"predictions__{p}_lgb_meta.json"


# ---- Backtest output ----
def get_backtest_dir(name: str = None) -> Path:
    return ROOT / "backtest" / (name or POOL_NAME)


# ---- Forecast HTML ----def get_forecast_lgb_dir(name: str = None) -> Path:
    return ROOT / "forecast_display" / "html_lgb" / (name or POOL_NAME)
