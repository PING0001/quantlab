"""
Central configuration for quantlab.
All pool-specific paths are derived from POOL_NAME.

Set QUANTLAB_POOL env var to switch between stock pools:
    set QUANTLAB_POOL=mainboard_smallcap && python run_mlp_multi.py
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


# ---- Model ----
def get_model_dir(name: str = None) -> Path:
    return ROOT / "models" / (name or POOL_NAME)


def get_model_path(name: str = None) -> Path:
    return get_model_dir(name) / "mlp_multihead.pt"


def get_lgb_model_path(name: str = None) -> Path:
    return get_model_dir(name) / "lgb_multi.joblib"


# ---- Predictions cache ----
def get_predictions_path(name: str = None) -> Path:
    p = name or POOL_NAME
    return ROOT / "data" / f"predictions__{p}.parquet"


def get_predictions_meta_path(name: str = None) -> Path:
    p = name or POOL_NAME
    return ROOT / "data" / f"predictions__{p}_meta.json"


# ---- Backtest output ----
def get_backtest_dir(name: str = None) -> Path:
    return ROOT / "backtest" / (name or POOL_NAME)


# ---- Forecast HTML ----
def get_forecast_dir(name: str = None) -> Path:
    return ROOT / "forecast_display" / "html" / (name or POOL_NAME)


def get_forecast_lgb_dir(name: str = None) -> Path:
    return ROOT / "forecast_display" / "html_lgb" / (name or POOL_NAME)
