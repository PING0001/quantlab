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

# 增量维护的指数清单（ts_code, 存储code）：compute.py 的 _load_index_data
# 与 data/sources.py 的 index 源共用此定义
TRACKED_INDICES = [
    ("000985.CSI", "000985"),   # 中证全指
    ("000300.SH",  "000300"),   # 沪深300
    ("399303.SZ",  "399303"),   # 国证2000（微盘基准）
]

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
    # Calendar (2)：DaysToDelivery 距下一交割日（第三个周五）自然日数，当日=0；
    #             DaysToNextTrading 到下一交易日的休市天数（明天开市=0，周五=2，节前=假期长度）
    ["DaysToDelivery", "DaysToNextTrading"]
    +
    # ST status (1)
    ["IsST"]
    +
    # Chip distribution (4)
    ["WinnerRate", "CostPosition", "ChipDispersion", "ChipSkew"]
    +
    # Intraday shape (8, bench 2026-08)
    ["UpperShadow", "LowerShadow", "ClosePos", "OpenPos",
     "ShadowRatio", "RangeEfficiency", "ClosePos_mean_20d", "ClosePos_std_20d"]
    +
    # Alternative versions (old raw formulas)
    ["alpha1_v0", "alpha18_v0", "alpha50_v0", "alpha60_v0"]
)


# ---- Model registry (dual regression bench, spec 2026-08-20 §3.1) ----
# 模型名 = 标签窗口末端交易日（T+16~T+20 -> "20d"，T+4~T+6 -> "6d"）
MODEL_CONFIGS = {
    "20d": dict(
        label_window=(16, 20),
        label_price="open",
        baseline="next_open",     # v7 实验：锚改回 open[T+1]（v3 曾用；v4 曾裁定 close[T]）
        label_buffer=20,          # 最远引用仍为 T+20 开盘，buffer 不变（单变量纯净）
        horizon="label_20d",        # 预测列: pred_label_20d
    ),
    "6d": dict(
        label_window=(4, 6),
        label_price="open",
        baseline="next_open",
        label_buffer=6,
        horizon="label_6d",         # 预测列: pred_label_6d
    ),
}


def get_model_config(model: str = "20d") -> dict:
    if model not in MODEL_CONFIGS:
        raise ValueError(f"unknown model {model!r}, expected one of {sorted(MODEL_CONFIGS)}")
    return MODEL_CONFIGS[model]


# ---- Rolling fold CV（2026-08-21 用户裁定：连续 7 折半年窗，训练起点锁 2020-01，
#      扩张窗口；F7=现测试集作不变性回归检验）----
FOLDS = {
    "F1": ("2022-07-01", "2022-12-31"),
    "F2": ("2023-01-01", "2023-06-30"),
    "F3": ("2023-07-01", "2023-12-31"),
    "F4": ("2024-01-01", "2024-06-30"),
    "F5": ("2024-07-01", "2024-12-31"),
    "F6": ("2025-01-01", "2025-05-31"),
    "F7": ("2025-06-01", "2026-06-01"),
}
FOLD_TRAIN_START = "2020-01-01"   # 所有折训练起点一致（用户约束：不用更老数据）


def get_fold(fid: str) -> tuple[str, str]:
    """折定义 -> (test_start, test_end)，均 'YYYY-MM-DD'。未知折抛错。"""
    if fid not in FOLDS:
        raise ValueError(f"unknown fold {fid!r}, expected one of {sorted(FOLDS)}")
    return FOLDS[fid]


# ---- Model ----
def get_model_dir(name: str = None, fold: str = None) -> Path:
    d = ROOT / "models" / (name or POOL_NAME)
    return d / "folds" / fold if fold else d


def get_lgb_model_path(model: str = "20d", name: str = None, fold: str = None) -> Path:
    return get_model_dir(name, fold) / f"lgb_{model}.joblib"


def get_legacy_lgb_model_path(name: str = None) -> Path:
    """bench 前的单分类模型：仅供 generate_lgb 回退分支与整体回滚使用，
    本 bench 任何代码不得写入该文件。"""
    return get_model_dir(name) / "lgb_multi.joblib"


# ---- Predictions cache ----
def get_lgb_predictions_path(model: str = "20d", name: str = None, fold: str = None) -> Path:
    p = name or POOL_NAME
    if fold:
        return ROOT / "data" / "folds" / fold / f"predictions__{p}_lgb_{model}.parquet"
    return ROOT / "data" / f"predictions__{p}_lgb_{model}.parquet"


def get_lgb_predictions_meta_path(model: str = "20d", name: str = None, fold: str = None) -> Path:
    p = name or POOL_NAME
    if fold:
        return ROOT / "data" / "folds" / fold / f"predictions__{p}_lgb_{model}_meta.json"
    return ROOT / "data" / f"predictions__{p}_lgb_{model}_meta.json"


# ---- Backtest output ----
def get_backtest_dir(name: str = None, fold: str = None) -> Path:
    d = ROOT / "backtest" / (name or POOL_NAME)
    return d / "folds" / fold if fold else d


# ---- Forecast HTML ----
def get_forecast_lgb_dir(name: str = None) -> Path:
    return ROOT / "forecast_display" / "html_lgb" / (name or POOL_NAME)
