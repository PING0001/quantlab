"""
Central configuration for quantlab.
All pool-specific paths are derived from POOL_NAME.

Set QUANTLAB_POOL env var to switch between stock pools:
    QUANTLAB_POOL=mainboard_all python run_lgb.py
"""
from __future__ import annotations

import os
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POOL_NAME = os.environ.get("QUANTLAB_POOL", "mainboard_microcap")

log = logging.getLogger("config")

# ---- Database ----
# QUANTLAB_DB 可覆盖 DB 路径（默认本仓 data/ashare.duckdb）
DB_PATH = Path(os.environ.get("QUANTLAB_DB") or (ROOT / "data" / "ashare.duckdb"))

# ---- Stock pool ----
# 池注册表单源（2026-08-27）：pools/spec.py 的 POOLS（PoolSpec：快照表/
# 因子表/带宽/data_since/路径族）。本文件不含任何池表名或池路径。
# 池代码查询走 pools/membership，因子表 SQL 走 factors/store。
# 旧 pools/*.json 历史并集已物理删除，config 不再提供任何 json 池读取。


# ---- Selected factor set (single source of truth) ----

# 增量维护的指数清单（ts_code, 存储code）：factors/update.py 的 _load_index_data
# 与 data/sources.py 的 index 源共用此定义
TRACKED_INDICES = [
    ("000985.CSI", "000985"),   # 中证全指
    ("000300.SH",  "000300"),   # 沪深300
    ("399303.SZ",  "399303"),   # 国证2000（微盘基准）
]

# alpha101 全系已于 2026-08-22 移除（DSL 引擎与 vnpy 参考一并删除），
# 8 个经典表达式随挖矿层删除（2026-09-03，git 史可查），不再有基准实现。
SELECTED_FACTORS = (
    # Momentum (3)
    ["Return_5d", "Return_20d", "Reversal_60d"]
    +
    # Volatility (4)
    # 2026-08-22 审计修复：ATR/MACD_signal/SMA 为 qfq 绝对水平（含 latest_adj
    # 未来信息），换比值形式 ATR_pct / MACD_hist_pct / CloseBIAS_20d
    ["ATR_pct", "Volatility", "Volatility_60d", "Bollinger_width"]
    +
    # Price position / technical (6)
    ["Price_position_252d", "Stochastic_K", "Return_skew_20d",
     "Trend_strength", "CloseBIAS_20d", "MACD_hist_pct"]
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
    # Short-window variants (6, bench 2026-08-22)：原有因子的 3/5 日短窗形式，
    # 供短 horizon 模型（open2d 等）独立筛选取用
    ["Return_3d", "Volatility_3d", "Amihud_3d", "AvgAmount_3d",
     "ClosePos_mean_3d", "Price_position_5d"]
    +
    # LLM 挖矿第一批幸存因子 (6, 2026-08-22)：300 假设库首测 16 取 7 后
    # 又删 LogClose（qfq 水平因子带 latest_adj 未来信息且与 SMA 冗余 0.93），
    # 实证见 git 史（factors/mining.py，2026-09-03 随挖矿层删除）；
    # 涨停次数为 |ret|>=9.5% 近似口径
    ["LimitUpCnt_20d", "PostHighDrawdown_10d", "MIN_5d",
     "IntradaySkew_60d", "VolPriceCorr_20d", "OvernightMean_20d"]
    +
    # LLM 挖矿第二批幸存因子 (7, bench 2026-08-22 深夜)：批次二 16 测 7 幸存，
    # 全池 max 相关 <0.75；StockIndexCorr 需池等权日收益（compute 内横截面广播）
    ["StockIndexCorr_20d", "AmountShrink_5_60", "FirstVolumeSpike_5d",
     "AmountConc_20d", "OpenPos_mean_20d", "LimitUpStreakMax_60d", "DownsideVol_20d"]
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
    # 2026-09-03 用户裁定：gap1d 降位——跳空预测归 ML 因子层（gb_gap1d），
    # 主模型仅三（open2d/6d/20d）
    # 独立实验模型（2026-08-22 用户要求，暂不接入）：预测 T+2 开盘
    # open[T+2]/open[T+1]−1（= median_open 窗口(2,2)+next_open 锚，与 6d/20d
    # 同锚族；曾用 close 锚 open[T+2]/close[T]−1，IC 0.10~0.15），因子复用 6d 清单
    "open2d": dict(
        label_window=(2, 2),
        label_price="open",
        baseline="next_open",
        label_buffer=2,
        horizon="label_open2d",     # 预测列: pred_label_open2d
    ),
}


# ---- Blend (v8 single source) ----
# 融合列名与权重的唯一出处（2026-08-27 自 backtest/run_lgb.py 迁入）：
# backtest 与 forecast_display 同源 import，防两处漂移暗改有效权重
# （v8 的核心教训——幅度/权重漂移曾在主窗口摆动 ±20pp）。
# 权重是用户手工配比（2026-08-22 裁定 0.40/0.35/0.25；2026-08-28 用户裁定改为
# 0.30/0.40/0.30——6d 加重、open2d 减负），勿改；调优留到实盘前。
PRED_COLS = {"open2d": "pred_label_open2d", "6d": "pred_label_6d", "20d": "pred_label_20d"}
W2D, W6D, W20D = 0.30, 0.40, 0.30  # score = 0.3*p2d + 0.4*p6d + 0.3*p20d


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
# 池路径族（2026-08-27 池概念清零）：models/{pool}/、data/predictions__{pool}_*、
# backtest/{pool}/、forecast html_lgb/{pool}/ 一律经 pools.spec.PoolSpec 方法
# （model_dir/lgb_model_path/lgb_predictions_path/...，ROOT 锚定）。本文件
# 不再含任何池路径。
