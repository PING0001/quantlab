# -*- coding: utf-8 -*-
"""因子注册表与血缘校验（模型因子范式基建，2026-08-23）。

范式（用户裁定）：
  第0层数据表 → 第1层公式因子（确定性公式，宽读法：市值/筹码等源表均可）
  → 第2层模型因子（LightGBM walk-forward OOF，只吃第0/1层）
  → 第3层主模型 → 第4层combiner（手工权重）
约束：DAG 无环（构造性保证：模型因子输入只允许公式因子与 OHLCV，永不引用
任何模型输出）；OHLCV 一律后复权（hfq 水平时点诚实且永不重绘）；删除公式
因子前必须查反向依赖；模型因子无结构特权（不是门控、不进模型结构）。
"""
from __future__ import annotations

# 模型因子列名前缀（factor_values 中该前缀列所有权归 build_model_factors.py）
MODEL_FACTOR_PREFIX = "mf_"

# 白名单允许的两类输入
#   ('factor', 公式因子列名)  —— 必须存在于 factor_values 且不带 mf_ 前缀
#   ('ohlcv', 特征名)          —— 仅限 OHLCV_HFQ_FEATURES（后复权派生，比值形态）
OHLCV_HFQ_FEATURES = [
    "ret_1d", "ret_5d", "ret_20d",          # hfq 收益
    "gap_1d",                                # 隔夜跳空
    "range_pct",                             # 当日振幅
    "range_mean_20d",                        # 20 日均振幅
    "volume_ratio_20d",                      # 量/20日均量
    "amount_ratio_20d",                      # 额/20日均额
]

# ---- 模型因子注册表 ----
# 每个条目：inputs（白名单声明）、target_key（build_model_factors 的目标构造器）、
# label_buffer（目标最远引用的交易日数）、cadence（重训间隔交易日）、params 指纹
MODEL_FACTORS: dict[str, dict] = {
    "mf_vol20": {
        "desc": "预测未来 20 日已实现波动（hfq 日收益 std，T+1..T+20）",
        "target_key": "realized_vol_20d",
        "label_buffer": 25,
        "cadence": 120,
        "inputs": {
            "factors": [
                "Return_5d", "Return_20d", "Reversal_60d",
                "Volatility", "Volatility_60d", "ATR_pct", "Bollinger_width",
                "Intraday_range_pct", "Turnover_3d", "AvgAmount_90d",
                "LnMktCap", "Price_position_252d", "GZ2000_vol_10d",
            ],
            "ohlcv": ["ret_1d", "ret_5d", "ret_20d", "range_pct",
                      "range_mean_20d", "volume_ratio_20d", "amount_ratio_20d"],
        },
        "params": dict(num_leaves=31, learning_rate=0.06, n_estimators=200,
                       min_child_samples=200, objective="regression_l1",
                       random_state=42),
    },
    "mf_volsurp5": {
        "desc": "预测未来 5 日量 surprise（T+1..T+5 总额 / 5×近20日均额 − 1）",
        "target_key": "volume_surprise_5d",
        "label_buffer": 8,
        "cadence": 120,
        "inputs": {
            "factors": [
                "Turnover_3d", "Turnover_3d_ratio", "Volume_ratio",
                "AvgAmount_3d", "AvgAmount_90d", "LnFloatCap",
                "Return_5d", "Volatility", "GZ2000_return_5d",
            ],
            "ohlcv": ["volume_ratio_20d", "amount_ratio_20d", "ret_1d",
                      "ret_5d", "range_pct"],
        },
        "params": dict(num_leaves=31, learning_rate=0.06, n_estimators=200,
                       min_child_samples=200, objective="regression_l1",
                       random_state=42),
    },
}


def validate_entry(name: str, entry: dict, factor_columns: set[str]) -> list[str]:
    """校验注册表条目，返回错误列表（空=通过）。

    factor_columns: factor_values 现有列（运行时传入，防声明引用不存在的列）。
    """
    errs = []
    for f in entry["inputs"]["factors"]:
        if f.startswith(MODEL_FACTOR_PREFIX):
            errs.append(f"{name}: 输入 {f} 是模型因子列——禁止（无环约束）")
        elif factor_columns and f not in factor_columns:
            errs.append(f"{name}: 公式因子 {f} 不在 factor_values")
    for o in entry["inputs"]["ohlcv"]:
        if o not in OHLCV_HFQ_FEATURES:
            errs.append(f"{name}: ohlcv 特征 {o} 不在白名单 {OHLCV_HFQ_FEATURES}")
    if not name.startswith(MODEL_FACTOR_PREFIX):
        errs.append(f"{name}: 模型因子列名必须以 {MODEL_FACTOR_PREFIX} 开头")
    return errs


def reverse_dependencies(factor_name: str) -> list[str]:
    """删除/退休公式因子前调用：返回声明依赖它的模型因子列表。"""
    return [mf for mf, e in MODEL_FACTORS.items()
            if factor_name in e["inputs"]["factors"]]


def lineage(factor_name: str) -> str:
    """审查工具用：返回因子的层级描述。"""
    if factor_name.startswith(MODEL_FACTOR_PREFIX):
        e = MODEL_FACTORS.get(factor_name)
        if e is None:
            return "模型(未注册)"
        return f"模型(依赖{len(e['inputs']['factors'])}公式因子+{len(e['inputs']['ohlcv'])}OHLCV)"
    return "公式"
