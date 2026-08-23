# -*- coding: utf-8 -*-
"""因子注册表与血缘校验（模型因子范式基建，2026-08-23）。

范式（用户裁定）：
  第0层数据表 → 第1层公式因子（确定性公式，宽读法：市值/筹码等源表均可）
  → 第2层模型因子（LightGBM walk-forward OOF，只吃第0/1层）
  → 第3层主模型 → 第4层combiner（手工权重）
约束：DAG 无环（构造性保证：模型因子输入只允许公式因子与 OHLCV，永不引用
任何模型输出）；OHLCV 一律后复权（hfq 水平时点诚实且永不重绘）；删除公式
因子前必须查反向依赖；模型因子无结构特权（不是门控、不进模型结构）。
分层不混同（2026-08-23 用户裁定）：模型因子与公式因子是不同认识论来源
（透明结构假设 vs 历史拟合产物，失效模式与维护责任不同）——模型因子不参与
公式因子的簇优先竞争，不会也不应"替换某公式簇代表"；其准入走独立门：对整
个在任集合（两层合并）的边际贡献 A/B。select_factors 的簇优先只在公式层内
有意义。
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
#
# 2026-08-23 用户裁定：删除首批全部两个模型因子（mf_vol20 / mf_volsurp5，
# DB 列已归档 data/archive/mf_columns_20260823.parquet 后 DROP），由用户
# 重新指导编写。机制（白名单/血缘/构建管道）保留待用。
MODEL_FACTORS: dict[str, dict] = {}


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
    """审查工具用：返回因子的层级与完整依赖名单。

    模型因子列出全部输入（公式因子 + OHLCV 白名单特征），报告自解释，
    无需回读注册表源码；公式因子恒返回 "公式"（第1层无下游依赖声明）。
    """
    if factor_name.startswith(MODEL_FACTOR_PREFIX):
        e = MODEL_FACTORS.get(factor_name)
        if e is None:
            return "模型(未注册)"
        fs = e["inputs"]["factors"]
        oh = e["inputs"]["ohlcv"]
        return (f"模型(公式因子{len(fs)}: {', '.join(fs)}; "
                f"OHLCV{len(oh)}: {', '.join(oh)})")
    return "公式"
