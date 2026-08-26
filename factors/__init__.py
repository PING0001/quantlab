# -*- coding: utf-8 -*-
"""因子工程包。

- extra_factors: 公式因子主载体（原生 Polars，宽读法含市值/筹码源表）
- update: 因子管道（增量日更 + --full 全量重建，原 compute.py 已并入）
- integrity: 完整性校验（硬失败 exit 1 / 软警告）
- select_factors: 筛选（簇优先，每模型一份清单）
- mining: 挖矿三件套（audit / contribution / batch）
- baseline_check: 评估器回归门禁（含 8 个基准 alpha 实现）
- build_gb_gap1d / build_nn_gap1d: gb/nn 因子构建（列所有权归各脚本）
"""
