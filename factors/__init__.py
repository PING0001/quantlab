# -*- coding: utf-8 -*-
"""因子工程包。

- extra_factors: 公式因子主载体（原生 Polars，宽读法含市值/筹码源表）
- update: 因子管道（增量日更 + --full 全量重建，原 compute.py 已并入）
- integrity: 完整性校验（硬失败 exit 1 / 软警告）
- build_gb_gap1d: gb 因子构建（列所有权归脚本；scoped 折同步模式供 fold_cv；nn_gap1d 已于 2026-09-04 整体退役删除）

挖矿/因子筛选无正式入口（2026-09-03 用户裁定）——一律 tmp/ 临时脚本，
积木从 strategies.lgb（_rank_ic_np）与 dataset 复用。
"""
