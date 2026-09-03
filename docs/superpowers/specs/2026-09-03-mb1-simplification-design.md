# mb1 线简化设计（2026-09-03）

## 用户指令（原话）

> 首先是两层因子表。公式因子和机器学习因子。保留一个全量的公式因子计算函数。机器学习因子先按下不表。保留一个可以按照股票池训练并保存模型权重的训练函数。保留一个7折训练流程，7折训练流程可以得到ICIR和回测收益与夏普。把所有挖矿层全部删掉，后续挖矿以及因子筛选全部走临时脚本

## 头脑风暴裁定记录（2026-09-03，三问）

| 问题 | 裁定 |
|---|---|
| 驱动力（可多选） | 架构面太大 + 方案线包袱 + 工序成本重（**非**合并准备） |
| 边界 | **仅 mainboard_all 线**：只动本分支代码与本池表/列/产物；共享 DB 只碰 factor_values_mainboard_all 等本池资产；微盘生产链（主仓代码+微盘表+cron 四步）零接触 |
| 等价底线 | **不设等价门**：简化优先，事后重训重测即新基线；六原则中"等价性验证"豁免，底线降为"重训可复现 + 管道冒烟" |

## 目标架构

两层因子范式不变：**公式因子**（唯一计算单源）＋ **机器学习因子**（构建器原样按下不表）。常驻入口收敛为：

1. **全量公式因子计算**：`factors.update`（增量 + `--full`，cron 契约路径不变；计算内核 `extra_factors.py` 单源）
2. **按池训练 + 存权重**：`run_lgb --pool`（四模型 L1 + 输出校准，现状即达标）
3. **七折流程**：`fold_cv`——每折训练 + market 回测，汇总报告含**每模型 test IC/ICIR + 组合收益/夏普**（现状只有回测侧，需补 ICIR）
4. 其余承重：`factors.store`（SQL 唯一点铁律）、`factors.integrity`（pull 链依赖）、`_leak_check`（泄漏断言）、`backtest.run_lgb`、`forecast_display/generate_lgb.py`（cron 契约）、`pools/`、`data/` 其余、`config`、`dataset`、`strategies` 其余

**挖矿与因子筛选 = 临时脚本纪律**（一律 tmp/，无正式入口；AGENTS 明文防未来会话走错路）。

## 判定表

### ❌ 删除（4 个 py + 产物）

| 模块/产物 | 理由 | 连带 |
|---|---|---|
| `factors/mining.py`（audit/contribution/batch） | 挖矿层，用户指令 | 盘上未跟踪产物一并清（factor_audit_*/、contribution_report_*、batch 报告；git 史即存档） |
| `factors/select_factors.py`（簇优先筛选器） | 因子筛选走临时脚本 | `_rank_ic_np` + `MIN_STOCKS_PER_DATE` 迁入 `strategies/lgb.py`（与 `rank_ic`/`ic_summary` 同居，作临时脚本复用积木） |
| `factors/baseline_check.py`（评估器回归门禁 + 8 基准 alpha） | 挖矿层 + "无等价门"裁定下失去存在理由 | `strategies/labels.py` 的 `compute_forward_returns`（唯一调用方）同删；冻结参考 json 一并删 |
| `factors/ic_probe.py` + `data/ic_probe_{pool}.json` ×2 | 挖矿族探针（Phase F 双池验证用过，已完结；用户已批） | —— |
| `factors/folds/` 全部 28 份折清单 json | 折清单机制随筛选器退场 | 见语义变化① |

### 🔧 接缝重写

| 模块 | 改什么 |
|---|---|
| `fold_cv.py` | ① 删每折 `select_factors` 子进程与 `--skip-select` 分支；折模式**固定读主清单**；② `leak_checks` 删"折 json train_end == test_start"断言（保留 meta `train_end < test_start`、预测窗包含两断言）；③ `fold_metrics` 补读各折 meta 的 `test_ic`，汇总报告 per_fold 加每模型 IC/ICIR |
| `run_lgb.py` | 折模式强制读 `factors/folds/{fid}/` 清单的逻辑 → 读主清单 `selected_{pool}_{model}.json`（主清单的 `train_start/train_end` 字段在折模式下仅存档，防误读需注释注明） |
| 注释/叙事同步 | `dataset.py` 头注（收编名单提及 mining/select/baseline）、`config.py` baseline 注释、`factors/__init__.py`、`strategies/labels.py` 头注、`AGENTS.md`（挖矿工作流章节 → 临时脚本纪律；折 CV 章节"每折独立筛选"叙事 → 固定清单口径；模块清单/入口清单更新） |

### ✅ 保留（承重，不动）

两层因子范式；`extra_factors` + `update`（= "一个全量的公式因子计算函数"，增量+`--full` 双模式是 cron 契约）；`store`（SQL 唯一点）；`integrity`（pull 链依赖，非挖矿）；**ML 因子构建器 ×4 原样按下不表**（`build_gb_gap1d`/`build_gb_4d_open2d`/`build_gb_30d_turn5d`/`build_nn_gap1d`——后者是 cron 契约）；`run_lgb`（训练函数）；`_leak_check`（泄漏断言，非挖矿）；`backtest/run_lgb`；`forecast_display`（cron）；`pools/`、`data/` 其余、`config`、`dataset`、`strategies` 其余。

**DB 零改动**：判死/候审列（HighVol×2、gb_4d、gb_30d、SizeSpread_20d、nn 稀疏列、CSI_return_252d 等）均为"保留列"终裁在册，本轮一律不 DROP。

## 语义变化（已获用户裁定）

① **折清单机制退场**：七折 = 固定主清单的滚动评估；per-fold 独立重筛（防筛选泄漏设计）退役——今后换清单 = tmp 脚本重筛 + 重跑七折即新基线。
② **baseline 门禁退场**：以后改标签/评估器，回归检查走临时脚本对拍，无正式门禁。

## 六原则校对

1. **按生命周期删**：mining/ic_probe 为一次性验证或已退役工序 ✓；折清单 = 每次折运行的衍生拷贝 ✓
2. **按裁定史删**：baseline_check 依"无等价门"裁定 ✓；折清单机制依判定表裁定 ✓；ML 构建器依"按下不表"裁定保留 ✓；"保留列"终裁列不动 DB ✓
3. **一域一文件**：挖矿域整体消失（非拆散重摆）；训练域 `run_lgb`/`fold_cv` 各一 ✓
4. **单源收敛**：`_rank_ic_np`/`MIN_STOCKS_PER_DATE` 收敛到 `strategies/lgb.py` 唯一处 ✓
5. **契约不动**：cron 四步模块路径（`data.pull`/`factors.update`/`factors.build_nn_gap1d`/`forecast_display/generate_lgb.py`）零触碰 ✓；store SQL 唯一点 ✓
6. **等价性验证**：豁免（用户裁定）；底线 = 重训可复现 + 管道冒烟

## 实施序

0. 基线提交：会话四件未提交改动先行入库（已完成：c340bd0）
1. 删除四 py + 产物 json；`_rank_ic_np` 积木迁移至 `strategies/lgb.py`
2. `fold_cv` / `run_lgb` 接缝改造（折固定读主清单 + 报告补 ICIR + leak_checks 调整）
3. 注释与 AGENTS 叙事同步
4. 冒烟验证：`run_lgb --model all` 重训四模型成功落盘；`fold_cv --folds F7` 单折全链成功（训练+泄漏断言+回测+汇总含 ICIR；会覆写 F7 折产物，属预期）；`_leak_check` 通过；`backtest.run_lgb` 主窗跑通
5. 新基线记录：**完整七折重跑**出简化后基线报告（固定清单口径，防折产物混代际；约 1 小时量级）；重训后的 IC/主窗数字入 AGENTS 横幅（或标注"简化后基线"）；memory 更新

## 风险与防护

- **删后走错路**：AGENTS 明文"挖矿/筛选无正式入口，一律 tmp 临时脚本"；`_rank_ic_np` 积木有明确新家
- **折模式读主清单的误读**：主 json 的 `train_end` 字段与折窗无关——run_lgb 注释注明
- **旧折产物兼容**：`models/{pool}/folds/` 存量 joblib/parquet 与新 fold_cv 的消费关系不变（leak_checks 读 meta/parquet，不读折 json，兼容）
- **cron 契约**：四步模块路径逐一核对存在且可 import
