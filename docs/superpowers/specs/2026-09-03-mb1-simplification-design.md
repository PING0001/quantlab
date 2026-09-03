# mb1 线简化设计（2026-09-03）

## 用户指令（原话）

> 首先是两层因子表。公式因子和机器学习因子。保留一个全量的公式因子计算函数。机器学习因子先按下不表。保留一个可以按照股票池训练并保存模型权重的训练函数。保留一个7折训练流程，7折训练流程可以得到ICIR和回测收益与夏普。把所有挖矿层全部删掉，后续挖矿以及因子筛选全部走临时脚本

**补充指令（同日）**：

> ML因子随7折叠训练测试流程同步训练，注意不要让ML因子泄露未来信息。7折叠流程结束后不要清空ML因子权重，让它自然保留以供未来进行实盘预测使用。对于所有模型，只需要保存一个权重文件即可，不需要留存档。折叠测试时直接覆盖即可

## 头脑风暴裁定记录（2026-09-03，三问）

| 问题 | 裁定 |
|---|---|
| 驱动力（可多选） | 架构面太大 + 方案线包袱 + 工序成本重（**非**合并准备） |
| 边界 | **仅 mainboard_all 线**：只动本分支代码与本池表/列/产物；共享 DB 只碰 factor_values_mainboard_all 等本池资产；微盘生产链（主仓代码+微盘表+cron 四步）零接触 |
| 等价底线 | **不设等价门**：简化优先，事后重训重测即新基线；六原则中"等价性验证"豁免，底线降为"重训可复现 + 管道冒烟" |

## 目标架构

两层因子范式不变：**公式因子**（唯一计算单源）＋ **机器学习因子**（**随折同步训练**，见下）。常驻入口收敛为：

1. **全量公式因子计算**：`factors.update`（增量 + `--full`，cron 契约路径不变；计算内核 `extra_factors.py` 单源）
2. **按池训练 + 存权重**：`run_lgb --pool`（**三主模型 open2d/6d/20d** L1 + 输出校准；gap1d 降位见语义变化⑤）
3. **七折流程**：`fold_cv`——每折 **① ML 因子同步训练（防泄漏）→ ② 三主模型训练 → ③ 泄漏断言 → ④ market 回测**，汇总报告含**每模型 test IC/ICIR + 组合收益/夏普**（现状只有回测侧，需补 ICIR）
4. 其余承重：`factors.store`（SQL 唯一点铁律）、`factors.integrity`（pull 链依赖）、`_leak_check`（泄漏断言）、`backtest.run_lgb`、`forecast_display/generate_lgb.py`（cron 契约）、`pools/`、`data/` 其余、`config`、`dataset`、`strategies` 其余

### ML 因子随折同步训练（防泄漏规则）

- **训练对象**：当前清单实际引用的 ML 因子列（gb_/nn_ 前缀自动发现；现役 = gb_gap1d）——清单换人自动跟随，无需注册表。
- **防泄漏铁规则**：折 F 的 ML 因子训练截止 = **折 test_start − ML 标签 buffer**（标签最远引用不越 test_start）。折训练行的特征值 = 截止前 walk-forward cadence 的 **OOS 值**（逐段模型只见过更早数据，与全局 OOF 同构——早段值折间不变、幂等）；折测试窗的特征值 = **截止时单一终态模型统一推理**（测试窗内不再重训——否则 D2 日的特征会携带 (test_start, D2) 间的标签信息 = 相对 D2 的未来泄漏）。
- **权重生命周期**：每个 ML 因子单一权重文件（`models/{pool}/{ml_factor}.joblib`），逐折覆盖；**流程结束后不清空，自然保留供实盘推理**（终态 = F7 折口径训练的权重）。
- **写表**：折流程经 `store.update_columns` 把本折 ML 特征值写回因子表列（列所有权仍归构建器；F7 test_end 之后的近沿日期仍由全局构建值兜底，实盘前可手动刷新一次）。

### 权重单文件纪律（全模型）

- 三主模型：`models/{pool}/lgb_{m}.joblib` 各一份，**折训练直接覆盖主路径**；`models/{pool}/folds/` 折模型目录机制整体删除。
- ML 因子：`models/{pool}/{ml_factor}.joblib` 各一份，同上。
- 不留任何权重存档/快照；折评估证据 = 逐折预测 parquet + meta（data/folds/，保留）+ 汇总报告。

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
| `models/mainboard_all/folds/` 折模型目录（7 折 × 4 = 28 joblib） | 权重单文件纪律：折训练直接覆盖主权重路径，不留档 | 见语义变化③；`pools/spec.py` 的 `lgb_model_path(fold=)` 分支删除（预测/meta 路径的 fold 分支保留）。**微盘 `models/mainboard_microcap/folds/` 属微盘线资产，边界外不动** |
| **gap1d 主模型全家桶**：`config.MODEL_CONFIGS["gap1d"]` 条目、`selected_{pool}_gap1d.json`、`lgb_gap1d.joblib`、gap1d parquet/meta、折清单 gap1d 份 | **用户裁定（2026-09-03）：gap1d 不算主模型——跳空预测职责归 ML 因子层（gb_gap1d 特征，脚本内自带目标定义，已核实不依赖 MODEL_CONFIGS）**；主模型层遗留实验位无消费方 | 见语义变化⑤；`_leak_check` 断言数 56→42（3 模型×14） |
| `data/folds/F*/predictions__mainboard_all_*`（旧折预测/meta 缓存，gitignored 盘上文件） | **用户裁定（2026-09-03）：旧折不用兼容**——混合代际历史产物，新基线由简化后重跑产生 | 同目录微盘折缓存属微盘线，边界外不动；`data/fold_cv_report_mainboard_all.json` 由新基线重跑覆写 |

### 🔧 接缝重写

| 模块 | 改什么 |
|---|---|
| `fold_cv.py` | ① 删每折 `select_factors` 子进程与 `--skip-select` 分支；折模式**固定读主清单**；② 每折新增 **ML 因子同步训练步骤**（调构建器 scoped 模式：截止 = test_start − ML 标签 buffer；OOS 训练值 + 终态模型推理测试窗；写表 + 单权重覆盖）；③ `leak_checks` 删"折 json train_end == test_start"断言（保留 meta `train_end < test_start`、预测窗包含两断言）；④ `fold_metrics` 补读各折 meta 的 `test_ic`，汇总报告 per_fold 加每模型 IC/ICIR |
| `run_lgb.py` | 折模式强制读 `factors/folds/{fid}/` 清单的逻辑 → 读主清单 `selected_{pool}_{model}.json`（主清单的 `train_start/train_end` 字段在折模式下仅存档，防误读需注释注明）；折模式模型落盘走**主权重路径**（单文件覆盖） |
| `factors/build_gb_gap1d.py` | 加 scoped 调用接缝（供 fold_cv 子进程调）：`--cutoff`（训练截止，含 ML 标签 buffer 回退）+ 终态模型落 `models/{pool}/gb_gap1d.joblib` + 折测试窗由终态模型推理写列；既有全量 OOF 行为不变 |
| 注释/叙事同步 | `dataset.py` 头注（收编名单提及 mining/select/baseline）、`config.py` baseline 注释、`factors/__init__.py`、`strategies/labels.py` 头注、`AGENTS.md`（挖矿工作流章节 → 临时脚本纪律；折 CV 章节"每折独立筛选"叙事 → 固定清单口径；模块清单/入口清单更新） |

### ✅ 保留（承重，不动）

两层因子范式；`extra_factors` + `update`（= "一个全量的公式因子计算函数"，增量+`--full` 双模式是 cron 契约）；`store`（SQL 唯一点）；`integrity`（pull 链依赖，非挖矿）；`build_gb_gap1d`（在册 ML 因子构建器，接缝化 scoped 模式，见接缝重写）；**候审构建器 ×3 原样不动**（`build_gb_4d_open2d`/`build_gb_30d_turn5d`/`build_nn_gap1d`——nn 是 cron 契约）；`run_lgb`（训练函数）；`_leak_check`（泄漏断言，非挖矿）；`backtest/run_lgb`；`forecast_display`（cron）；`pools/`、`data/` 其余、`config`、`dataset`、`strategies` 其余。

**DB 零改动**：判死/候审列（HighVol×2、gb_4d、gb_30d、SizeSpread_20d、nn 稀疏列、CSI_return_252d 等）均为"保留列"终裁在册，本轮一律不 DROP。

## 语义变化（已获用户裁定）

① **折清单机制退场**：七折 = 固定主清单的滚动评估；per-fold 独立重筛（防筛选泄漏设计）退役——今后换清单 = tmp 脚本重筛 + 重跑七折即新基线。
② **baseline 门禁退场**：以后改标签/评估器，回归检查走临时脚本对拍，无正式门禁。
③ **权重单文件不留档**：折训练直接覆盖主权重（跑完七折后主权重 = F7 折口径，恰为主窗口口径，兼作实盘权重）；折评估证据只剩预测 parquet/meta + 汇总报告。
④ **ML 因子折同步**：ML 因子从"手动全局构建的冻结列"改为"折流程内同步训练"（防泄漏规则见目标架构节）；折跑完保留的 ML 权重（F7 截止 2025-05 训练）供实盘推理。
⑤ **gap1d 降位**（用户裁定 2026-09-03："那个是 ML 因子"）：主模型收敛为 open2d/6d/20d 三者；跳空预测只在 ML 因子层存在（gb_gap1d）。合并回 main 时主仓训练层随之三模型化——cron 四步与报告（只吃三 parquet 融合）不受影响。

## 六原则校对

1. **按生命周期删**：mining/ic_probe 为一次性验证或已退役工序 ✓；折清单 = 每次折运行的衍生拷贝 ✓
2. **按裁定史删**：baseline_check 依"无等价门"裁定 ✓；折清单机制依判定表裁定 ✓；ML 构建器依"按下不表"裁定保留 ✓；"保留列"终裁列不动 DB ✓
3. **一域一文件**：挖矿域整体消失（非拆散重摆）；训练域 `run_lgb`/`fold_cv` 各一 ✓
4. **单源收敛**：`_rank_ic_np`/`MIN_STOCKS_PER_DATE` 收敛到 `strategies/lgb.py` 唯一处 ✓
5. **契约不动**：cron 四步模块路径（`data.pull`/`factors.update`/`factors.build_nn_gap1d`/`forecast_display/generate_lgb.py`）零触碰 ✓；store SQL 唯一点 ✓
6. **等价性验证**：豁免（用户裁定）；底线 = 重训可复现 + 管道冒烟

## 实施序

0. 基线提交：会话四件未提交改动先行入库（已完成：c340bd0）
1. 删除四 py + 产物 json + 折模型目录；`_rank_ic_np` 积木迁移至 `strategies/lgb.py`
2. `build_gb_gap1d` scoped 接缝（--cutoff / 终态权重落盘 / 测试窗终态推理）
3. `fold_cv` / `run_lgb` 接缝改造（折固定读主清单 + ML 同步训练步 + 权重写主路径 + 报告补 ICIR + leak_checks 调整）
4. 注释与 AGENTS 叙事同步
5. 冒烟验证：`run_lgb --model all` 重训三模型成功落盘；`fold_cv --folds F7` 单折全链成功（ML 同步训练→三模型→泄漏断言→回测→汇总含 ICIR；覆写主权重属预期）；`_leak_check` 通过；`backtest.run_lgb` 主窗跑通
6. 新基线记录：**完整七折重跑**出简化后基线报告（固定清单 + ML 折同步口径；每折约 9-12 分钟）；重训后的 IC/主窗数字入 AGENTS 横幅（或标注"简化后基线"）；memory 更新

## 风险与防护

- **ML 泄漏**：折 scoped 训练截止 = test_start − ML 标签 buffer；测试窗内不重训（终态模型推理）；leak_checks 增补一条——折 meta 的 ML 因子列值时间戳覆盖检查（若实现成本高，退为文档纪律 + scoped 逻辑单测）
- **删后走错路**：AGENTS 明文"挖矿/筛选无正式入口，一律 tmp 临时脚本"；`_rank_ic_np` 积木有明确新家
- **折模式读主清单的误读**：主 json 的 `train_end` 字段与折窗无关——run_lgb 注释注明
- **权重覆盖的中间态**：折跑中途主权重 = 中间折口径（bench 分支无 cron/LIVE 消费方，风险仅限人为中途取用——AGENTS 注明"折跑后权重为 F7 态"）
- **cron 契约**：四步模块路径逐一核对存在且可 import
