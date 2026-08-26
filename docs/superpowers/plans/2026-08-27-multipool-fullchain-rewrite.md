# 全链多池化重写方案（2026-08-27 定稿，待执行）

> 本文档是自包含交接件：新 session 读完此文件 + AGENTS.md 即可开工，无需原会话上下文。
> 制定过程：对 502ccdc 的全接缝审查（P0/P1/P2 发现）→ 用户裁定「重写而非修复」→ 范围扩展至
> 数据层+因子两层+训练链+回测层 → 按 main 分支破坏性简化六原则校对定稿。

## 0. 现状（开工前先自行核验，状态可能已移动）

- **主仓** `/Users/cui/Projects/quantlab`（main 分支）：
  - `2fec9ee` 破坏性简化落地（py 62→28，验证全绿：重训 parquet md5 一致/泄漏断言 56/56/baseline 零漂移）
  - `410017c` 回测产物退出版本库（旧三件 CSV 移除，输出目录 gitignore）
  - 设计文档：`docs/superpowers/specs/2026-08-25-destructive-simplification-design.md`（六原则出处）
- **本 worktree** `/Users/cui/Projects/quantlab-bench-mainboard-all`（分支 `bench-mainboard-all`）：
  - `cc6884c` 主仓工作区快照（**已冗余**，内容被 2fec9ee 正式取代）
  - `502ccdc` mainboard_all 数据层（本方案的重写对象）
- **DB**：本 worktree 无本地 DB。用 `QUANTLAB_DB=/Users/cui/Projects/quantlab/data/ashare.duckdb`
  共享主仓实体；解释器一律 `.venv/bin/python`。
- **已落地数据资产（勿重建、勿 DROP，全部经只读实证）**：
  - `pool_snapshots_mainboard_all`：23 档，最新档 2026-06-01 共 3147 只，877 微盘 ⊆ 最新档（差集空）
  - `factor_values_mainboard_all`：5,714,554 行 / 79 列 / 2019-01-02..2026-08-25
  - rank 参考系隔离：2,314,358 重叠行中 2,314,346 行 `Return_20d_rank` 不同
  - `data/ic_probe_{mainboard_all,mainboard_microcap}.json`：两池零信号方向翻转，主力因子族
    方向保留幅度衰减（微盘 −0.152 → 全主板 −0.116 @AvgAmount_3d）

## 1. 审查发现（重写的动机，均已实证）

| 级别 | 问题 | 位置（502ccdc 行号，rebase 后以 grep 为准） |
|---|---|---|
| P0 | SQL 漏 f 前缀，`{fv_table}` 字面量触发 ParserException 被宽 except 吞掉 → 软警告 6（股票级覆盖缺口）静默失效 | `factors/integrity.py:225`（except 在 :251，log.debug 吞掉） |
| P1 | 读侧硬编码 `factor_values` 13+ 处：非微盘 env 下静默错宇宙 | run_lgb.py:98；select_factors.py:56；mining.py:85,447；baseline_check.py:229；backtest/run_lgb.py:481,644；generate_lgb.py:109,166,568；_leak_check.py:248,254,268 |
| P1' | 写侧硬编码（比读侧更糟）：gb/nn 会把模型因子列写进微盘表 | build_gb_gap1d.py:88,156,163,166；build_nn_gap1d.py:86,101,210,245,248,305,306,313,316 |
| P2 | 人读 JSON 锚定 DB 路径 → bench 产物写进主仓 `pools/` | membership.py `_history_path`（DB_PATH.parent.parent） |
| P3 | 6 个输出不带池名跨池互覆写（含 baseline 冻结参考） | fold_cv_report.json / fold_cv_run.log / factor_audit_report.json / factor_contribution_report.json（含 folds 变体）/ baseline_reference.json / integrity_report.json |
| P3 | `HISTORY_SINCE="2019-12-02"` 是微盘首档日期却当全局常量，被跨文件 import | run_lgb.py:42 → select_factors.py:39 |
| P3 | pull.py 靠匹配 integrity 中文 reason 里的 "factor_values" 字样分流退出码 | data/pull.py:316 |
| P3 | integrity 内 pool 参数与全局 POOL_NAME 混用 | integrity.py:196,223 |
| 数据 | 微盘生产表陈旧值：daily_kline 被重述后因子行不重算，对账不可见 | ~13.3k 行 / ~350 只代码（2020+，Return_20d/Volatility 口径实测） |
| 死列 | 14 个零引用因子列 | GZ2000_{return_1d,vol_10d,vol_60d,reversal_60d,pricepos_252d,atr_14d,boll_width}、shibor_on/1m、alpha1/18/50/60_v0、mf_volchg3。**保留**：GZ2000_return_5d（gb/nn 活输入）、GZ2000_return_20d（integrity 引用） |

## 2. 已裁定决策（勿重新讨论）

- **池身份一等公民化**：`pools/spec.py` 的 `PoolSpec`（frozen dataclass：name/snap_table/factor_table/band/data_since）
  是唯一池注册表，吸收 membership.POOLS + config.FACTOR_TABLES 两处注册表。
- **因子表 SQL 单点**：`factors/store.py` 持有全部因子表 SQL（读写/对账/列操作）。铁律：因子表 SQL
  只准出现在 store.py。
- **训练装配单点**：`dataset.py` 收编五个文件复制的装载+过滤链（含 far_cross 纯函数）。
- **PRED_COLS/W2D/W6D/W20D 迁 config**（消 report←backtest 反向 import）。
- **CLI 显式 `--pool`**（默认 env `QUANTLAB_POOL`，默认微盘；cron 无参=微盘契约不变）。
- **人读 JSON 命名**：`{pool}_snapshots.json`（沿用 main 的改名裁定），ROOT 锚定。
- **保护清单（语义绝不碰）**：backtest 模拟器开盘市价语义；strategies/labels.py（C4 依赖源码可
  inspect）；strategies/lgb.py 的 LGBStrategy 类形态（旧折 joblib 兼容）；extra_factors.py 计算内核
  （345 行纯 Polars）；gb/nn 的 OOF 管道/列所有权/冻结推理；cron 四步模块路径
  （data.pull / factors.update / factors.build_nn_gap1d / forecast_display/generate_lgb.py）。
- **不做**：目录大重组（28 文件结构刚定型）；七折全链终裁（只到 F4 冒烟）；任何 tushare 拉取
  （全程预期零网络）；两张池表/快照重建。

## 3. 重写判定表（哪些重写/哪些保留）

| 层 | 模块 | 判定 |
|---|---|---|
| 数据 | pools/membership.py | 小改（spec 接入 + ROOT 锚定 + 改名） |
| 数据 | factors/update.py | 重写（结构）：编排/IO 分离，丢 table= 硬编码缺省 |
| 数据 | factors/integrity.py | 重写：store 化 + 异常显性化（check_errors）+ 参数统一 |
| 数据 | data/pull.py / sources.py | 不动 / 微调 |
| 因子L1 | factors/extra_factors.py | 内核保留；清 14 死列（store.drop_column） |
| 因子L2 | build_gb/nn_gap1d.py | 接缝重写：store 列操作 + spec 路径 + CLI --pool |
| 训练 | run_lgb.py | 重写（分层）：装配下沉 dataset.py，入口瘦身 |
| 训练 | strategies/labels.py、strategies/lgb.py | **保留** |
| 训练 | factors/select_factors.py | 接缝重写（算法不动） |
| 训练 | fold_cv.py | 接缝重写：子进程显式 --pool + 产物池命名 |
| 训练 | _leak_check.py | 重构：过滤链纯函数共享，IO 级独立保留，56 项语义不变 |
| 回测 | backtest/run_lgb.py | 接缝重写：IsST 走 store、reset_points 显式池、--pool；**模拟器不动** |
| 报告 | forecast_display/generate_lgb.py | 接缝重写：spec/store/懒解析目录 |
| 挖矿/门禁 | mining.py / baseline_check.py | 接缝重写：IO+产物池化（baseline_reference_{pool}.json） |
| 配置 | config.py | 重写（收缩）：池概念整体迁出 |

## 4. 六原则对照（main 简化哲学 → 本方案）

1. **按生命周期删**：14 死列清理（反向依赖已查：仅 return_5d/20d 有引用）；MIN_STOCKS_PER_DATE
   死常量实现或删。
2. **按裁定史删**：rebase 吸收 registry.py/旧 CSV/改名；config 池概念整体迁出后删 FACTOR_TABLES/
   get_factor_table/路径 helper。
3. **一域一文件**：spec（池）/ store（因子表 SQL）/ dataset（训练装配）三新模块各对应一个域。
4. **单源收敛**：两注册表→一；13+ SQL→一；5 份装配→一；PRED_COLS→config；HISTORY_SINCE→spec.data_since；
   `python -m pools.spec` 自描述 CLI。
5. **契约不动**：保护清单 + cron 无参=微盘 + 微盘路径逐字节不变。
6. **等价性验证收尾**：沿用 2fec9ee 口径（见 §6）。

## 5. 实施阶段（每阶段 1-2 commit，微盘全链保持绿）

**Phase 0 — rebase 到新 main**
`git rebase --onto 410017c cc6884c bench-mainboard-all`（把 502ccdc 摘到新 main 上，丢冗余快照）。
预期冲突：factors/update.py（CRLF→LF，取 bench LF 版）、AGENTS.md（取 main 文案订正）。
rebase 自动吸收四项 neat-freak 修正（registry 删除/_snapshots.json 改名/AGENTS 数字 62→28、56 项/
macro_daily 表）。

**Phase A — 数据层地基**
- `pools/spec.py`：PoolSpec + POOLS + get_pool；`__main__` 打印各池表名/带宽/行数/最新档（自描述）。
  依赖方向 spec→config（只取 ROOT/POOL_NAME），无环。
- `factors/store.py`：从 update.py 迁 rebuild_table/upsert_panel/latest_date/missing_dates/
  stock_coverage/get_lookback_start（必传 spec）；新增 load_panel/load_isst/columns/max_date_where/
  ensure_column/update_column/drop_column。
- update.py 编排化 + `--pool`；`--dry-run` 对拍目标日期/回补清单与迁移前一致。
- integrity 重写：run_checks(con, spec) 全走 store；逐检查异常→`report["check_errors"]` + log.warning
  （禁 debug 吞）；报告 `data/integrity_report_{pool}.json`（先 grep 确认无程序化消费方，有则微盘保旧名）；
  reason 文案保持兼容（pull.py:316 字符串匹配暂不动，`failed_side` 结构化字段加法式共存）。

**Phase B — 因子两层**
- 死列清理：14 列 DROP（两张因子表都清；列清单写入 commit message 备查）。
- gb/nn 接缝重写：表 SQL 走 store、nn STATE_PATH 走 spec.model_dir()、补 `--pool`（gb 现无任何 CLI）。
  cron 的 `build_nn_gap1d --infer-only` 无参=微盘不变。

**Phase C — 训练链**
- `dataset.py`：装配（store 装载 + labels + IsST/limit/delist/industry + member_mask + far_cross 纯函数）。
  run_lgb/select_factors/mining/_leak_check/baseline_check 全部改用。
- run_lgb 瘦身为「装配→训练→校准→落盘」；select_factors 接缝迁移。
- _leak_check：C3 复刻改为「IO 独立构建 + 共享纯函数过滤」（保留断言价值，消手动同步）。
- fold_cv：`--pool`；子进程链显式传 `--pool`（不再靠 env 继承）；报告/日志带池名。

**Phase D — 回测层 + 报告**
- backtest：IsST 两处走 store、reset_points 显式池、`--pool`；PRED_COLS/W 迁 config（backtest 与
  forecast 同源 import）；**模拟器语义与回测口径零改动**。
- forecast_display：spec/store、HTML_DIR 从模块级求值改为入口解析。

**Phase E — 收尾**
- config 收缩定稿（池概念清零）。
- AGENTS.md 多池章节重写：spec/store/dataset 三单源、池化边界（哪些池感知）、worktree 约定
  （QUANTLAB_DB/QUANTLAB_POOL）、store 铁律。
- `store.staleness_audit(con, spec, n_dates=20)`：分层抽样历史日期 compute_panel 纯 CPU 重算 vs 存量
  diff，输出漂移清单；微盘应复现 ~13.3k 行/350 只（自校验）。修复动作仅报告不执行。

**Phase F — 终验**
1. 微盘等价门：`run_lgb.py` 重训四模型 parquet 数值（md5 或逐列 allclose）与基线一致；`_leak_check.py`
   56/56；`factors.baseline_check` 零漂移；`backtest.run_lgb` 对横幅（+48.72%/2.47/−6.70% vs 基准
   +26.17%，超额 +22.55pp）；`fold_cv --skip-train` 抽折复跑一致。
2. integrity 双池各跑：check_errors 为空、stock_gaps 检查真实执行（P0 修复的可观测证据）。
3. ic_probe 双池重跑与现存 JSON 数值一致。
4. staleness_audit 复现已知发现。
5. **F4 冒烟**：`QUANTLAB_DB=... QUANTLAB_POOL=mainboard_all python fold_cv.py --folds F4` 端到端
   （泄漏断言过；记录 vs 微盘 F4 −22.8%/基准 −26.7% 对照；gb/nn 新池列未建则先探明再定）。

## 6. 硬约束与纪律

- **零 tushare 拉取**（速率限制+耗时；本方案全程不需要网络）。
- **DB 单写者锁**：写库前 `ps aux | grep -E 'data\.pull|factors\.update'`；避开工作日 21:05 前后
  （cron 窗口）；membership 在写连接进程内查询必须传 con。
- **勿提交 DuckDB/.env**；勿 `--full` 全量重建。
- cron 四步模块路径是契约；改名必须同步主仓 cron automation。
- 每阶段结束跑该阶段验证门，绿了才 commit；commit 粒度=阶段内可回退单元。
- 主仓（main）不动——全部工作在本 worktree，验证后另行裁定合并。

## 7. 行号速查（基于 502ccdc，rebase 后漂移以 grep 为准）

- 硬编码读：run_lgb:98 / select_factors:56 / mining:85,447 / baseline_check:229 / backtest:481,644 /
  generate_lgb:109,166,568 / _leak_check:248,254,268
- 硬编码写：build_gb_gap1d:88,156,163,166 / build_nn_gap1d:86,101,210,245,248,305,306,313,316
- P0 bug：integrity:225（f 前缀缺失）/ except:251
- 混用参数：integrity:196,223；config 路径 helper 的 name= 参数（:200-232）
- HISTORY_SINCE：run_lgb:42；selected json 内联拼接：run_lgb:171-178 / select_factors:344-350 /
  fold_cv:59 / mining:93
- pull 字符串匹配：data/pull.py:316
