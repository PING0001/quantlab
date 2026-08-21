# 双回归模型（20d + 6d 开盘中位数）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把单 LightGBM 三分类改造成两个回归模型（20d/6d 开盘中位数收益），加权综合分（rank/exec 双通道）进每日调仓回测，并新增 8 个日线衍生日内形态因子。

**Architecture:** 参数化统一训练入口（模型注册表驱动标签窗口/基准价/buffer/产物路径）；标签与撮合入场基准对齐（T+1 开盘）；撮合层解耦排序分与执行分；生产侧 generate_lgb 带单模型回退分支。旧分类产物全部保留作回滚。

**Tech Stack:** 既有栈（DuckDB / Polars / pandas / LightGBM / joblib / parquet），零新增第三方依赖。

**Spec:** `docs/superpowers/specs/2026-08-20-dual-regression-models-design.md`（本计划从 spec 出发，执行者须同时读 spec）

## Global Constraints

- **解释器**：一律用 `/Users/cui/.workbuddy-ai/quantlab-env/bin/python`（`.venv` 是 Windows 的，勿用）
- **工作区**：`/Users/cui/Projects/quantlab-dual`（git worktree，分支 `feat/regression-dual-model`，基于 fix HEAD 6fa11fc）；DB 为符号链接共享主工作区 `data/ashare.duckdb`
- **DB 写窗口**：写库前 `ps aux | grep -E 'data\.pull|factors\.update'` 确认无进程；避开工作日 19:30–21:30（workbuddy 流水线 + cyq 重试时段）
- **模型命名**：模型名 = 标签窗口末端日（T+16~T+20 → `20d`；T+4~T+6 → `6d`）；预测列 `pred_label_{model}`；标签列 `label_{model}`
- **调仓**：`REBALANCE_FREQ = 1`（每日调仓，2026-08-21 用户裁定）
- **旧产物不删不改**：`lgb_multi.joblib`、`predictions__{pool}_lgb.parquet`、`selected_{pool}.json`、旧 equity/benchmark CSV、旧 HTML
- **列所有权**：compute/update 拥有全部因子列（含新增 8 列）；`ai_gz2000_*` 归 build_ai_factor——任何写入不得整行替换
- **factor_values.date 保持 VARCHAR**；新列必须永久进 compute 计算路径（禁止表内手工列，DROP TABLE 重建会丢）
- **提交规范**：中文 conventional commits（`feat:`/`fix:`/`docs:`/`chore:`）
- **中间态声明**：任务 2 落地后、任务 8 完成前，本分支的 generate_lgb/backtest 处于不可用中间态——仅限 worktree 内，主工作区（fix 分支）不受影响，workbuddy 照常运行

---

### Task 1: compute_median_open 标签函数 + 独立复算验证

**Files:**
- Modify: `strategies/labels.py`（文件末尾、`compute_nextopen_limit_mask` 之前追加）
- Create: `_check_median_open.py`（项目根，一次性验证脚本，完成后保留备查）

**Interfaces:**
- Produces: `compute_median_open(kline_df, start_day=4, end_day=6, baseline="next_open") -> pd.Series`，(date, code) MultiIndex，值 = `median(open[T+s..T+e]) / baseline_price - 1`；baseline ∈ {"next_open"→open[T+1], "open"→open[T], "close"→close[T]}；无 delist_info 参数（退市语义见下）
- 后续 Task 5（训练）与 Task 7（回测 IC）以 `(16,20)`/`(4,6)` + `baseline="next_open"` 调用

**退市语义（spec §3.2，P0-1 修订）**：部分窗口保留可得开盘价的中位数（真实可成交退出价）；窗口全缺 → NaN（由调用方 notna 过滤）；**不实现 -1.0 填充**（旧填充是死代码）。

- [x] **Step 1: 写验证脚本（先于实现）** `_check_median_open.py`：从 DB read_only 取 3 只样本股全历史 OHLC——正常股（池内行数最多）、退市股（`delist_info` 首条且 kline 有行）、停牌股（池内最大日内 gap）；脚本内含**独立朴素重算**（逐行收集 `opens[t+s..t+e]` 取 median / baseline），对 `compute_median_open` 三种 baseline 全部断言 `np.allclose(equal_nan=True)`；断言：退市股尾部行 NaN、全库无 -1.0 值、停牌股 T+k 跨复牌日时窗口用复牌后开盘价（打印人工目检）
- [x] **Step 2: 跑脚本确认失败**：`cd /Users/cui/Projects/quantlab-dual && /Users/cui/.workbuddy-ai/quantlab-env/bin/python _check_median_open.py` → 期望 `ImportError: cannot import name 'compute_median_open'`
- [x] **Step 3: 实现**（追加到 labels.py，模式照抄 `compute_median_close`，groupby("code").apply 内 `o.shift(-d)` concat + median，baseline 列选择；非法 baseline 抛 ValueError；docstring 写明退市语义）
- [x] **Step 4: 跑脚本通过**：期望输出三案例全部 `MISMATCH=0`、`n_neg1=0`、退市股尾部 NaN 行数与窗口几何一致
- [x] **Step 5: Commit**：`git add strategies/labels.py _check_median_open.py && git commit -m "feat: compute_median_open 开盘中位数收益标签（next_open 基准 + 实证退市语义）"`

### Task 2: config 模型注册表 + 路径参数化

**Files:**
- Modify: `config.py:124-141`（Model/Predictions 段）

**Interfaces:**
- Produces: `MODEL_CONFIGS: dict`（"20d"/"6d" → label_window/label_price/baseline/label_buffer/horizon，内容 = spec §3.1 代码块逐字）；`get_lgb_model_path(model="20d", name=None) -> models/{pool}/lgb_{model}.joblib`；`get_lgb_predictions_path(model="20d", name=None)`；`get_lgb_predictions_meta_path(model="20d", name=None)`；`get_legacy_lgb_model_path(name=None) -> models/{pool}/lgb_multi.joblib`（Task 8 回退分支用）
- 验证：`python -c` 断言四个路径函数对 "20d"/"6d"/legacy 的返回值字符串；`MODEL_CONFIGS` 两项的 label_buffer 为 20/6

- [x] **Step 1: 修改 config.py**（新增 MODEL_CONFIGS + 四个路径函数；旧无参调用默认 model="20d" 指向新路径——本分支内旧消费方在 Task 7/8 一并迁移）
- [x] **Step 2: 验证**：`python -c "from config import *; ..."` 打印并断言全部路径
- [x] **Step 3: Commit**：`feat: config 模型注册表 + 双模型产物路径参数化`

### Task 3: 8 个日内形态因子 + 表迁移 + 全历史回填

**Files:**
- Modify: `factors/extra_factors.py`（`compute_non_alpha_factors` 内追加 8 列 polars 表达式）
- Modify: `config.py`（`SELECTED_FACTORS` 注册 8 个新名）
- Create: `factors/migrate_intraday_shape.py`（一次性工具：ALTER TABLE ADD COLUMN ×8 → 全历史 compute_panel → 仅新列 UPDATE 写入 factor_values）

**因子定义（spec §3.8 逐字）**：UpperShadow、LowerShadow、ClosePos（high==low→NULL）、OpenPos、ShadowRatio（分母0→NULL）、RangeEfficiency（Intraday_range_pct / 当日换手率 amount/circ_mv/10）、ClosePos_mean_20d、ClosePos_std_20d。实现前先读 `extra_factors.py` 与 `compute.py` 的 `compute_panel`/`store_factor_values`，确认输入列（前复权 OHLC 在 panel 中的列名、turnover 列名）与既有 Gap_pct 的写法模式。

**顺序约束（spec §3.8）**：ALTER 之后、回填完成前**不得跑 `factors.update`**；回填时确认无 pull/update 进程；回填只 UPDATE 8 个新列 + 缺行 INSERT 由 write_panel 列所有权逻辑决定——不得触碰 ai_gz2000_*。

- [ ] **Step 1: 读 extra_factors.py / compute.py 相关段**，确定表达式落点与列名
- [ ] **Step 2: 实现 8 列表达式 + SELECTED_FACTORS 注册**
- [ ] **Step 3: 写 migrate_intraday_shape.py**（幂等：列已存在则跳过 ALTER；--dry-run 打印计划）
- [ ] **Step 4: 验证表达式正确性**：小样本（3 只股 1 年）跑 compute_panel，SQL 抽 3 个 (code,date) 手算公式比对
- [ ] **Step 5: 全历史回填**（大计算；确认无并发后跑；预期 ~1100 股 × 全历史）
- [ ] **Step 6: 回填后验证**：8 列非空率（应接近现有因子列水平）、ai_gz2000_* 两列逐字节不变（回填前后各 SELECT checksum）、`python -m factors.update --dry-run` 无新列告警
- [ ] **Step 7: Commit**（分两次：表达式+工具 / 回填为数据操作不入库）

### Task 4: select_factors --model 参数化

**Files:**
- Modify: `factors/select_factors.py`

**Interfaces:**
- Produces: CLI `python -m factors.select_factors --model 20d|6d`；标签 = `compute_median_open(**MODEL_CONFIGS[model] 的窗口与 baseline)`；输出 `factors/selected_{pool}_{model}.json`；删除 `(labels == -1.0)` 死条件（:126）；json meta 记录 train_start 口径（2015 vs run_lgb 2020 差异如实记录）；IC 循环缩为单标签

- [ ] **Step 1: 参数化改造**（argparse；PRIMARY_HORIZON/ic_20d 命名改为按 model 命名）
- [ ] **Step 2: 跑 20d 与 6d 各一遍**，产出两个 selected json；肉眼检查 6d 清单与 20d 差异合理（动量/反转类权重应不同）
- [ ] **Step 3: Commit**：`feat: select_factors 按模型参数化（open 标签驱动 IC）`

### Task 5: run_lgb.py 参数化 + 训练两模型

**Files:**
- Modify: `run_lgb.py`

**Interfaces:**
- Consumes: Task 1 `compute_median_open`、Task 2 `MODEL_CONFIGS`/路径函数、Task 4 `selected_{pool}_{model}.json`
- Produces: `python run_lgb.py --model 20d|6d|all`；`model_type="regressor"`；删除 `_classify`/sign 准确率；评估 = rank IC/IR/hit（`rank_ic`/`ic_summary`）+ MAE + 十分位单调性；meta json 含 label_window/baseline/buffer；预测列 `pred_label_{model}`

- [ ] **Step 1: 改造**（标签/评估/meta 全部由注册表驱动；LGB_KWARGS 沿用但 model_type 改 regressor，样本权重分支自然失效）
- [ ] **Step 2: 训练 20d**：期望测试期 rank IC 显著为正、IR/hit 打印入 meta（首跑即基线）；异常塌方则停（spec 验收 3）
- [ ] **Step 3: 训练 6d**：同上
- [ ] **Step 4: Commit**：`feat: run_lgb 双回归模型统一入口（--model 20d|6d|all）`

### Task 6: strategies/combine.py 综合分双通道

**Files:**
- Create: `strategies/combine.py`；Create: `_check_combine.py`

**Interfaces:**
- Produces: `combine_scores(pred_20d: pd.Series, pred_6d: pd.Series, w20=0.6, w6=0.4) -> pd.DataFrame`，列 `["rank_score","exec_score"]`；rank_score = 逐日横截面 percentile 归一加权（单模型缺失按可用权重重归一，NaN 不参与 percentile）；exec_score = 原始值加权（同重归一规则）。Task 7/8 消费
- 验证：构造含 NaN/单边缺失/平局值的合成 Series，断言值域、重归一权重和为 1、平局不炸

- [ ] **Step 1: 实现 + 合成数据断言脚本**；**Step 2: 跑通**；**Step 3: Commit**：`feat: combine_scores 双通道综合分（rank/exec）`

### Task 7: signals 解耦 + 回测改造（每日调仓）

**Files:**
- Modify: `backtest/signals.py`（`run_portfolio_rebalance` 与 `run_long_short` 增加可选 `rank_scores=None`，默认回退 `predictions`；排序用 rank_scores，`_buy_limit/_sell_limit` 一律用 `predictions`——exec 量纲，P0-2）
- Modify: `backtest/run_lgb.py`（加载两份 parquet → combine → 双 Series 传入；`REBALANCE_FREQ=1`；IC 块对两标签 + rank_score；输出 `equity_lgb_combined_daily_rebalance.csv` + `benchmark_combined.csv`；`PRED_COL` 逻辑重写）
- Modify: `backtest/test_holding.py`（读 `pred_5d` 的坏代码：改为读 `pred_label_6d` 并更新路径调用；若改造代价过高则在文件头加 DEPRECATED 注释并跳过——实现时定夺）
- 读 `backtest/signals.py` 全文后再动笔（本任务前必读）

**验证（spec 验收 5）**：全流程跑通；抽查成交记录卖单限价 ∈ 前收 ±10% 带内（断言无 ≥1.5× 永不成交单）；换手率打印（每日调仓 sanity）；旧 `equity_lgb_20d_5d_rebalance.csv`/`benchmark.csv` 未被覆写（mtime/内容比对）。

- [ ] Step 1-4: 改造 → 回测跑通 → 断言脚本/抽查 → Commit `feat: 双模型综合分每日调仓回测（rank/exec 解耦）`

### Task 8: generate_lgb 双模型 + 生产回退 + integrity 参数化

**Files:**
- Modify: `forecast_display/generate_lgb.py`（双模型加载；缺一 → 单模型 + HTML 顶部 DOWNGRADED 标注；双缺 → `get_legacy_lgb_model_path()` 旧分类模型兜底；删 NEUTRAL/旧 WEIGHTS；文件名 `{date}_forecast_lgb_dual.html`；表格含 pred_label_20d/pred_label_6d/rank_score）
- Modify: `factors/integrity.py`（selected 清单读取参数化：存在 per-model 清单时用 20d 版，否则回退旧 `selected_{pool}.json`）
- 实现前必读两个文件全文

**验证（spec 验收 6）**：本地演练三种状态——双模型在、删 6d joblib、删双 joblib（演练后恢复）；HTML 渲染含三列分数。

- [ ] Step 1-4: 改造 → 三态演练 → Commit `feat: generate_lgb 双模型报告 + 生产回退分支`

### Task 9: _leak_check 参数化 + 个股级泄漏断言

**Files:**
- Modify: `_leak_check.py`（标签窗口参数化 --model；新增个股行移位断言：测试期边界前 30 交易日内每个训练行，其窗口末端行的日期 < TEST_START——对池内全部股票）

- [ ] Step 1-3: 改造 → 两模型跑通（停牌跨界案例单独打印）→ Commit `feat: _leak_check 按模型参数化 + 个股级泄漏断言`

### Task 10: 全链路验收 + AGENTS.md 同步

- [ ] 按 spec §5 验收 1-8 逐项执行并记录结果（生产演练 = 主工作区三步命令顺序本地跑——**在主工作区 fix 分支跑**，不动 bench）
- [ ] AGENTS.md 同步：双回归架构、155→163 因子、退市标签语义纠正、每日调仓、模型产物路径、worktree 开发模式
- [ ] Commit：`docs: AGENTS.md 同步双回归模型架构`

---

## Self-Review 记录

- Spec 覆盖：§3.1→T2、§3.2→T1、§3.3→T5、§3.4→T4、§3.5→T6、§3.6→T7、§3.7→T8、§3.8→T3、§5→T10、§6 声明→各任务约束、§7 顺序→任务序、§8→T8 回退分支 + 全局约束中间态声明。无遗漏。
- 类型一致性：`compute_median_open` 签名在 T1 定义、T4/T5/T7 消费一致；`combine_scores` 返回列名 rank_score/exec_score 在 T6 定义、T7/T8 消费一致；路径函数签名 T2 定义、T5/T7/T8 消费一致。
- 占位符：T3/T7/T8 含"实现前必读文件"步骤（extra_factors/compute/signals/generate_lgb/integrity 未在计划撰写时全文读过——按项目八荣八耻"以瞎猜接口为耻"，行为规格 + 锚点 + 必读步骤优于编造代码）；无 TBD/TODO。
