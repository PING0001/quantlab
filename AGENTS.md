# Quantlab — AI Agent Instructions

> **⚠️ 本工作树是双回归模型体系主线（worktree `/Users/cui/Projects/quantlab-dual`，2026-08-21 开工于 dev 分支，2026-08-24 经用户裁定合并入 `main` 并切换——此后在本工作树的 `main` 上工作；dev 分支指针保留于同提交）**
> 本文件主体已于 2026-08-23（Task 10）重写为双回归体系；主工作区 `/Users/cui/Projects/quantlab`（fix 分支，workbuddy 生产流水线所在，含另一 session 未提交改动）的三分类架构文档见主区 AGENTS.md，本工作树不碰。
>
> **本分支现役架构（2026-08-23，v8 正式版）速览**：
> - **三模型 LightGBM 回归（objective=regression_l1，条件中位数），全部 next_open 锚**：open2d / 6d / 20d；训练起点锁 2020-01，固定测试集 walk-forward
> - **定稿清单（2026-08-23 用户裁定，折存活投票）：20d 广谱 32 因子（剔低票 Return_1d_rank 3/7、UpperShadow 3/7；Volatility 保留）/ 6d 极简 4（AvgAmount_3d+StockIndexCorr_20d+LnMktCap+CSI 门控）/ open2d 极简 3**——6d/open2d 是"少而精"体质、20d 是"广谱"体质（清单尺寸实验实证）；无候补名单制度，后续只经六关考核换人（批测→画像→贡献→边际贡献门→泛化缺口→折存活投票）。三 IC 0.0942/0.0890/0.0621（34 清单时代实测），主窗口 +30.19%/1.64/−10.26（单窗口证据；**模型尚未按 32 清单重训**，引用主窗口数字注意清单不同步；v8 版官方折验证经用户裁定跳过，清洗版/44 因子版折见 Key Decisions #6）
> - **输出校准（根治幅度事故）**：训练窗留出尾段 60 交易日测 L1 收缩斜率与横截面中位数，输出×斜率还原为"模型真实相信的到期涨幅"；卖出零点平移 `score < Σwᵢ×calib_medianᵢ`（排序与 parquet 原始值不变）。教训：原始幅度加权融合对训练幅度漂移敏感（纯缩放 open2d 预测即可摆动主窗口 12.6pp）
> - **融合分** `score = 0.4×p2d + 0.35×p6d + 0.25×p20d`（手工权重，用户按幅度特性亲自配比；权重调优留到实盘前最后做）
> - **执行语义：开盘市价**（买=次日开盘必成交取前 k；卖=score 低于平移零点时开盘市价卖出）；`--exec limit` 仅供旧语义对照
> - **因子分层范式**：第1层**公式因子**（确定性公式，宽读法含市值/筹码源表）→ 第2层**模型因子**（`factors/build_model_factors.py` walk-forward OOF，只吃公式因子+后复权 OHLCV，`mf_` 前缀列，`factors/registry.py` 白名单+血缘+反向依赖）；禁环、OHLCV 一律后复权（永不重绘）、模型因子无结构特权。**分层不混同**：模型因子不参与公式因子的簇竞争，准入走独立门=对整个在任集合的边际贡献 A/B（与单一公式的相关度只是背景信息）。首批 mf_vol20/mf_volsurp5 已删（2026-08-23 用户裁定重新指导编写；DB 列归档 data/archive/ 后 DROP，注册表已清空，机制保留）。**新一代（2026-08-23 深夜，独立脚本不走注册表）**：`gb_gap1d`（XGBoost 预测 T+1 隔夜跳空，11 公式因子输入，OOS rank IC 0.198，盘上候审）；`mf_volchg3`（XGBoost 波动率变化率，rank IC 0.128，用户裁定暂放）。**命名惯例**：前缀区分模型家族——gb_=梯度提升、nn_=神经网络（预留）、mf_=旧前缀遗留；任务后缀（gap1d 等）跨模型保持一致可比。ai_gz2000_* 已审查删除（in-sample 泄漏）
> - **审查三件套**：`factor_audit.py`（画像：四标签 IC/衰减/全池 max 相关/血缘）、`factor_contribution.py`（gain+日内截面 permutation）、`test_new_factors.py`（候选批测）。纪律：加因子看 train-test 泛化缺口；冗余判定对全池取 max（<0.75 增量/>0.95 冗余，用户标准）；强因子替换弱因子优先于堆加
> - **验证框架**：`python fold_cv.py`（7 折半年窗，每折独立筛选+训练+双语义回测，含泄漏断言）；修复后干净折：market 平均 +23.4% vs 基准 +9.4%、6/7 折超额为正、最差折 −17.5%
> - **已判死（勿再提出）**：close 锚 open2d；limit 执行语义；qfq 水平因子（latest_adj 未来信息）；显式门控逻辑（门控类非线性关系由模型自学，2026-08-23 裁定）
> - **纪律**：F1-F6 做开发、F7/新数据终裁；收益判定看折超额/alpha（主窗口 ≈ 0.53×池 beta + 年化 ~13% alpha）
> - 裁定史与实验全记录：agent memory（dual-regression-bench-status）；spec/plan 见 `docs/superpowers/`

## Project Overview

Quantlab 是一个 **A股主板微盘股量化选股系统**（流通市值 1-20 亿，池 ~1112 只）。本 worktree 是**双回归模型 bench**：三个 LightGBM 回归模型分别预测 open2d / 6d / 20d 前向开盘收益（next_open 锚，条件中位数），手工权重融合成单一 score 驱动每日开盘市价组合。核心流程：

```
Tushare 数据 → DuckDB 存储
  → 因子分层：公式因子（Polars，compute/update）+ 模型因子（mf_*，OOF 管道）
  → 因子筛选（簇优先，相关度>IC，每模型一份清单）
  → 三回归模型训练（run_lgb.py，L1 目标 + 输出校准）
  → 融合分 → 回测（backtest/run_lgb.py）→ HTML 预测报告（三级降级）
  → 泄漏断言（_leak_check.py）+ 折 CV（fold_cv.py）
```

**与主工作区的关系**：主区（`/Users/cui/Projects/quantlab`，fix 分支）是三分类生产流水线（workbuddy 每晚跑）；本 worktree 做双回归主线，重大重构先进 dual 再合并。`data/ashare.duckdb` 与 `.env` 是指向主区的**符号链接**（DB 共享）——主区 compute/update 不产的列（mf_*、挖矿因子等）由本分支写入，两区**共享 DuckDB 单写者锁**，勿同时跑写库任务。

## Technology Stack

| 层 | 技术 |
|------|-----------|
| 数据源 | Tushare（通过 quicksync.cn 中继） |
| 数据库 | **DuckDB**（嵌入式 OLAP，所有数据单一来源，与主区共享） |
| 数值计算 | numpy, pandas, scipy, polars |
| 机器学习 | **LightGBM ×3**（LGBMRegressor，objective=regression_l1） |
| 序列化 | joblib（模型）、parquet（预测）、JSON（meta/清单） |
| 配置 | python-dotenv（.env）、config.py（中心配置） |

## Project Structure

```
quantlab-dual/
├── config.py                # ★ 中心配置：DB 路径、MODEL_CONFIGS（四模型标签窗/锚/buffer）、FOLDS、输出路径
├── run_lgb.py               # ★ 训练入口：三回归模型（+gap1d 实验），L1 目标 + 输出校准（calib_slope/median 入 meta）
├── fold_cv.py               # 滚动 7 折 CV 驱动：每折独立筛选+训练+双语义回测+泄漏断言
├── _leak_check.py           # ★ 主窗口泄漏断言（C1 训练掩码/C2 校准尾段/C3 样本排除/C4 标签方向，52 项）
├── _check_pkgs.py           # 依赖检查
│
├── factors/                 # 因子工程（分层范式，五铁律见 registry.py 头注释）
│   ├── registry.py          # ★ 模型因子注册表：白名单输入/血缘/反向依赖/超参指纹；范式约束单源
│   ├── extra_factors.py     # 公式因子主载体（原生 Polars）；新公式因子加这里
│   ├── compute.py           # 全量计算流水线：DuckDB → Polars → factor_values
│   ├── update.py            # 增量因子更新（日期+股票级对账，lookback 锚点=最晚目标日）
│   ├── integrity.py         # 完整性校验（硬失败 exit 1 / 软警告）
│   ├── select_factors.py    # ★ 筛选：簇优先（average-linkage，相关度>IC），每模型 selected_{pool}_{model}.json
│   ├── baseline_alphas.py   # 8 个经典 alpha 原生 Polars 重实现（评估器回归基准，不入模）
│   ├── baseline_check.py    # ★ 门禁：固定窗 rank IC vs 冻结参考值（改评估器/标签必跑）
│   ├── build_model_factors.py # 模型因子 OOF 管道（mf_ 前缀列所有权，walk-forward + label_buffer）
│   ├── factor_audit.py      # 审查 A：因子画像（四标签 IC/ICIR/衰减/全池 max 相关）
│   ├── factor_contribution.py # 审查 B1：gain + 测试窗日内截面 permutation ΔIC
│   ├── test_new_factors.py  # ★ 挖矿批测：候选因子 IC + 全池相关（新因子入口）
│   ├── selected_*_{model}.json # 各模型入模清单（定稿 20d 32 / 6d 4 / open2d 3 / gap1d 35）
│   ├── build_mf_volchg3.py    # XGBoost 波动率变化率因子（用户裁定暂放）
│   └── build_gb_gap1d.py      # XGBoost 隔夜跳空因子（11 公式因子，rank IC 0.198）
│   ├── folds/{F1..F7}/      # 折专属筛选清单（防筛选泄漏，fold_cv 消费）
│   └── migrate_*.py         # 历史回填工具（short/mined/delivery/calendar_gap/intraday_shape）
│
├── strategies/
│   ├── base.py              # BaseStrategy + walk_forward() + buffered_train_end()（排他上界）
│   ├── labels.py            # ★ compute_median_open（回归标签）+ compute_nextopen_limit_mask（比率口径）
│   ├── combine.py           # combine_scores3（v8 三模型融合，缺失侧重归一）
│   ├── lgb.py               # LGBStrategy（回归版）
│   └── evaluation.py        # rank IC / IC 汇总
│
├── backtest/
│   ├── run_lgb.py           # ★ 主回测：融合分 + 开盘市价执行 + 卖出零点平移；权重/列名单源
│   ├── signals.py           # 组合模拟器（market_open 语义/ST/退市/费用/长空诊断件）
│   └── mainboard_microcap/  # 输出（equity_lgb_combined_daily_v8mo_* 等 + folds/）
│
├── forecast_display/
│   ├── generate_lgb.py      # ★ v8 报告：读三 parquet+meta 融合出榜；三级降级永不 exit 1
│   └── html_lgb/{pool}/     # HTML 报告
│
├── models/{pool}/           # lgb_{model}.joblib ×4 + folds/（旧 lgb_multi.joblib 为泄漏三分类，勿用）
├── data/                    # 摄入层与主区同构（pull/sources/trading_calendar/lock）；ashare.duckdb → 主区符号链接
├── document/llm_factor_mining/  # LLM 挖矿假设库（300 条）
└── docs/superpowers/        # spec 与 plan（dual-regression-models）
```

## Key Architectural Decisions

### 1. 因子分层范式与五铁律（单源：`factors/registry.py` 头注释）

```
第0层数据表 → 第1层公式因子（确定性公式）→ 第2层模型因子（OOF）→ 第3层主模型 → 第4层 combiner（手工权重）
```

- **DAG 无环**（构造性保证：模型因子输入只允许公式因子与 OHLCV，永不引用任何模型输出）
- **OHLCV 一律后复权**（hfq 水平时点诚实、永不重绘；水平型因子禁用 qfq——LogClose 教训）
- **删除公式因子前必须查反向依赖**（registry 血缘）
- **模型因子无结构特权**（不是门控、不进模型结构）
- **分层不混同**：模型因子不参与公式因子簇竞争；准入走独立门=对整个在任集合的边际贡献 A/B

裁定史与实验全记录在 agent memory（dual-regression-bench-status），勿在代码注释外重复堆细节。

### 2. 三模型 + 融合 + 输出校准
- **模型**（`config.py MODEL_CONFIGS`）：open2d（T+2 开盘/open[T+1]，buffer 2）、6d（T+4~6 中位开盘，buffer 6）、20d（T+16~20，buffer 20）；均 `objective=regression_l1`、固定测试集 walk-forward（训练 2020 起，测试 2025-06-01~2026-06-01）
- **融合**：`score = 0.4×p2d + 0.35×p6d + 0.25×p20d`（`strategies/combine.py combine_scores3`，缺失侧重归一）；权重是用户手工配比，**勿改**；调优留到实盘前
- **输出校准**（run_lgb.py 训练尾部）：训练窗留出尾段 60 交易日训校准模型测 OOS 收缩斜率 k（L1 过原点=|x| 加权中位数），输出×k；meta 存 `calib_slope`/`calib_median`。parquet 里的预测**已乘 k**（诚实幅度）；模型 joblib 的裸输出**未乘**（如手动推理需自行套用 meta 斜率）
- **卖出零点**：回测侧 `sell_threshold = Σwᵢ×calib_medianᵢ`（L1 中位数输出的典型水平为负）；仅平移卖出判定，排序与 parquet 不变

### 3. 标签体系（`strategies/labels.py`）
- 回归标签 `compute_median_open(kline, start_day, end_day, baseline)`：前向窗口开盘价中位数 / baseline − 1；`baseline="next_open"`（open[T+1]，与回测开盘市价成交对齐）
- **锚裁定史**：close 锚曾在 v4/v6 使用、next_open 在 v3/v7/v8 使用，最终 v8 全系 next_open（分母=可成交入场价）；close 锚 open2d 永久废弃（不可交易的隔夜跳空污染）
- `label_buffer`（排他上界，`buffered_train_end`）：训练掩码截到 test_start 前 buffer 个交易日；**buffer ≥ 标签窗末日**（20d=20/6d=6/open2d=2/gap1d=1）
- 标签只做**部分窗口中位数**（退市前真实可成交价携带崩盘信号），全缺窗口 NaN 丢弃；无 -1.0 归零填充（死代码已证不触发，退市风险在组合层处理）

### 4. 执行语义（开盘市价）
- 买=次日开盘必成交（取分数前 k 填空仓位）；卖=score 低于平移零点时开盘市价卖出，否则持有（无目标价止盈）
- `backtest/run_lgb.py` 的 `PRED_COLS`/`W2D/W6D/W20D` 是融合列名与权重的**单源**（forecast_display 直接 import）
- **limit 执行语义已判死**（7 折平均 +3.9% 跑输基准；`--exec limit` 仅对照）
- 多空信号（`run_long_short`）是纯诊断件，历史极不稳定，勿当版本优劣依据

### 5. 泄漏断言（`_leak_check.py`，2026-08-23 重写）
对主窗口盘上产物四类断言（52 项，当前全过，exit 1=有失败）：
- **C1** 训练掩码终点=重算 `buffered_train_end`；末训练日标签最远引用 < 测试窗首日
- **C2** 输出校准尾段全部 < train_end 且不触测试期
- **C3** 预测行集与复刻过滤链（IsST/退市/次日封板排除）逐行一致，行数=meta.n_pred
- **C4** 标签函数源码只含 `shift(-d)` + 合成面板对拍
折产物由 `fold_cv.py` 内置 `leak_checks` 覆盖。重训/改标签/改 buffer 后必跑。

### 6. 折 CV 验证框架（`fold_cv.py`）
连续 7 折半年窗（F1=2022H2 … F7=2025-06~2026-06），训练起点锁 2020 扩张窗口，每折**独立重跑筛选+训练+双语义回测**（折清单强制读 `factors/folds/{fid}/`，防筛选泄漏）。44 因子版干净折（全修复语义）：market 平均 +23.4%/半年 vs 基准 +9.4%、6/7 折超额为正、最差折 −17.5%（F4 崩盘折超额 +6.8pp）、平均夏普 1.93。**搭便车清洗版折验证（`fold_cv_slim.py`，2026-08-23 用户裁定"精简版"）**：每折重演 v8 配方的机械段——贡献分析（`factor_contribution --fold`）→ 零增益且零边际贡献因子剔除 → 重训 → market 回测；结果 平均 +23.9%/最差折 −7.9%（崩盘折 F4 防御大幅优于胖版）/平均夏普 1.82/超额 5/7（F2 失守）。诚实边界：清洗在折自己的测试窗上度量（配方复现，非纯 OOS）；未做 6d/open2d 极限瘦身（4/3 清单含人工裁定，不重演）；规则清洗幅度折间差异大（0~31 个）——对模型形态敏感，非处处温和。**v8 正式版（34/4/3）官方折验证经用户裁定跳过**（2026-08-23，省时）——引用折数字时注意区分版本；同日清单经折存活投票**定稿为 32/4/3**（见横幅），折内清洗产物已入库。纪律：F1-F6 开发迭代、F7/新数据终裁；`--skip-train` 复用折产物只重跑回测。

### 7. ST/退市三层防御（训练排除 + 回测过滤 + 报告过滤同款）
1. `excluded_codes`：名称快照含 "ST"/"退"（兜底）
2. `IsST` 因子（namechange 解析，当日时点，主力）
3. `delist_info`（date >= delist_date）

预测 parquet 在训练入口已排除三类观测（故下游计数为 0 属预期）。

### 8. IC 口径
测试集 IC 剔除：① 次日开盘封板观测（`compute_nextopen_limit_mask`，纯比率判断 ±0.05% 容差——qfq 价格网格判断已修）② 当日 IsST=1。训练集同样排除三类（见 #7）。

### 9. gap1d / open2d 实验模型
- **gap1d**（隔夜跳空，close 锚窗口(1,1)）：独立实验，IC ~0.19（隔夜信息最浓、跨风格稳），**不入 score/回测**；清单为历史拷贝（不随 6d 自动同步）
- **open2d**：已转正入融合（2026-08-22）。锚对照实验定论：T 时点可预测性浓缩在 T→T+1 隔夜段，第二夜跳空基本不可预测

### 10. DuckDB 单一数据源（与主区共享）
所有行情/因子在 `data/ashare.duckdb`（符号链接→主区）。前复权价格通过 VIEW `daily_kline`（`raw × adj_factor / latest_adj`，2026-08-20 修复分子分母写反）。**factor_values 双写入方分列所有权**：主区 compute/update 拥有 ~52 列基础因子；本分支拥有挖矿/短窗/mf_* 列。`(code,date)` 行级 INSERT/UPDATE，**勿整行替换**，date 保持 VARCHAR。**单写者锁跨进程互斥**：写库前 `ps aux | grep -E 'data\.pull|factors\.update'`，避开工作日 21:05 前后。

### 11. 换手率/市值来源（与主区同）
- 换手率实时算：`amount / NULLIF(circ_mv, 0) / 10`（`daily_kline.turn` 为 NULL）
- `total_mv`/`circ_mv` 来自 `daily_basic`（单位万元；`daily_raw` 同名字段全 NULL）

### 12. 已判死清单（勿再提出）
close 锚 open2d ｜ limit 执行语义 ｜ qfq 水平因子（latest_adj 未来信息）｜ 显式门控逻辑（非线性关系由模型自学）｜ 模型因子入簇竞争 ｜ ai_gz2000_*（in-sample 泄漏已删）

### 13. 数据库表清单
与主区同构（stock_info/daily_raw/daily_basic/daily_kline/cyq_perf/industry/index_daily/namechange/delist_info/trading_calendar/pending_pulls），差异仅在 `factor_values`：本分支额外写入挖矿因子、短窗因子（Return_3d 等 6 个）、水平替换（ATR_pct/MACD_hist_pct/CloseBIAS_20d）、模型因子（mf_*，所有权归 build_model_factors）。表结构详情见主区 AGENTS.md。

## Common Workflows

所有命令默认 `mainboard_microcap` 池；解释器用主区 `.venv`（`/Users/cui/Projects/quantlab/.venv/bin/python`，系统 python3 无 duckdb）。

### 数据更新（与主区同构）
```bash
python -m data.pull            # 增量拉取（勿用 --full，2-4 小时）
python -m factors.update       # 增量因子（本分支新列在此写入）
python -m factors.integrity    # 独立完整性校验
```

### 训练与评估
```bash
python run_lgb.py                    # ★ 训练三模型+gap1d（约 5 分钟），写 models/ + predictions parquet + meta
python _leak_check.py                # ★ 泄漏断言（52 项，重训后必跑）
python -m backtest.run_lgb           # ★ 主回测（开盘市价，融合分+卖出零点）
python fold_cv.py                    # 7 折全链（~35 分钟）；--skip-train 只重跑回测
python fold_cv_slim.py               # 搭便车清洗版折验证（~80 分钟，单折 ~3 分钟）
```

### 预测报告（三级降级，永不 exit 1）
```bash
python forecast_display/generate_lgb.py   # L1 完整 / L2 DOWNGRADED（缺成分重归一）/ L3 红色占位
```
报告读三 parquet+meta 融合出榜；**刻意不回退旧 lgb_multi.joblib**（2026-08-13 泄漏模型）。注意 parquet 止于 TEST_END，报告日期=产物最新可用日。

### 挖矿工作流（新因子主路径）
```bash
# 1. 候选批测（原生 pandas 实现，IC + 全池 max 相关）
python factors/test_new_factors.py
# 2. 画像审计（衰减/冗余对/口径）与模型内贡献（gain/permutation）
python factors/factor_audit.py
python factors/factor_contribution.py
# 3. 入池重筛（簇优先，每模型一份清单）→ 重训 → 回测
python -m factors.select_factors --model 20d
python run_lgb.py && python -m backtest.run_lgb
```
纪律：加因子看 train-test 泛化缺口（20d 当前正则下事实容量 ~44 因子，靠替换不靠堆加）；相关性判定对全池取 max（<0.75 增量 / >0.95 冗余）；强因子替换弱因子优先。模型因子走 `build_model_factors.py` + registry 注册，独立准入门。

### 门禁与审查
```bash
python -m factors.baseline_check     # 评估器回归门禁（改标签/评估器必跑）
python _leak_check.py                # 泄漏断言
```

## Important Constraints

- **勿启动 `--full` 全量构建**（2-4 小时，13800+ 次 API）；日常增量 `python -m data.pull`
- **DB 单写者锁**（与主区共享）：写库前确认无 `data.pull`/`factors.update` 在跑；避开工作日 21:05 前后（流水线窗口，当前暂停但勿赌）；从持库进程 spawn 写库子进程前先 `con.close()`
- **勿提交 DuckDB/.env**（.env 为符号链接）
- **Tushare 中继限流** 200 次/分钟（上限 600）
- **无 notebook**；分析一律 Python 脚本
- **权重/列名单源**：改融合相关代码从 `backtest/run_lgb.py` import，勿复制常量
- **主区未提交改动不是本分支的事**：主工作区 fix 分支由另一 session/流水线维护，本分支只动 worktree 内文件

## Coding Conventions

- 新公式因子加 `factors/extra_factors.py`（原生 Polars）；候选批测用 `factors/test_new_factors.py` 范式（pandas，口径对齐筛选）
- 模型因子注册进 `factors/registry.py`（白名单输入+超参指纹），构建走 `build_model_factors.py`
- 新策略继承 `strategies/base.py`；时序窗口只用向后 shift
- 类型标注按需；遵循各模块已有风格；路径基于 `__file__`

## 八荣八耻
以瞎猜接口为耻，以认真查询为荣；
以模糊执行为耻，以寻求确认为荣；
以臆想业务为耻，以人类确认为荣；
以创造接口为耻，以复用现有为荣；
以跳过验证为耻，以主动测试为荣；
以破坏架构为耻，以遵循规范为荣；
以假装理解为耻，以诚实无知为荣；
以盲目修改为耻，以谨慎重构为荣。
