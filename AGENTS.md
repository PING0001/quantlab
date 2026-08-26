# Quantlab — AI Agent Instructions

> **⚠️ 本仓库（`/Users/cui/Projects/quantlab`，`main` 分支）是双回归体系唯一工作区（2026-08-24 从 quantlab-dual 合并入 main；dual worktree 仍留在磁盘上但已停用，不碰）。2026-08-25 完成破坏性简化瘦身（用户指令"合并/删功能/删重写"）：py 文件 62→28（净删 35 个、新增 mining.py），详见 `docs/superpowers/specs/2026-08-25-destructive-simplification-design.md`**
>
> **🏛 本 worktree（`quantlab-bench-mainboard-all`，分支 `bench-mainboard-all`）：2026-08-27 全链多池化重写已落地（方案 `docs/superpowers/plans/2026-08-27-multipool-fullchain-rewrite.md`）。微盘全链等价门全绿（重训 parquet 逐值相等 / 泄漏断言 56/56 / baseline 零漂移 / 回测对横幅精确复现）。在此分支工作时：**
> - **环境**：`QUANTLAB_DB=/Users/cui/Projects/quantlab/data/ashare.duckdb`（共享主仓 DB 实体，本 worktree 无本地 DB）+ `QUANTLAB_POOL`（缺省微盘；或各入口 `--pool`）。主仓 `main` 不动。
> - **三个单源模块（架构核心）**：`pools/spec.py`（唯一池注册表 PoolSpec：snap_table/factor_table/band/data_since/路径族方法；`python -m pools.spec` 自描述）、`factors/store.py`（因子表 SQL 唯一点，**铁律：池因子表 SQL 只准出现在 store.py**）、`dataset.py`（训练装配单点：装载 + training_panel_index/label_far_cross 纯函数 + 训练协议常量）。config 已池概念清零（无池表名/池路径）。
> - **现役双池**：`mainboard_microcap`（微盘，生产池，cron 无参默认）与 `mainboard_all`（全 A 主板，bench 验证池）。横截面参考系按池隔离，绝不可共表。
> - **池化边界**：池感知入口 = update/integrity/gb/nn/select_factors/mining/baseline_check/run_lgb/_leak_check/fold_cv/backtest/generate_lgb/ic_probe（均有 `--pool`，缺省 env）；`data/pull`、`strategies/*`、`extra_factors.py` 计算内核、cron 四步路径不感知池（契约：无参=微盘）。
> - **产物池命名**：selected_{pool}_{model}.json、integrity_report_{pool}.json、factor_audit/contribution_report_{pool}.json、baseline_reference_{pool}.json、fold_cv_report_{pool}.json（跨池互覆写已根治）。
> - **死列已清**（2026-08-27）：GZ2000 7 死列 + shibor_on/1m + alpha*_v0 ×4 + mf_volchg3 共 14 列（计算源头+双表 DROP，保留 GZ2000_return_5d/20d）；微盘表 72 列、全主板表 70 列。
> - **陈旧值审计**：`python -m factors.store [--pool X]`（staleness_audit：抽样重算 vs 存量 diff，只报告不修复；已知微盘 2020+ 存在 kline 重述致 ~13.3k 行级漂移）。
>
> **本分支现役架构（v8 正式版）速览**：
> - **三模型 LightGBM 回归（objective=regression_l1，条件中位数），全部 next_open 锚**：open2d / 6d / 20d（+gap1d 独立实验模型）；训练起点锁 2020-01，固定测试集 walk-forward
> - **股票池已时点化（2026-08-24 用户四项裁定）**：沪深300式半年度快照（pool_snapshots 表，`pools/membership.py` 单源：查询 API + 快照构建器）；带宽 **1~40 亿流通市值**（通胀调整带）+ 主板 + 次新排除（上市 <252 交易日）；生效日=6/12 月首个交易日、选样截止=前一月末。**旧池 json（历史并集，非时点）已于 2026-08-25 物理删除**——池代码一律走 membership（config 不再有 json 池读取）；基准=半年重置等权指数。宇宙口径已换，与旧版本数字不可直接比较
> - **现役清单（2026-08-24）：20d 广谱 32（折存活投票定稿）/ 6d 极简 5（4+nn_gap1d）/ open2d 极简 4（3+nn_gap1d）**——无候补名单制度，换人只经六关考核（批测→画像→贡献→边际贡献门→泛化缺口→折存活投票）。**2026-08-24 审计修复批后实测**：三 IC **0.1421/0.1074/0.0725**（20d 历史最高），gap1d 0.1747；主窗口（时点池口径）**+48.72%/2.47/−6.70% vs 半年重置等权基准 +26.17%/1.20/−14.12%，超额 +22.55pp**（**时点池口径七折已跑**：market 平均 +20.9%/半年 vs 基准 +9.5%、超额 5/7 为正（F2 −1.2/F6 −0.3pp 失守）、最差折 F4 −22.8% vs 基准 −26.7%（崩盘防御仍在）、平均夏普 1.44——宇宙诚实化后较旧口径（+23.4%/1.93）回落属预期，旧数字含未来名册偏差）。**ST/退市判定已时点化**（2026-08-24 裁定：名称快照层退役，日度 IsST + delist 日期；IsST 变级记录解析缺陷同步修复——'从ST变为*ST' 类记录此前漏开区间致 12 只 ST 股漏判）。训练排斥为"仅训练"语义（ST/退市/封板/标签远引用越界不进训练但预测照常输出，回测宇宙不再被 T+1 信息条件化）
> - **输出校准（根治幅度事故）**：训练窗留出尾段 60 交易日测 L1 收缩斜率与横截面中位数，输出×斜率还原为"模型真实相信的到期涨幅"；卖出零点平移 `score < Σwᵢ×calib_medianᵢ`（排序与 parquet 原始值不变）。教训：原始幅度加权融合对训练幅度漂移敏感（纯缩放 open2d 预测即可摆动主窗口 12.6pp）
> - **融合分** `score = 0.4×p2d + 0.35×p6d + 0.25×p20d`（手工权重，用户按幅度特性亲自配比；权重调优留到实盘前最后做）
> - **执行语义：开盘市价唯一**（买=次日开盘必成交取前 k，一字封板跳过；卖=score 低于平移零点时开盘市价卖出，一字跌停顺延）。**limit 执行语义已物理删除**（2026-08-25；判死于七折 +3.9% 跑输基准）；长空诊断件（run_long_short）同批删除（历史极不稳定，勿当版本优劣依据）
> - **因子分层范式（mf_ 注册表机制已删，铁律在此单源）**：第1层**公式因子**（确定性公式，宽读法含市值/筹码源表，`factors/extra_factors.py`）→ 第2层**模型因子**（gb_/nn_ 独立脚本，walk-forward OOF，只吃公式因子+后复权 OHLCV，列所有权归各构建脚本）→ 第3层主模型 → 第4层 combiner（手工权重）。**五铁律**：DAG 无环（模型因子输入永不引用任何模型输出）；OHLCV 一律后复权（hfq 时点诚实、永不重绘，水平型因子禁用 qfq）；删除公式因子前查反向依赖；模型因子无结构特权（不是门控、不进模型结构）；分层不混同（不参与公式因子簇竞争，准入走独立门=对在任集合的边际贡献 A/B）。现役模型因子：`gb_gap1d`（XGBoost 隔夜跳空，11 公式因子，OOS rank IC 0.198，盘上候审）、`nn_gap1d`（MLP 隔夜跳空，30 日 OHLCV 窗口+12 公式因子，已入 6d/open2d 清单，每交易日前沿推理）。旧 mf_ 注册表/build_model_factors/mf_volchg3 已删（2026-08-25，git 可恢复）
> - **挖矿三件套已合一**：`python -m factors.mining {audit|contribution|batch}`（画像/贡献/批测）。纪律：加因子看 train-test 泛化缺口；冗余判定对全池取 max（<0.75 增量/>0.95 冗余，用户标准）；强因子替换弱因子优先于堆加
> - **验证框架**：`python fold_cv.py`（7 折半年窗，每折独立筛选+训练+market 回测，含泄漏断言）；时点池口径官方折：market 平均 +20.9% vs 基准 +9.5%（见上）
> - **已判死（勿再提出）**：close 锚 open2d；limit 执行语义（已物理删除）；qfq 水平因子（latest_adj 未来信息）；显式门控逻辑（门控类非线性关系由模型自学，2026-08-23 裁定）；模型因子入簇竞争；ai_gz2000_*（in-sample 泄漏）；长空信号当版本依据（诊断件已删）
> - **纪律**：F1-F6 做开发、F7/新数据终裁；收益判定看折超额/alpha（主窗口 ≈ 0.53×池 beta + 年化 ~13% alpha）
> - 裁定史与实验全记录：agent memory（dual-regression-bench-status）；spec/plan 见 `docs/superpowers/`

## Project Overview

Quantlab 是一个 **A股主板微盘股量化选股系统**（时点池 1-40 亿流通市值带，各档约数百只）。三个 LightGBM 回归模型分别预测 open2d / 6d / 20d 前向开盘收益（next_open 锚，条件中位数），手工权重融合成单一 score 驱动每日开盘市价组合。核心流程：

```
Tushare 数据 → DuckDB 存储（data/pull）
  → 因子管道（factors/update：增量日更 + --full 全量；公式因子 Polars）
  → 模型因子（factors/build_gb_gap1d / build_nn_gap1d，OOF 管道）
  → 因子筛选（factors/select_factors，簇优先，相关度>IC，每模型一份清单）
  → 回归模型训练（run_lgb.py，L1 目标 + 输出校准）
  → 融合分 → 回测（backtest/run_lgb.py，开盘市价模拟器内置）→ HTML 预测报告（三级降级）
  → 泄漏断言（_leak_check.py）+ 折 CV（fold_cv.py）
```

**唯一工作区**：本仓库即主线（`main`）。`data/ashare.duckdb`（5.8GB）与 `.env` 是本仓库实体文件——DB 属主在此，勿提交。夜间流水线（cron automation-596d04c5，每交易日 21:05）四步：`data.pull` → `factors.update` → `factors.build_nn_gap1d --infer-only` → `forecast_display/generate_lgb.py`（fail-fast，模块路径是契约，改名必须同步改 cron）。

## Technology Stack

| 层 | 技术 |
|------|-----------|
| 数据源 | Tushare（通过 quicksync.cn 中继） |
| 数据库 | **DuckDB**（嵌入式 OLAP，所有数据单一来源） |
| 数值计算 | numpy, pandas, scipy, polars |
| 机器学习 | **LightGBM ×3**（LGBMRegressor，objective=regression_l1）+ XGBoost/sklearn-MLP（模型因子） |
| 序列化 | joblib（模型）、parquet（预测）、JSON（meta/清单） |
| 配置 | python-dotenv（.env）、config.py（中心配置） |

## Project Structure

```
quantlab/
├── config.py                # ★ 中心配置：DB 路径、MODEL_CONFIGS（四模型标签窗/锚/buffer）、FOLDS、SELECTED_FACTORS、PRED_COLS/W2D/W6D/W20D 融合单源（池概念已清零）
├── dataset.py               # ★ 训练装配单点：装载（factors/kline/delist/industry/IsST）+ assemble 束 + training_panel_index/label_far_cross 纯函数 + 训练协议常量
├── run_lgb.py               # ★ 训练入口：装配→训练→校准→落盘（三回归+gap1d，L1 目标 + 输出校准）
├── fold_cv.py               # 滚动 7 折 CV 驱动：每折独立筛选+训练+market 回测+泄漏断言（子进程显式 --pool）
├── _leak_check.py           # ★ 主窗口泄漏断言（C1 训练掩码/C2 校准尾段/C3 样本排除/C4 标签方向，56 项；IO 独立构建 + dataset 共享纯函数）
│
├── factors/
│   ├── extra_factors.py     # ★ 公式因子主载体（原生 Polars）；新公式因子加这里
│   ├── store.py             # ★ 因子表 SQL 唯一点（铁律）：读写/对账/列操作/staleness_audit；表名一律经 spec
│   ├── update.py            # ★ 因子管道编排：增量日更（日期+股票级对账）+ --full 全量重建；compute_panel 计算内核
│   ├── integrity.py         # 完整性校验（硬失败 exit 1 / 软警告 + check_errors 显性化；integrity_report_{pool}.json）
│   ├── select_factors.py    # ★ 筛选：簇优先（average-linkage，相关度>IC），每模型 selected_{pool}_{model}.json
│   ├── mining.py            # ★ 挖矿三件套：audit（画像）/ contribution（gain+permutation）/ batch（候选批测）
│   ├── baseline_check.py    # ★ 评估器回归门禁（8 基准 alpha + baseline_reference_{pool}.json 冻结比对）
│   ├── build_gb_gap1d.py    # XGBoost 隔夜跳空因子（11 公式因子，rank IC 0.198；--pool）
│   ├── build_nn_gap1d.py    # MLP 隔夜跳空因子（cron 每日 --infer-only 前沿推理；状态按池隔离）
│   ├── selected_{pool}_{model}.json # 各池各模型入模清单（微盘定稿 20d 32 / 6d 5 / open2d 4）
│   └── folds/{F1..F7}/      # 折专属筛选清单（防筛选泄漏，fold_cv 消费）
│
├── strategies/              # 策略库（池无关）
│   ├── labels.py            # ★ 标签（compute_median_open 回归标签 + compute_forward_returns 门禁基准 + compute_nextopen_limit_mask 比率口径）；泄漏断言 C4 依赖其源码可 inspect
│   └── lgb.py               # ★ LGBStrategy（纯回归）+ walk_forward（固定测试集）+ buffered_train_end + rank_ic/ic_summary + combine_scores3
│
├── backtest/
│   ├── run_lgb.py           # ★ 主回测：融合分 + 开盘市价模拟器 + 时点池基准；--pool；PRED_COLS/W 自 config
│   └── {pool}/              # 输出（equity_lgb_combined_daily_v8mo_rebalance.csv 等 + folds/）
│
├── forecast_display/
│   └── generate_lgb.py      # ★ v8 报告：三 parquet+meta 融合出榜 + LIVE 前沿实时推理；三级降级永不 exit 1；--pool
│
├── pools/
│   ├── spec.py              # ★ 唯一池注册表：PoolSpec（snap_table/factor_table/band/data_since + 路径族方法）；python -m pools.spec 自描述
│   └── membership.py        # ★ 池时点化：查询 API（member_mask/union_codes/latest_codes/codes_on/reset_points）+ 快照构建器
│
├── data/                    # 摄入层（cron 路径，勿乱动；拉取范围=全部注册池快照并集）
│   ├── pull.py              # 统一拉取入口（增量/全量/对账；末尾 integrity 硬失败 exit 1）
│   ├── sources.py           # 数据源注册表（POOLS 自 spec）
│   ├── trading_calendar.py / lock.py / _ts.py / build_industry.py（pull 子进程调用）
│   └── ashare.duckdb        # DB 实体（勿提交；bench worktree 经 QUANTLAB_DB 共享主仓）
│
├── models/{pool}/           # lgb_{model}.joblib ×4 + nn_gap1d_state.joblib + folds/（折产物）
├── document/                # 参考资料：llm_factor_mining（300 假设库）/ alpha101 论文 / tushare API 文档 / vnpy 源码拷贝（研究参考，不参与运行）
└── docs/superpowers/        # spec 与 plan
```
```

## Key Architectural Decisions

### 1. 因子分层范式与五铁律（单源：本文件；mf_ 注册表机制已删）
分层见横幅。铁律执行靠纪律与 review，不再有 registry 代码强制；gb_/nn_ 脚本头部注释声明列所有权与输入白名单。

### 2. 三模型 + 融合 + 输出校准（代码单源 strategies/lgb.py）
- **模型**（`config.py MODEL_CONFIGS`）：open2d（T+2 开盘/open[T+1]，buffer 2）、6d（T+4~6 中位开盘，buffer 6）、20d（T+16~20，buffer 20）、gap1d（窗口(1,1)+close 锚，buffer 1，独立实验不入 score）；均 `objective=regression_l1`、固定测试集 walk-forward（训练 2020 起，测试 2025-06-01~2026-06-01）
- **融合**：`score = 0.4×p2d + 0.35×p6d + 0.25×p20d`（combine_scores3，缺失侧重归一）；权重是用户手工配比，**勿改**；调优留到实盘前
- **输出校准**（run_lgb.py 训练尾部）：训练窗留出尾段 60 交易日训校准模型测 OOS 收缩斜率 k（L1 过原点=|x| 加权中位数），输出×k；meta 存 `calib_slope`/`calib_median`。parquet 里的预测**已乘 k**；模型 joblib 裸输出**未乘**（手动推理需自行套用 meta 斜率）
- **卖出零点**：回测侧 `sell_threshold = Σwᵢ×calib_medianᵢ`；仅平移卖出判定，排序与 parquet 不变

### 3. 标签体系（`strategies/labels.py`）
- 回归标签 `compute_median_open(kline, start_day, end_day, baseline)`：前向窗口开盘价中位数 / baseline − 1；`baseline="next_open"`（open[T+1]，与回测开盘市价成交对齐）
- 锚裁定史：v8 全系 next_open；close 锚 open2d 永久废弃（不可交易的隔夜跳空污染）
- `label_buffer`（排他上界，`buffered_train_end`）：训练掩码截到 test_start 前 buffer 个交易日；buffer ≥ 标签窗末日
- 标签只做部分窗口中位数（退市前真实可成交价携带崩盘信号），全缺窗口 NaN 丢弃

### 4. 执行语义（开盘市价，唯一）
- 买=次日开盘必成交（取分数前 k 填空仓位，开盘一字涨停跳过）；卖=score 低于平移零点时开盘市价卖出（一字跌停顺延），否则持有（无目标价止盈）；T+1 锁定
- 退市持仓到达退市日强制清仓计零（与基准侧同口径，审计 #9）；封板判定纯比率口径 ±0.05% 容差
- `backtest/run_lgb.py` 的 `PRED_COLS`/`W2D/W6D/W20D` 是融合列名与权重的**单源**（forecast_display 直接 import）

### 5. 泄漏断言（`_leak_check.py`）
对主窗口盘上产物四类断言（四模型共 56 项，exit 1=有失败）：C1 训练掩码终点、C2 校准尾段、C3 预测行集与复刻过滤链逐行一致、C4 标签函数源码只含 `shift(-d)` + 合成面板对拍。折产物由 `fold_cv.py` 内置 `leak_checks` 覆盖。重训/改标签/改 buffer 后必跑。

### 6. 折 CV 验证框架（`fold_cv.py`）
连续 7 折半年窗（F1=2022H2 … F7=2025-06~2026-06），训练起点锁 2020 扩张窗口，每折**独立重跑筛选+训练+market 回测**（折清单强制读 `factors/folds/{fid}/`）。时点池口径官方折见横幅（market 平均 +20.9%/最差 F4 −22.8%）。搭便车清洗版验证（fold_cv_slim）与 v8 折验证裁定跳过的记录见 agent memory；脚本已删（2026-08-25，git 可恢复）。纪律：F1-F6 开发迭代、F7/新数据终裁；`--skip-train` 复用折产物只重跑回测。

### 7. ST/退市三层防御（时点口径，2026-08-24 裁定）
① 日度 IsST 因子（namechange 区间解析，含变级修复，主力）② delist_info（date >= delist_date）③ 训练排斥语义="仅训练"（预测照常输出，下游各自过滤）。名称快照层已退役。回测候选过滤（IsST/退市）在模拟器内执行。

### 8. IC 口径
测试集 IC 剔除：① 次日开盘封板观测（`compute_nextopen_limit_mask`，纯比率判断 ±0.05% 容差）② 当日 IsST=1。训练集同样排除。

### 9. gap1d / 模型因子
- **gap1d**（主模型第四位，独立实验）：预测隔夜跳空，IC ~0.19，**不入 score/回测**；清单为 6d 清单历史拷贝
- **gb_gap1d**（XGBoost 特征列）：盘上候审；**nn_gap1d**（MLP 特征列）：已入 6d/open2d 清单，每交易日前沿推理（冻结模型，绝不重训）
- 因子表列所有权：公式因子列归 factors/update；gb_/nn_ 列归各构建脚本；全量重建自动保全他方列

### 10. DuckDB 单一数据源
所有行情/因子在 `data/ashare.duckdb`。前复权价格通过 VIEW `daily_kline`（`raw × adj_factor / latest_adj`）。**单写者锁跨进程互斥**：写库前 `ps aux | grep -E 'data\.pull|factors\.update'`，避开工作日 21:05 前后（cron 流水线窗口）。`(code,date)` 行级 INSERT/UPDATE，勿整行替换（增量路径）；date 保持 VARCHAR。**membership 查询在持有写连接的进程内必须传 con**（`union_codes(con=con)`），否则自开只读连接会撞锁。

### 11. 换手率/市值来源
换手率实时算：`amount / NULLIF(circ_mv, 0) / 10`；`total_mv`/`circ_mv` 来自 `daily_basic`（单位万元）。

### 12. 已判死清单（勿再提出）
close 锚 open2d ｜ limit 执行语义（已物理删除）｜ qfq 水平因子 ｜ 显式门控逻辑 ｜ 模型因子入簇竞争 ｜ ai_gz2000_*（泄漏已删）｜ 长空信号当版本依据 ｜ mf_ 注册表机制（已删，如重建走独立脚本范式）

### 13. 数据库表清单
stock_info/daily_raw/daily_basic/daily_kline(VIEW)/cyq_perf/industry/index_daily/namechange/delist_info/trading_calendar/pending_pulls/macro_daily(shibor)。
池资产（每池一套，见 pools/spec.py）：mainboard_microcap = pool_snapshots + factor_values（72 列）；mainboard_all = pool_snapshots_mainboard_all + factor_values_mainboard_all（70 列）。

**在途实验表（用户 2026-08-27 确认为新工作合法成果，勿清理）**：`pool_snapshots_mainboard_all`（全主板无市值带时点快照，23 档）+ `factor_values_mainboard_all`（配套因子表，79 列）+ 人读版 `pools/mainboard_all_history.json`（gitignore）。生成代码暂未入库——清理孤儿表前先问用户。

## Common Workflows

所有命令默认 `mainboard_microcap` 池（env `QUANTLAB_POOL` 或各入口 `--pool` 可切换；
本 bench worktree 另需 `QUANTLAB_DB=/Users/cui/Projects/quantlab/data/ashare.duckdb`）；
解释器用 `.venv/bin/python`（系统 python3 无 duckdb）。

### 数据更新（cron 每交易日 21:05 自动跑）
```bash
python -m data.pull            # 增量拉取（勿用 --full，2-4 小时）
python -m factors.update       # 增量因子（--dry-run 预览 / --backfill-stocks 回补）
python -m factors.update --full   # 全量重建 factor_values（勿轻易运行）
python -m factors.build_nn_gap1d --infer-only   # nn 因子前沿推理（冻结 MLP，秒级）
python -m factors.integrity    # 独立完整性校验
python -m pools.membership     # 重建池快照（半年度，通常 6/12 月跑一次）
```

### 训练与评估
```bash
python run_lgb.py                    # ★ 训练三模型+gap1d（约 5 分钟），写 models/{pool}/ + predictions parquet + meta
python _leak_check.py                # ★ 泄漏断言（56 项，重训后必跑）
python -m backtest.run_lgb           # ★ 主回测（开盘市价，融合分+卖出零点）
python fold_cv.py                    # 7 折全链（~35 分钟）；--skip-train 只重跑回测
python -m factors.spec               # 池注册表自描述（表名/带宽/行数/最新档）
python -m factors.store --pool X     # 因子表陈旧值审计（抽样重算 vs 存量，只报告）
# 换池：QUANTLAB_POOL=mainboard_all <命令>  或  <命令> --pool mainboard_all
```

### 预测报告（三级降级，永不 exit 1）
```bash
python forecast_display/generate_lgb.py   # L1 完整 / L2 DOWNGRADED / L3 红色占位
```
报告读三 parquet+meta 融合出榜；LIVE 通道用冻结模型对最新因子日实时推理。**刻意不回退旧 lgb_multi.joblib**（2026-08-13 泄漏模型）。

### 挖矿工作流（新因子主路径）
```bash
# 1. 候选批测（IC + 全池 max 相关）
python -m factors.mining batch
# 2. 画像审计（衰减/冗余对/口径）与模型内贡献（gain/permutation）
python -m factors.mining audit [--selected]
python -m factors.mining contribution [--fold F4]
# 3. 入池重筛（簇优先，每模型一份清单）→ 重训 → 回测
python -m factors.select_factors --model 20d
python run_lgb.py && python -m backtest.run_lgb
```
纪律：加因子看 train-test 泛化缺口；相关性判定对全池取 max（<0.75 增量 / >0.95 冗余）；强因子替换弱因子优先。模型因子走独立构建脚本（gb_/nn_ 前缀），独立准入门。

### 门禁与审查
```bash
python -m factors.baseline_check     # 评估器回归门禁（改标签/评估器必跑；--init 重冻结参考值）
python _leak_check.py                # 泄漏断言
```

## Important Constraints

- **勿启动 `--full` 全量构建**（pull 2-4 小时 13800+ API；factor --full 整表重建）
- **DB 单写者锁**：写库前确认无 `data.pull`/`factors.update` 在跑；避开工作日 21:05 前后；membership 在写连接进程内查询必须传 con
- **勿提交 DuckDB/.env**
- **Tushare 中继限流** 200 次/分钟
- **无 notebook**；分析一律 Python 脚本
- **权重/列名单源**：改融合相关代码从 `backtest/run_lgb.py` import，勿复制常量
- **cron 四步模块路径是契约**（data.pull / factors.update / factors.build_nn_gap1d / forecast_display/generate_lgb.py），改名必须同步改 cron automation-596d04c5
- **池代码单源 pools/membership**：不新增任何 json 池读取（2026-08-25 已删干净）

## Coding Conventions

- 新公式因子加 `factors/extra_factors.py`（原生 Polars）；候选批测 `python -m factors.mining batch`
- 模型因子：独立构建脚本（gb_/nn_ 前缀），头部声明列所有权+输入白名单+超参指纹
- 时序窗口只用向后 shift；类型标注按需；路径基于 `__file__`

## 八荣八耻
以瞎猜接口为耻，以认真查询为荣；
以模糊执行为耻，以寻求确认为荣；
以臆想业务为耻，以人类确认为荣；
以创造接口为耻，以复用现有为荣；
以跳过验证为耻，以主动测试为荣；
以破坏架构为耻，以遵循规范为荣；
以假装理解为耻，以诚实无知为荣；
以盲目修改为耻，以谨慎重构为荣。
