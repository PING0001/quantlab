# Quantlab — AI Agent Instructions

> **🏛 唯一工作区**：`/Users/cui/Projects/quantlab`（`main` 分支）。2026-09-04 用户授权全项目清理后，worktree 与多余分支已全部删除（原 quantlab-dual / quantlab-bench-mainboard-all 已不存在，分支只剩 main），本仓即一切。DB 实体 `data/ashare.duckdb`（5.8GB）与 `.env` 在仓内但勿提交。
>
> **架构现役基线（2026-09-03 简化版，七折口径）**：market 平均 **+16.6%/夏普 1.19** vs 半年重置等权基准 +5.6%（超额 **+11.0pp**，5/7 折为正；最差折 F4 −18.2% vs 基准 −16.3%）；模型平均 test IC/ICIR：**20d 0.135/0.99、6d 0.103/0.88、open2d 0.064/0.45**；主窗口（时点池口径）超额 **+3.84%**。
>
> **纪律**：F1-F6 做开发迭代、F7/新数据终裁；收益判定看折超额/alpha（主窗口 ≈ 0.53×池 beta + 年化 ~13% alpha）。裁定史与实验全记录在 agent memory（dual-regression-bench-status）；spec/plan 见 `docs/superpowers/`。

## Project Overview

Quantlab 是一个 **A股主板微盘股量化选股系统**（时点池 1-40 亿流通市值带，各档约数百只）。三个 LightGBM 回归模型分别预测 open2d / 6d / 20d 前向开盘收益（next_open 锚，条件中位数），手工权重融合成单一 score 驱动每日开盘市价组合。核心流程：

```
Tushare 数据 → DuckDB 存储（data/pull）
  → 因子管道（factors/update：增量日更 + --full 全量；公式因子 Polars）
  → 模型因子（factors/build_gb_gap1d：全局 OOF 日更 + 随折同步 scoped，双池共用）
  → 回归模型训练（run_lgb.py，三主模型，L1 目标 + 输出校准；清单=人工裁定 json）
  → 融合分 → 回测（backtest/run_lgb.py，开盘市价模拟器内置）→ HTML 预测报告（三级降级）
  → 泄漏断言（_leak_check.py）+ 折 CV（fold_cv.py）
```

数据更新标准四步（原 cron 自动化已于 2026-09-04 删除，现手动执行）：`data.pull` → `factors.update` → `factors.build_gb_gap1d`（全局 OOF，分钟级，刷新全历史+前沿）→ `forecast_display/generate_lgb.py`（fail-fast 顺序，前步失败即停；模块路径是契约，改名需全链核对）。

## Technology Stack

| 层 | 技术 |
|------|-----------|
| 数据源 | Tushare（通过 quicksync.cn 中继，限流 200 次/分钟） |
| 数据库 | **DuckDB**（嵌入式 OLAP，所有数据单一来源） |
| 数值计算 | numpy, pandas, scipy, polars |
| 机器学习 | **LightGBM ×3**（LGBMRegressor，objective=regression_l1）+ XGBoost/sklearn-MLP（模型因子） |
| 序列化 | joblib（模型）、parquet（预测）、JSON（meta/清单） |
| 配置 | python-dotenv（.env）、config.py（中心配置） |

## Project Structure

```
quantlab/
├── config.py                # ★ 中心配置：DB 路径、MODEL_CONFIGS（三模型标签窗/锚/buffer）、FOLDS、SELECTED_FACTORS、PRED_COLS/W2D/W6D/W20D 融合单源（池概念已清零）
├── dataset.py               # ★ 训练装配单点：装载（factors/kline/delist/industry/IsST）+ assemble 束 + training_panel_index/label_far_cross 纯函数 + 训练协议常量 + DATA_FLOOR
├── run_lgb.py               # ★ 训练入口：装配→训练→校准→落盘（三主模型，L1 + 输出校准；折模式权重覆盖主路径）
├── fold_cv.py               # 滚动 7 折 CV：每折 ML 因子同步训练→三主模型→泄漏断言→回测（ICIR+收益+夏普汇总）
├── _leak_check.py           # ★ 主窗口泄漏断言（C1 训练掩码/C2 校准尾段/C3 样本排除/C4 标签方向，43 项=三模型×14+1；IO 独立构建 + dataset 共享纯函数）
│
├── factors/
│   ├── extra_factors.py     # ★ 公式因子主载体（原生 Polars）；新公式因子加这里
│   ├── store.py             # ★ 因子表 SQL 唯一点（铁律）：读写/对账/列操作/staleness_audit；表名一律经 spec
│   ├── update.py            # ★ 因子管道编排：增量日更（日期+股票级对账）+ --full 全量重建；compute_panel 计算内核
│   ├── integrity.py         # 完整性校验（硬失败 exit 1 / 软警告；integrity_report_{pool}.json）
│   ├── build_gb_gap1d.py    # XGBoost 隔夜跳空因子（--pool；全局 OOF 日更 + --cutoff/--through scoped 折同步；双池共用，2026-09-04 起唯一 ML 因子）
│   └── selected_{pool}_{model}.json # 各池各模型入模清单（mb1：20d 32/6d 10/open2d 11；微盘：20d 32/6d 5/open2d 4）
│
├── strategies/              # 策略库（池无关）
│   ├── labels.py            # ★ 标签（compute_median_open 回归标签 + compute_forward_returns 门禁基准 + compute_nextopen_limit_mask 比率口径）；泄漏断言 C4 依赖其源码可 inspect
│   └── lgb.py               # ★ LGBStrategy（纯回归）+ walk_forward（固定测试集）+ buffered_train_end + rank_ic/ic_summary + combine_scores3 + _rank_ic_np 积木
│
├── backtest/
│   ├── run_lgb.py           # ★ 主回测：融合分 + 开盘市价模拟器 + 时点池基准；--pool
│   └── {pool}/              # 输出（gitignore，可再生）
│
├── forecast_display/
│   └── generate_lgb.py      # ★ v8 报告：三 parquet+meta 融合出榜 + LIVE 前沿实时推理；三级降级永不 exit 1；--pool
│
├── pools/
│   ├── spec.py              # ★ 唯一池注册表：PoolSpec（snap_table/factor_table/band/data_since + 路径族方法）；python -m pools.spec 自描述
│   └── membership.py        # ★ 池时点化：查询 API（member_mask/union_codes/latest_codes/codes_on/reset_points）+ 快照构建器
│
├── data/                    # 摄入层（流水线首步，勿乱动；拉取范围=全部注册池快照并集）
│   ├── pull.py              # 统一拉取入口（增量/全量/对账；末尾 integrity 硬失败 exit 1）
│   ├── sources.py           # 数据源注册表（POOLS 自 spec）
│   ├── trading_calendar.py / lock.py / _ts.py / build_industry.py（pull 子进程调用）
│   └── ashare.duckdb        # DB 实体（勿提交）
│
├── models/{pool}/           # lgb_{model}.joblib ×3（单文件不留档，折训练直接覆盖）+ gb_gap1d.joblib（scoped 折同步落盘，全局模式不落盘）
├── document/                # 参考资料：llm_factor_mining（300 假设库）/ alpha101 论文 / tushare API 文档
└── docs/superpowers/        # spec 与 plan
```

## Key Architectural Decisions

### 1. 因子分层范式与五铁律（单源：本文件）
第1层**公式因子**（确定性公式，宽读法含市值/筹码源表，`factors/extra_factors.py`）→ 第2层**模型因子**（gb_/nn_ 独立脚本，walk-forward OOF，只吃公式因子+后复权 OHLCV，列所有权归各构建脚本）→ 第3层主模型 → 第4层 combiner（手工权重）。**五铁律**：DAG 无环（模型因子输入永不引用任何模型输出）；OHLCV 一律后复权（hfq 时点诚实、永不重绘，水平型因子禁用 qfq）；删除公式因子前查反向依赖；模型因子无结构特权（不是门控、不进模型结构）；分层不混同（不参与公式因子簇竞争，准入走独立门=对在任集合的边际贡献 A/B）。

### 2. 三模型 + 融合 + 输出校准（代码单源 strategies/lgb.py）
- **模型**（`config.py MODEL_CONFIGS`）：open2d（T+2 开盘/open[T+1]，buffer 2）、6d（T+4~6 中位开盘，buffer 6）、20d（T+16~20，buffer 20）；均 `objective=regression_l1`、固定测试集 walk-forward（训练 2020 起，测试 2025-06-01~2026-06-01）
- **融合**：`score = 0.3×p2d + 0.4×p6d + 0.3×p20d`（combine_scores3，缺失侧重归一）；权重是用户手工配比（2026-08-28 裁定），**勿改**；权威值= config.py `W2D/W6D/W20D`（backtest 与 forecast_display 同源 import）
- **输出校准**（run_lgb.py 训练尾部）：训练窗留出尾段 60 交易日训校准模型测 OOS 收缩斜率 k（L1 过原点=|x| 加权中位数），输出×k；meta 存 `calib_slope`/`calib_median`。parquet 里的预测**已乘 k**；模型 joblib 裸输出**未乘**（手动推理需自行套用 meta 斜率）
- **卖出零点**：回测侧 `sell_threshold = Σwᵢ×calib_medianᵢ`；仅平移卖出判定，排序与 parquet 不变

### 3. 标签体系（`strategies/labels.py`）
- 回归标签 `compute_median_open(kline, start_day, end_day, baseline)`：前向窗口开盘价中位数 / baseline − 1；`baseline="next_open"`（open[T+1]，与回测开盘市价成交对齐）；close 锚 open2d 永久废弃
- `label_buffer`（排他上界，`buffered_train_end`）：训练掩码截到 test_start 前 buffer 个交易日；buffer ≥ 标签窗末日
- 标签只做部分窗口中位数（退市前真实可成交价携带崩盘信号），全缺窗口 NaN 丢弃

### 4. 执行语义（开盘市价，唯一）
- 买=次日开盘必成交（取分数前 k 填空仓位，开盘一字涨停跳过）；卖=score 低于平移零点时开盘市价卖出（一字跌停顺延），否则持有（无目标价止盈）；T+1 锁定
- 退市持仓到达退市日强制清仓计零（与基准侧同口径）；封板判定纯比率口径 ±0.05% 容差

### 5. 泄漏断言（`_leak_check.py`）
对主窗口盘上产物四类断言（43 项，exit 1=有失败）：C1 训练掩码终点、C2 校准尾段、C3 预测行集与复刻过滤链逐行一致、C4 标签函数源码只含 `shift(-d)` + 合成面板对拍。折产物由 `fold_cv.py` 内置 `leak_checks` 覆盖。重训/改标签/改 buffer 后必跑。

### 6. 折 CV 验证框架（`fold_cv.py`）
连续 7 折半年窗（F1=2022H2 … F7=2025-06~2026-06），训练起点锁 2020 扩张窗口，每折 = **ML 因子同步训练（scoped 防泄漏：训练截止=折 test_start、测试窗冻结模型推理）→ 三主模型固定主清单训练（权重覆盖主路径）→ 泄漏断言 → market 回测**；汇总含每模型 test IC/ICIR + 收益/夏普。换清单 = tmp 重筛 + 就地改唯一 json（note 留痕）→ run_lgb + fold_cv 即新基线。

### 7. 股票池时点化与 ST/退市三层防御
- **池时点化**：沪深300式半年度快照（pool_snapshots 表，`pools/membership.py` 单源：查询 API + 快照构建器）；带宽 **1~40 亿流通市值**（通胀调整带）+ 主板 + 次新排除（上市 <252 交易日）；生效日=6/12 月首个交易日、选样截止=前一月末；基准=半年重置等权指数；宇宙口径与旧版本不可直接比较
- **现役双池**：`mainboard_all`（全 A 主板，**现役主工作线，2026-09-04 用户裁定**）与 `mainboard_microcap`（微盘，**封存不动**——2026-09-04 已换血 gb 并重训但七折未跑，战力仍以 nn 时代口径为准）；横截面参考系按池隔离，绝不可共表。池感知入口 = update/integrity/gb/run_lgb/_leak_check/fold_cv/backtest/generate_lgb（均有 `--pool`，缺省 env=微盘——**做 mb1 必须显式 `--pool mainboard_all` 或设 `QUANTLAB_POOL`**）；`data/pull`、`strategies/*`、`extra_factors.py` 计算内核不感知池。产物池命名 selected_{pool}_{model}.json / integrity_report_{pool}.json / fold_cv_report_{pool}.json（跨池互覆写已根治）。池代码一律走 membership（config 无 json 池读取）
- **ST/退市**（时点口径，三层）：① 日度 IsST 因子（namechange 区间解析，含变级修复）② delist_info（date >= delist_date）③ 训练排斥语义="仅训练"（ST/退市/封板/标签远引用越界不进训练但预测照常输出，回测宇宙不被 T+1 信息条件化；回测候选过滤在模拟器内执行）

### 8. IC 口径
测试集 IC 剔除：① 次日开盘封板观测（`compute_nextopen_limit_mask`，纯比率判断 ±0.05% 容差）② 当日 IsST=1。训练集同样排除。

### 9. ML 因子（模型因子）
- **gb_gap1d**（XGBoost，双池在册：微盘 6d/open2d + mb1 6d/open2d 各含 1 列）：唯一 ML 因子（nn_gap1d 已 2026-09-04 整体退役删除，微盘清单同日 nn→gb 换血）；随折同步训练（终态权重 `models/{pool}/gb_gap1d.joblib`，训练截止=F7 test_start）；日更 = 全局 OOF 重建（分钟级，2018-12 前缺失由 Boost 容忍）
- 因子表列所有权：公式因子列归 factors/update；gb_/nn_ 列归各构建脚本；全量重建自动保全他方列

### 10. DuckDB 单一数据源
所有行情/因子在 `data/ashare.duckdb`。前复权价格通过 VIEW `daily_kline`（`raw × adj_factor / latest_adj`）。**单写者锁跨进程互斥**：写库前 `ps aux | grep -E 'data\.pull|factors\.update'` 确认无流水线在跑。`(code,date)` 行级 INSERT/UPDATE，勿整行替换（增量路径）；date 保持 VARCHAR。**membership 查询在持有写连接的进程内必须传 con**（`union_codes(con=con)`），否则自开只读连接会撞锁。换手率实时算：`amount / NULLIF(circ_mv, 0) / 10`；`total_mv`/`circ_mv` 来自 daily_basic（单位万元）。

### 11. 数据窗口裁定
模型训练/回测/判断只用 2020-01-01 起的交易日；数据装载最早回看 2019-01-01（最长因子回看 ≤1 年 headroom）。单源 `dataset.DATA_FLOOR`，`load_factors/load_kline` 钳制式封顶——2020 前行集本就被 `training_panel_index` 截去，封顶对训练/预测/回测零漂移，纯装载瘦身。

### 12. 挖矿 = 临时脚本纪律（挖矿层已删，无正式入口）
一律 tmp/ 脚本（gitignored）；积木 = `dataset.load_factors(cols=)`（按需列）+ `strategies.lgb._rank_ic_np` + `factors.store`。纪律：加因子看 train-test 泛化缺口；冗余对全池 max（<0.75 增量/>0.95 冗余）；强因子替换弱因子优先。

### 13. 已判死清单（勿再提出）
close 锚 open2d ｜ limit 执行语义（已物理删除）｜ qfq 水平因子 ｜ 显式门控逻辑（门控类非线性关系由模型自学）｜ 模型因子入簇竞争 ｜ ai_gz2000_*（泄漏已删）｜ 长空信号当版本依据 ｜ mf_ 注册表机制 ｜ gap1d 主模型（已降位归 ML 因子层）｜ nn_gap1d（MLP 跳空因子，2026-09-04 整体删除，微盘清单已换 gb_gap1d）

### 14. 数据库表清单
stock_info/daily_raw/daily_basic/daily_kline(VIEW)/cyq_perf/industry/index_daily/namechange/delist_info/trading_calendar/pending_pulls/macro_daily(shibor)。
池资产（每池一套，见 pools/spec.py）：mainboard_microcap = pool_snapshots + factor_values（72 列）；mainboard_all = pool_snapshots_mainboard_all + factor_values_mainboard_all（70 列）。陈旧值审计：`python -m factors.store [--pool X]`（抽样重算 vs 存量 diff，只报告不修复；已知微盘 2020+ 存在 kline 重述致 ~13.3k 行级漂移）。

## Common Workflows

所有命令默认 `mainboard_microcap` 池（env `QUANTLAB_POOL` 或各入口 `--pool` 切换）；解释器用 `.venv/bin/python`（系统 python3 无 duckdb）。

### 数据更新（四步流水线，手动执行）
```bash
python -m data.pull            # 增量拉取（勿用 --full，2-4 小时）
python -m factors.update       # 增量因子（--dry-run 预览 / --backfill-stocks 回补）
python -m factors.update --full   # 全量重建 factor_values（勿轻易运行）
python -m factors.build_gb_gap1d   # gb 因子全局 OOF 重建（~1-3 分钟，刷新全历史+前沿；--pool 可换池）
python -m factors.integrity    # 独立完整性校验
python -m pools.membership     # 重建池快照（半年度，通常 6/12 月跑一次）
```

### 训练与评估
```bash
python run_lgb.py                    # ★ 训练三主模型（约 5 分钟），写 models/{pool}/ + predictions parquet + meta
python _leak_check.py                # ★ 泄漏断言（43 项，重训后必跑）
python -m backtest.run_lgb           # ★ 主回测（开盘市价，融合分+卖出零点）
python fold_cv.py                    # 7 折全链（含 ML 同步训练，~80-100 分钟）
python -m pools.spec                 # 池注册表自描述（表名/带宽/行数/最新档）
# 换池：QUANTLAB_POOL=mainboard_all <命令>  或  <命令> --pool mainboard_all
```

### 预测报告（三级降级，永不 exit 1）
```bash
python forecast_display/generate_lgb.py   # L1 完整 / L2 DOWNGRADED / L3 红色占位
```
报告读三 parquet+meta 融合出榜；LIVE 通道用冻结模型对最新因子日实时推理。

## Important Constraints

- **勿启动 `--full` 全量构建**（pull 2-4 小时 13800+ API；factor --full 整表重建）
- **DB 单写者锁**：写库前确认无 `data.pull`/`factors.update` 在跑；membership 在写连接进程内查询必须传 con
- **勿提交 DuckDB/.env**；**无 notebook**，分析一律 Python 脚本
- **权重/列名单源**：融合权重以 `config.py`（W2D/W6D/W20D）为权威，backtest 与 forecast_display 同源 import，勿复制常量
- **四步流水线契约**（data.pull / factors.update / factors.build_gb_gap1d / forecast_display/generate_lgb.py）：顺序与模块路径即生产流程，重排或改名需全链核对（若重建自动调度须同步调度侧）
- **池代码单源 pools/membership**：不新增任何 json 池读取
- **模型权重单文件不留档**：折训练直接覆盖 models/{pool}/ 主路径，跑完=F7/主窗口口径

## Coding Conventions

- 新公式因子加 `factors/extra_factors.py`（原生 Polars）；批测/筛选走 tmp/ 临时脚本（积木见挖矿节）
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
