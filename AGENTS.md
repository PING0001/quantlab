# Quantlab — AI Agent Instructions

## Project Overview

Quantlab 是一个 **A股量化选股系统**，针对主板微盘股（流通市值 1-20 亿）进行多周期收益预测。核心流程：

```
Tushare 数据 → DuckDB 存储 → 因子计算（138个，含101个 Alpha101） → LightGBM 多周期训练（4 个独立模型） → 回测 → HTML 预测报告
```

## Technology Stack

| 层 | 技术 |
|------|-----------|
| 数据源 | Tushare（通过 quicksync.cn 中继） |
| 数据库 | **DuckDB**（嵌入式 OLAP，所有数据单一来源） |
| 数值计算 | numpy, pandas, scipy, polars |
| 机器学习 | **LightGBM**（主模型，4 个独立 LGBMRegressor 每 horizon 一个） |
| 序列化 | joblib（LightGBM）、parquet（预测缓存）、JSON（meta） |
| 配置 | python-dotenv（.env 中的 Tushare token）、config.py（中心配置） |

## Project Structure

```
quantlab/
├── config.py                # ★ 中心配置：DB 路径、股票池加载、各模块输出路径
├── pools/                   # 股票池定义（JSON，多池支持）
│   ├── mainboard_microcap.json   # ★ 默认池，~1112只主板微盘股（1e < circ_mv < 20e）
│   ├── smallcap_on_mainboard.json # ~2965只主板小市值股票（1e < circ_mv < 40e）
│   ├── mainboard_commodity_mega.json # 24只商品周期大盘股
│   └── build_microcap.py    # 微盘池构建：半年度 circ_mv 通胀调整筛选
│
├── data/                    # 数据摄入
│   ├── build_db.py          # 全量建库：按日拉全市场日线、复权因子、市值/估值（2008-2026）
│   ├── build_index_db.py    # 全量建库：拉取指数日线（中证全指 000985 等）
│   ├── build_cyq.py         # 全量/增量拉取筹码分布数据（cyq_perf, 2018-至今）
│   ├── build_delist_info.py # 全量拉取 namechange → 构建 delist_info + ISST 历史
│   ├── pull_adj.py          # 增量更新：每日行情 + 每日 namechange 增量 + 每日 namechange 增量
│   └── ashare.duckdb        # DuckDB 数据库，所有数据唯一来源
│
├── factors/                 # 因子工程（基于 vnpy 表达式 DSL 引擎）
│   ├── ops.py               # DataProxy：Polars 列的算术/比较运算符重载
│   ├── ts_ops.py            # 22 个时序算子（ts_delay, ts_rank, ts_corr 等）
│   ├── cs_ops.py            # 5 个横截面算子（cs_rank, cs_mean, cs_std 等）
│   ├── math_ops.py          # 9 个数学/控制流函数（sign, pow1, quesval 等）
│   ├── utility.py           # 表达式求值引擎 calculate_by_expression()
│   ├── alpha101.py          # 101 个 WorldQuant alpha 表达式定义
│   ├── extra_factors.py     # 非 alpha 因子 + IndNeutralize 行业中性化（申万 L3）
│   ├── compute.py           # 主计算流水线：DuckDB → Polars → 并行计算 → factor_values
│   ├── update.py            # 增量因子更新入口
│   └── __init__.py
│
├── strategies/              # 策略与模型
│   ├── base.py              # BaseStrategy 抽象基类 + walk_forward() 滚动框架
│   ├── labels.py            # 前向收益标签 + 退市感知 + 涨跌停 mask
│   ├── evaluation.py        # rank IC, Pearson IC, IC 汇总统计
│   └── lgb.py               # LightGBM 多目标策略（★ 主模型，4 个独立 LGBMRegressor）
│
├── backtest/                # 回测
│   ├── run_lgb.py           # LightGBM 回测：5日调仓 long-only（★ 主用）
│   ├── signals.py           # 组合模拟器，含跳空过滤 + ST 过滤 + 退市处理
│   └── {pool_name}/         # 输出按股票池分子目录
│       ├── equity.csv       # 权益曲线
│       └── benchmark.csv    # 基准权益曲线
│
├── forecast_display/        # 预测展示
│   ├── generate_lgb.py      # LightGBM 预测 HTML（★ 主用），自动过滤 ST/退市股
│   └── html_lgb/{pool_name}/ # LightGBM HTML 报告
│
├── models/                  # 训练好的模型权重
│   └── {pool_name}/         # 按股票池分子目录
│       └── lgb_multi.joblib     # LightGBM 模型（★ 主模型）
│
├── run_lgb.py               # ★ 主训练入口：LightGBM 多周期
├── _check_pkgs.py           # 依赖检查工具
└── .env                     # Tushare API token
```

## Key Architectural Decisions

### 1. DuckDB 单一数据源
所有行情数据和因子值均存储在 `data/ashare.duckdb` 这一个嵌入式数据库中。前复权价格通过 SQL VIEW `daily_kline` 实时计算，原始数据保持不变。**切勿引入其他数据库或文件格式来存储市场数据。**

### 2. LightGBM 多目标架构（★ 主模型）
- **4 个独立 LGBMRegressor**：每 horizon 一个，参数完全不共享，各自训练独立的梯度提升树
- **138 个因子输入**（101 个 Alpha101 + 37 个非 alpha，含 IsST、LnAge、筹码分布因子）
- 每个模型独立使用早停、L1+L2 正则、bagging 防过拟合

### 3. 固定测试集的 Walk-Forward
- 在 2025-06-01 之前的所有数据上一次性训练
- 使用该冻结模型预测整个测试期（2025-06-01 至 2026-06-01，约 242 个交易日）
- 计算高效，且避免前视偏差
- **不是**在扩展窗口上迭代重训练

### 4. 因子集（138个）
涵盖：Alpha101（101个 WorldQuant alpha，基于 vnpy 表达式 DSL，字符串表达式 + Polars DataProxy 延迟计算）、动量（3）、波动率（4）、价格位置/技术（6）、日内形态（2）、成交量/流动性（2）、市值/成交额（3）、换手率（2）、日内（1）、市场状态（5）、横截面排名（3）、个股年龄（1）、ST状态（1）、筹码分布（4）。

关键新增：
- **Alpha101 全量因子**：从 vnpy 端口全部 101 个 WorldQuant alpha 表达式，含 18 个行业中性化（申万 L3 IndNeutralize）+ alpha56 市值因子（total_mv → cap）
- **表达式 DSL 引擎**：字符串表达式 → eval() → DataProxy 链式延迟计算（Polars），横截面 rank 原生正确（cross-sectional by construction）
- `LnMktCap`：对数总市值（Size 因子），`total_mv` from daily_basic
- `Turnover_3d` / `Turnover_3d_ratio`：3日均换手率及其与20日均的比值，换手率由 `volume × close / circ_mv` 实时计算
- `AvgAmount_90d`：90日均成交额
- `Intraday_return`：日内收益 `(close-open)/open`
- `CSI_return_1d/5d/20d`、`CSI_volatility_20d`：中证全指（000985）市场状态特征，横截面广播（同一日期所有股票共享相同值）
- `Return_1d_rank` / `Return_20d_rank` / `Turnover_3d_rank`：对现有因子做横截面排名（同日期所有股票百分位 − 0.5），捕捉相对强弱信号
- `IsST`：当日是否处于 ST/*ST 状态（从 namechange 表解析，0/1 二值因子）
- `LnAge`：上市日至今日的自然对数天数
- `WinnerRate`、`CostPosition`、`ChipDispersion`、`ChipSkew`：筹码分布因子（2018-至今）

### 5. ST/退市处理机制
- **namechange 表**：通过 Tushare `namechange` API 拉取全池股票的名称变更历史
  - `change_reason` 识别：`'ST'` / `'*ST'` / `'撤销ST'` / `'终止上市'`
  - `start_date` / `end_date` 定义状态区间
- **delist_info 表**：从 namechange 中提取 `change_reason='终止上市'` 记录，存储退市日期
  - **不使用名称匹配**（不查 name 含"退"字），避免误匹配正常股票名
- **IsST 因子**：`_merge_st_flag()` 在因子计算后处理中广播，对每个 `(code, date)` 判断是否处于 ST 期间
  - 修复了 NULL end_date 的覆盖问题：自动截断到下一条 namechange 记录之前
- **退市感知 Forward Return**：`compute_forward_returns()` 对退市股在 `delist_date` 之后、forward horizon 跨过最后交易日时，填充 `-1.0`（价值归零）
- **训练排除**：训练集中剔除 IsST=1 和退市后的观测（`run_lgb.py` 中实现）
- **回测过滤**（3 层防御）：
  1. `excluded_codes`（名称快照）：兜底，当前名称含 "ST"/"退"
  2. `isst_map`（`factor_values.IsST`）：主力，每日 ST 状态，来自 namechange 表
  3. `delist_info`（`delist_date`）：排除已退市股票（当前日期 >= delist_date）
  - 适用于 `run_portfolio`、`run_portfolio_rebalance`、`run_long_short`、`run_holding_test`
- **增量更新**：`pull_adj.py` 每次运行时调用 `_incremental_namechange()`，通过 Tushare `namechange` API（不指定 ts_code）拉取全市场近期 namechange 记录，合并到 namechange 表并重新提取 delist_info

### 6. 测试集 IC 过滤
为保证 IC 反映实盘可复现的预测能力，测试集 IC 计算时排除以下观测：
- **涨跌停过滤**：`compute_nextopen_limit_mask()` 检测 T+1 日 open 是否在涨跌停价（±10% 普通股 / ±5% ST 股），如封板则剔除该预测——因为买不到/卖不出
- **ST 过滤**：测试 IC 计算时剔除当日 IsST=1 的所有观测——ST 股流动性差且涨跌停频繁，IC 不可复现
- 过滤后 IC 不降反升（20d: 0.2564 → 0.2600），说明 ST 在稀释信号而非虚增 IC

### 7. 预测 Horizon
- **Horizon**：`[5, 10, 20, 30]` 日（从 `[3,5,10,20]` 改为去掉 3d、加 30d）
- **训练权重**：均分 `{5:0.25, 10:0.25, 20:0.25, 30:0.25}`
- **展示权重**：均分 `{5:0.25, 10:0.25, 20:0.25, 30:0.25}`，按 pred_20d 排名
- 回测建仓用 `pred_20d` 列（保持不变）

### 8. 多股票池配置
- **中心配置**：所有模块通过 `config.py` 获取路径和股票池，不再硬编码
- **默认股票池**：`mainboard_microcap`（~1112只，1e < circ_mv < 20e）
- **股票池文件**：每个池一个 JSON 文件，放在 `pools/` 下
- **切换股票池**：通过环境变量 `QUANTLAB_POOL` 设置，默认 `mainboard_microcap`
  ```bash
  set QUANTLAB_POOL=smallcap_on_mainboard && python run_lgb.py
  ```
- **输出隔离**：模型、预测缓存、回测、HTML 报告均按 `{pool_name}/` 分子目录存储
- **数据拉取**：`build_db.py` / `pull_adj.py` 使用 `load_all_pool_stocks()` 加载所有池的并集，确保数据库覆盖所有股票

### 9. 换手率实时计算
`daily_kline` VIEW 继承的 `turn` 字段为 NULL（Tushare `daily` 接口不返回换手率）。计算因子时通过 `volume × close / NULLIF(circ_mv, 0)` 在 `load_all_stocks` SQL 中实时算得换手率。

### 10. 市值数据来源
总市值 `total_mv` 和流通市值 `circ_mv` 来自 `daily_basic` 表（Tushare `daily_basic` 接口），单位为**万元**。`daily_raw` 中的同名字段全为 NULL（Tushare `daily` 接口不返回市值）。因子计算时通过 LEFT JOIN `daily_basic` 获取，注意 `daily_basic.code` 不含后缀（`.SH`/`.SZ`）。

### 11. 市场状态特征（中证全指）
- 指数日线数据存储在 `index_daily` 表（通过 `data/build_index_db.py` 拉取）
- 当前使用 **中证全指（000985）**，从 2008 年起全程覆盖
- 因子计算流水线（`compute_panel`）会自动从 `index_daily` 提取指数数据，计算 `CSI_return_1d/5d/20d` 和 `CSI_volatility_20d`，然后横截面广播到每个股票-日期行
- 增量计算（`compute_panel_incremental`）同样会自动合并市场特征
- 如需切换指数，修改 `compute.py` 中 `compute_market_features()` 的 WHERE 条件

### 12. Alpha 因子横截面排名
- 101 个 alpha 因子中的 `cs_rank()` 调用**原生为横截面排名**（`groupby('datetime').rank()`），因为表达式求值引擎在 DataProxy 上执行，横截面算子 `cs_rank` 天然按日期分组，无需手动后处理
- **实现方式**：`calculate_by_expression()` 将各列包装为 DataProxy，表达式中的 `cs_rank()` → 调用 `cs_function.cs_rank()` → `pl.col('data').rank().over('datetime')`
- 横截面排名因子 `Return_1d_rank`、`Return_20d_rank`、`Turnover_3d_rank` 在 `extra_factors.py` 中通过 `rank().over('datetime')` 生成
- 极端值保护：表达式引擎内 DataProxy 通过 `fill_nan(null)` + `is_infinite → null` 自动处理溢出值

### 13. 数据库表清单
| 表 / VIEW | 来源 | 说明 |
|-----------|------|------|
| `stock_info` | `stock_basic(list_status='L')` | 当前上市股票信息（code, name, market, list_date） |
| `daily_raw` | `daily` + `adj_factor` | 原始日线 OHLCV + 复权因子（2008-至今） |
| `daily_basic` | `daily_basic` | 市值/估值指标（total_mv, circ_mv, PE, PB 等） |
| `daily_kline` | VIEW → daily_raw + latest_adj | 前复权 OHLCV（实时计算） |
| `factor_values` | `compute.py` | 因子宽表（code, date, 138 因子列） |
| `cyq_perf` | `cyq_perf` | 筹码分布（his_low/high, cost_*, winner_rate, 2018-至今） |
| `industry` | `build_industry.py` | 行业分类（申万 SW2021 L1/L2/L3，含 Tushare 行业） |
| `index_daily` | `index_daily` | 指数日线（000985 中证全指, 000300 沪深300） |
| `namechange` | `namechange` | 股票名称变更历史（ST/*ST/终止上市/改名） |
| `delist_info` | 从 namechange 提取 | 退市日期（code, delist_date） |

## Common Workflows

所有命令默认使用 `QUANTLAB_POOL` 环境变量指定的股票池（默认 `smallcap_on_mainboard`）。切换方式：
```bash
set QUANTLAB_POOL=mainboard_microcap && python run_lgb.py
```

### 更新数据（每日运行）
```bash
python data/pull_adj.py      # 拉取最新日线行情（含指数、cyq_perf、namechange 增量）
python -m factors.update     # 增量计算因子
```

### 拉取指数数据（首次/补充）
```bash
python data/build_index_db.py   # 拉取指数日线（中证全指等）入库
```

### 拉取筹码分布数据（首次/补充）
```bash
python data/build_cyq.py        # 全量拉取当前池的 cyq_perf（2018-至今）
python data/build_cyq.py --incr # 增量拉取最近缺失交易日
```

### 拉取退市/ST 数据（首次/补充）
```bash
python data/build_delist_info.py   # 拉取全池 namechange → delist_info + IsST 历史
```

### 训练模型
```bash
python run_lgb.py      # ★ 训练 LightGBM（主模型），打印 IC 统计
```

### 运行完整回测
```bash
python -m backtest.run_lgb   # LightGBM 回测：5日调仓 long-only（★ 主用）
```

### 生成预测报告
```bash
python forecast_display/generate_lgb.py   # 输出 LightGBM HTML 到 forecast_display/html_lgb/
```

### 导出交易信号
```bash
python trade_signals/export.py
```

### 运行测试
```bash
# No tests configured yet
```

### 检查依赖
```bash
python _check_pkgs.py
```

## Important Constraints

- **不要启动 `build_db.py` 全量构建**：该脚本拉取 2008-2026 全年全市场 K 线、复权因子和估值指标，按日拉取约 4600 个交易日 × 3 次 API ≈ 13800 次 API 调用，预计耗时 **2-4 小时**。频繁重跑不仅浪费时间，还会加重中转站负担。**除非用户明确要求，否则不要启动 `build_db.py`。**
- **不要提交 DuckDB 文件**：`.gitignore` 已排除 `*.duckdb`，数据库文件较大且包含敏感配置
- **不要提交 .env 文件**：包含 Tushare token
- **Tushare 中继限流**：quicksync 中继稳定速率 200次/分钟，上限 600次/分钟。`build_db.py` 按日拉取全市场数据，每交易日 3 次 API（daily + adj_factor + daily_basic），无主动 sleep，由 relay 响应天然限速（实际 ~80-160 次/分钟）。修改数据拉取代码时注意保持此限制
- **无 notebook**：本项目不使用 Jupyter notebook，所有分析均通过 Python 脚本完成
- **无正式依赖文件**：项目没有 requirements.txt 或 pyproject.toml。所需包见 `_check_pkgs.py`
- **仅支持 A 股主板**：股票池定义在 `pools/` 目录下的 JSON 文件中，聚焦流通市值 1-28 亿的主板股票

## Coding Conventions

- 类型标注按需使用（非强制）
- 遵循各模块已有的代码风格
- 新因子添加到 `factors/alpha101.py` 或 `factors/extra_factors.py`
- 新策略继承 `strategies/base.py` 中的 `BaseStrategy`
- 路径优先使用绝对路径或基于 `__file__` 的相对路径

## 八荣八耻
以瞎猜接口为耻，以认真查询为荣；
以模糊执行为耻，以寻求确认为荣；
以臆想业务为耻，以人类确认为荣；
以创造接口为耻，以复用现有为荣；
以跳过验证为耻，以主动测试为荣；
以破坏架构为耻，以遵循规范为荣；
以假装理解为耻，以诚实无知为荣；
以盲目修改为耻，以谨慎重构为荣。