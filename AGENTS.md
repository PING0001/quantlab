# Quantlab — AI Agent Instructions

## Project Overview

Quantlab 是一个 **A股量化选股系统**，针对主板微盘股（流通市值 1-28 亿）进行多周期收益预测。核心流程：

```
Tushare 数据 → DuckDB 存储 → 因子计算（60个） → 多周期 MLP 训练（45因子输入） → 回测 → HTML 预测报告 → JSON 交易信号
```

## Technology Stack

| 层 | 技术 |
|------|-----------|
| 数据源 | Tushare（通过 quicksync.cn 中继） |
| 数据库 | **DuckDB**（嵌入式 OLAP，所有数据单一来源） |
| 数值计算 | numpy, pandas, scipy |
| 机器学习 | **PyTorch**（多目标 MLP）、scikit-learn（标准化） |
| 序列化 | joblib, parquet |
| 配置 | python-dotenv（.env 中的 Tushare token）、config.py（中心配置） |

## Project Structure

```
quantlab/
├── config.py                # ★ 中心配置：DB 路径、股票池加载、各模块输出路径
├── pools/                   # 股票池定义（JSON，多池支持）
│   └── smallcap_on_mainboard.json  # ~2941只主板小市值股票（1e < circ_mv < 40e）
│
├── data/                    # 数据摄入
│   ├── build_db.py          # 全量建库：按日拉全市场日线、复权因子、市值/估值（2008-2026）
│   ├── build_index_db.py    # 全量建库：拉取指数日线（中证全指 000985 等）
│   ├── build_cyq.py         # 全量/增量拉取筹码分布数据（cyq_perf, 2018-至今）
│   ├── pull_adj.py          # 增量更新：获取上次记录日期之后的新数据（含指数、cyq_perf）
│   └── ashare.duckdb        # DuckDB 数据库（~650 MB），所有数据唯一来源
│
├── factors/                 # 因子工程
│   ├── ops.py               # 底层算子：rank, ts_sum, ts_rank, correlation 等
│   ├── factors.py           # 60个因子函数 + FACTOR_HUB 注册表
│   ├── compute.py           # 因子计算流水线：读取 K 线 → 全量或增量计算 → 存储
│   ├── update.py            # 增量因子更新入口
│   └── selection.py         # 因子相关性分析 + 贪心多样化选择（可选）
│
├── strategies/              # 策略与模型
│   ├── base.py              # BaseStrategy 抽象基类 + walk_forward() 滚动框架
│   ├── labels.py            # 前向收益标签计算（任意周期）
│   ├── evaluation.py        # rank IC, Pearson IC, IC 汇总统计
│   ├── mlp.py               # PyTorch 多目标 MLP 实现（MLPStrategy）
│   └── test_synthetic.py    # 合成数据冒烟测试
│
├── backtest/                # 回测
│   ├── run.py               # 完整回测：训练 MLP + 多空组合模拟
│   ├── signals.py           # 多空组合模拟器，含跳空过滤器
│   └── {pool_name}/         # 输出按股票池分子目录
│       ├── equity.csv       # 权益曲线
│       └── benchmark.csv    # 基准权益曲线
│
├── forecast_display/        # 预测展示
│   ├── generate.py          # 加载模型 → 预测最新日期 → 生成交互式 HTML 报告
│   └── html/{pool_name}/    # 输出的 HTML 预测报告，按股票池分子目录
│
├── trade_signals/           # 交易信号导出
│   ├── export.py            # JSON 信号导出器
│   ├── schema.py            # JSON schema 定义
│   └── output/              # 输出的信号 JSON
│
├── models/                  # 训练好的模型权重
│   └── {pool_name}/         # 按股票池分子目录
│       └── mlp_multihead.pt # 多目标 MLP（主模型）
│
├── run_mlp_multi.py         # ★ 主训练入口
├── _check_pkgs.py           # 依赖检查工具
└── .env                     # Tushare API token
```

## Key Architectural Decisions

### 1. DuckDB 单一数据源
所有行情数据和因子值均存储在 `data/ashare.duckdb` 这一个嵌入式数据库中。前复权价格通过 SQL VIEW `daily_kline` 实时计算，原始数据保持不变。**切勿引入其他数据库或文件格式来存储市场数据。**

### 2. 多目标 MLP 架构
- **共享主干网络**：32→16→8 隐藏层，从45个精选因子中提取共同特征
- **4个独立线性输出头**：分别预测 1、3、5、10 日收益
- 带 L2 正则化、dropout、早停和学习率调度防止过拟合
- 每个周期的目标收益在训练前独立进行 z-score 标准化，预测时再反标准化

### 3. 固定测试集的 Walk-Forward
- 在 2024-06-01 之前的所有数据上一次性训练（约 3959 个交易日）
- 使用该冻结模型预测整个测试期（2024-06-01 至 2026-06-26，约 499 个交易日）
- 计算高效，且避免前视偏差
- **不是**在扩展窗口上迭代重训练

### 4. 精选因子集（45个）
涵盖：动量（4）、波动率（5）、价格位置/其他（6）、日内形态（4）、成交量/流动性（2）、WorldQuant alpha 复合（13）、市值/成交额（2）、换手率（2）、市场状态（4）、横截面排名（3）。

关键新增：
- `LnMktCap`：对数总市值（Size 因子），`total_mv` from daily_basic
- `Turnover_3d` / `Turnover_3d_ratio`：3日均换手率及其与20日均的比值，换手率由 `volume × close / circ_mv` 实时计算
- `AvgAmount_90d`：90日均成交额
- `Intraday_return`：日内收益 `(close-open)/open`
- `CSI_return_1d/5d/20d`、`CSI_volatility_20d`：中证全指（000985）市场状态特征，横截面广播（同一日期所有股票共享相同值），帮助 MLP 感知大盘环境
- `Return_1d_rank` / `Return_20d_rank` / `Turnover_3d_rank`：对现有因子做横截面排名（同日期所有股票百分位 − 0.5），捕捉相对强弱信号

### 5. 组合模拟中的跳空过滤器
仅在次日开盘时建仓，前提是股票未出现向上跳空（多头）或向下跳空（空头）超过 1.5% 的情况。

### 6. 多股票池配置
- **中心配置**：所有模块通过 `config.py` 获取路径和股票池，不再硬编码
- **股票池文件**：每个池一个 JSON 文件，放在 `pools/` 下
- **切换股票池**：通过环境变量 `QUANTLAB_POOL` 设置，默认 `smallcap_on_mainboard`
  ```bash
  set QUANTLAB_POOL=mainboard_smallcap && python run_mlp_multi.py
  ```
- **输出隔离**：模型、预测缓存、回测、HTML 报告均按 `{pool_name}/` 分子目录存储
- **数据拉取**：`build_db.py` / `pull_adj.py` 使用 `load_all_pool_stocks()` 加载所有池的并集，确保数据库覆盖所有股票

### 7. 换手率实时计算
`daily_kline` VIEW 继承的 `turn` 字段为 NULL（Tushare `daily` 接口不返回换手率）。计算因子时通过 `volume × close / NULLIF(circ_mv, 0)` 在 `load_all_stocks` SQL 中实时算得换手率。

### 8. 市值数据来源
总市值 `total_mv` 和流通市值 `circ_mv` 来自 `daily_basic` 表（Tushare `daily_basic` 接口），单位为**万元**。`daily_raw` 中的同名字段全为 NULL（Tushare `daily` 接口不返回市值）。因子计算时通过 LEFT JOIN `daily_basic` 获取，注意 `daily_basic.code` 不含后缀（`.SH`/`.SZ`）。

### 9. 市场状态特征（中证全指）
- 指数日线数据存储在 `index_daily` 表（通过 `data/build_index_db.py` 拉取）
- 当前使用 **中证全指（000985）**，从 2008 年起全程覆盖
- 因子计算流水线（`compute_panel`）会自动从 `index_daily` 提取指数数据，计算 `CSI_return_1d/5d/20d` 和 `CSI_volatility_20d`，然后横截面广播到每个股票-日期行
- 增量计算（`compute_panel_incremental`）同样会自动合并市场特征
- 如需切换指数，修改 `compute.py` 中 `compute_market_features()` 的 WHERE 条件

### 10. Alpha 因子横截面排名
- 13 个入选 alpha 因子中的 `rank()` 调用已从**时序排名**（同股票历史上排）修正为**横截面排名**（同日期全市场排），还原 WorldQuant 原始公式语义
- **实现方式**：因子计算分两阶段：
  1. 预计算阶段 `_compute_cs_rank_cols()`：在全量 DataFrame 上对 close、volume、low 及派生字段（dc1、dv1、ret1d）做 `groupby('date').transform(cs_rank)`，生成 6 个 `_cs` 后缀列供因子函数引用
  2. 后处理阶段 `_apply_cs_rank_post()`：对 8 个需要外层横截面 rank 的 alpha 因子在全量 panel 上重算 rank
- 新增横截面排名因子 `Return_1d_rank`、`Return_20d_rank`、`Turnover_3d_rank` 通过 `_merge_rank_factors()` 同样在后处理阶段生成
- 极端值保护：先 `clip(-1e10, 1e10)` 夹住溢出值，再 `replace(inf→NaN)` 兜底，确保 cs_rank 不会遇到不可计算的值

## Common Workflows

所有命令默认使用 `QUANTLAB_POOL` 环境变量指定的股票池（默认 `smallcap_on_mainboard`）。切换方式：
```bash
set QUANTLAB_POOL=mainboard_smallcap && python run_mlp_multi.py
```

### 更新数据（每日运行）
```bash
python data/pull_adj.py      # 拉取最新日线行情（含指数、cyq_perf）
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

### 训练模型
```bash
python run_mlp_multi.py      # 训练多周期 MLP，打印 IC 统计
```

### 运行完整回测
```bash
python -m backtest.run       # 训练 + 多空组合模拟
```

### 生成预测报告
```bash
python forecast_display/generate.py   # 输出 HTML 到 forecast_display/html/
```

### 导出交易信号
```bash
python trade_signals/export.py
```

### 运行测试
```bash
python strategies/test_synthetic.py   # 合成数据冒烟测试
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
- 新因子添加到 `factors/factors.py` 并注册到 `FACTOR_HUB`
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