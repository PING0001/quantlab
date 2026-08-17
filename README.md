# Quantlab


一个小量化系统，专注于A股微盘。

**多因子 + LightGBM 分类**：因子库 155 个因子，模型选用其中 30 个。

```
Tushare 数据 → DuckDB 存储 → 因子计算 → LightGBM 分类训练 → 回测 → HTML 预测报告
```

## 技术栈

| 层 | 技术 |
|---|---|
| 数据源 | Tushare |
| 数据库 | DuckDB（嵌入式，所有行情/因子数据单一来源） |
| 数值计算 | numpy / pandas / scipy / polars |
| 机器学习 | LightGBM（单 LGBMClassifier，三分类 + 期望收益打分） |
| 配置 | python-dotenv（`.env` 中的 Tushare token）+ `config.py` |

## 快速开始

### 1. 环境准备

进入项目根目录（以下所有命令均在此目录下执行）：

```bash
cd quantlab
```

建议使用 venv 虚拟环境隔离依赖（需 Python >= 3.9）：

```bash
python -m venv .venv
```

激活环境后安装依赖：

| 环境 | 激活命令 |
|---|---|
| Windows cmd | `.venv\Scripts\activate` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Linux / macOS | `source .venv/bin/activate` |

```bash
pip install -r requirements.txt
```

可用 [`_check_pkgs.py`](_check_pkgs.py) 自检已安装版本。

在项目根目录创建 `.env`，写入 Tushare token（[官网](https://tushare.pro)注册获取）：

```
TUSHARE_TOKEN=你的token
```

> 注：数据拉取的 API 中转地址统一在 `data/_ts.py` 设置，请将其改为官方地址 `https://api.tushare.pro`（或直接删除该行）。以下脚本仍各自硬编码了中转地址，需单独修改：
> - `data/build_index_db.py`
> - `data/build_cyq.py`
> - `data/build_delist_info.py`
> - `data/build_industry.py`

### 2. 首次建库（耗时较长，仅需一次）

```bash
python -m data.pull --full   # 全市场日线+复权+估值+筹码+指数+SHIBOR+日历（2008 至今，约 2-4 小时）
python data/build_cyq.py     # （可选）筹码分布按股补全（2018 至今）
```

> ⚠️ 全量拉取约 1.4 万次 API 调用，请确认有必要再跑，并注意 Tushare 接口限频。

### 3. 每日更新

```bash
python -m data.pull          # 统一增量拉取（滚动重拉近5日幂等 + 缺口 pending + 当日 cyq 拉空自动延迟重试至 21:00）
python -m factors.update     # 增量计算因子（日期集合对账，历史空洞自动回补；末尾输出完整性报告）
```

其他常用命令：

```bash
python -m data.pull --dry-run     # 只打印各数据源的目标日期
python -m data.pull --reconcile   # 深对账：全历史 vs 交易日历，缺口入 pending 并补拉（建议月度）
python -m factors.integrity       # 独立运行完整性校验（硬失败 exit 1）
```

### 4. 训练 / 回测 / 报告

```bash
python run_lgb.py                          # 训练 LightGBM，打印 IC 统计
python -m backtest.run_lgb                 # 5 日调仓 long-only 回测
python forecast_display/generate_lgb.py    # 生成 HTML 预测报告
python trade_signals/export.py             # 导出交易信号
```

## 股票池

股票池定义在 `pools/mainboard_microcap.json`：~1112 只主板微盘股（流通市值 1-20 亿，半年度筛选更新）。系统默认使用该池，无需额外配置；模型、预测缓存、回测结果、HTML 报告均输出到对应的 `mainboard_microcap/` 子目录。

## 输出信号
模型对每只股票每日输出一个**看多打分**：未来 16~20 个交易日中位数收益**涨超 8% 的概率**减去**跌超 4% 的概率**。打分越高，模型越看好；中性值在 0.015 左右。

HTML 预测报告（`forecast_display/html_lgb/mainboard_microcap/`）用HTML的形式展示某日预测结果

## 目录结构
```
quantlab/
├── config.py          # 中心配置：DB 路径、股票池、各模块输出路径
├── pools/             # 股票池定义（JSON）
├── data/              # 数据摄入 + DuckDB 数据库（pull 统一入口 / sources 数据源 / calendar 日历 / integrity 校验）
├── factors/           # 因子工程：表达式 DSL 引擎 + Alpha101 + 附加因子
├── strategies/        # 策略与模型：LightGBM 分类、标签、IC 评估
├── backtest/          # 回测（按股票池分子目录输出 equity/benchmark）
├── forecast_display/  # 预测 HTML 报告
├── models/            # 训练好的模型权重（joblib）
├── trade_signals/     # 交易信号导出
└── run_lgb.py         # ★ 主训练入口
```

## 说明

- 前复权价格由 DuckDB VIEW `daily_kline` 实时计算，原始数据不变。
- ST / 退市处理：训练集剔除 ST 与退市后观测；回测三层过滤（名称快照、IsST 因子、退市日期）。
- 测试集 IC 已排除涨跌停封板与 ST 股观测，贴近实盘可复现性。
- 数据库文件（`*.duckdb`）与 `.env` 不入库（已在 `.gitignore` 排除）。

## 免责声明

本项目仅供学习研究，不构成任何投资建议。
