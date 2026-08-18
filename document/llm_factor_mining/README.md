# LLM 因子挖掘调研与微盘池 300 因子假设库

> 生成日期：2026-08-18 | 生成方式：LLM 头脑风暴范式（调研报告范式 A 的实践）
> 状态：**全部为未验证假设**，落地需走 `factors/select_factors.py` IC 筛选流水线

## 文件清单

| 文件 | 内容 | 字数/条目 |
|---|---|---|
| `00_summary.md` | 400 字供人总结 | — |
| `01_survey_academic.md` | 学术前沿：AlphaGen→Alpha-GPT→LLM+MCTS→CogAlpha→AlphaBench | ~2.5k 字 |
| `02_survey_industry.md` | 业界实践（WorldQuant/国产评测）与四种工程范式 | ~2k 字 |
| `factors_01_momentum_reversal.md` | A：动量与反转 | MO01–MO30 |
| `factors_02_volatility_moments.md` | B：波动率与高阶矩 | VO01–VO30 |
| `factors_03_liquidity_volume.md` | C：流动性与成交量 | LQ01–LQ30 |
| `factors_04_chip_distribution.md` | D：筹码分布（本池特色数据） | CH01–CH40 |
| `factors_05_microcap_structure.md` | E：微盘池结构性 | MC01–MC30 |
| `factors_06_limit_microstructure.md` | F：涨跌停与微观结构 | LM01–LM30 |
| `factors_07_intraday_overnight.md` | G：隔夜与日内分解 | IO01–IO25 |
| `factors_08_market_regime_interaction.md` | H：市场状态与横截面交互 | MR01–MR35 |
| `factors_09_calendar_event.md` | I：日历与事件 | CE01–CE25 |
| `factors_10_nonlinear_hybrid.md` | J：非线性变换与混合 | NL01–NL25 |

**合计 300 条因子定义**（每条含：DSL 风格表达式 / 经济直觉 / 依据强度 / 预期方向 / 数据可得性）。

## 依据标注体系

- `[W#]` 经典文献（Jegadeesh-Titman 1993、Lehmann 1990、Ang et al. 2006、Bali et al. 2011、Amihud 2002、George-Hwang 2004、Datar et al. 1998 等，在各文件头部编号）
- `[实践]` 业界常用、无可靠学术出处
- `[推测]` 本文档假说（最弱，约占四成）
- `[A101≈]` 与 Alpha101 结构近似（需去重）
- `[现有对照]` 与现有因子重复，仅作对照（实现可跳过）

## 数据可得性三档

1. **现有数据库可直接实现**：约 285 条（daily_kline / daily_basic / cyq_perf / index_daily / macro_daily / industry / stock_info）
2. **需新增 `stk_limit`（涨跌停价，约 2000 积分）**：LM 系列精确版约 8 条——接入只需在 `data/sources.py` 注册一个新 source
3. **需新增财报披露/分红数据**：CE 系列约 7 条（成本较高，暂缓）

## 落地路径（建议顺序）

1. 先跑零成本的：第一档因子中标注 `[W#]` 依据强的（VO01 特质波动率、VO04 MAX、LQ 系列 Amihud 变体）进入 `select_factors.py` 筛选
2. 结构去重：对 300 条先做表达式 token 相似度聚类（LLM+MCTS 论文的"频繁子树规避"思想的简化版），再进相关性去冗余（现有 corr 0.75 阈值）
3. 多重检验纪律：分批入筛选（每批 ≤20），时间外样本复核，警惕批量生成的样本内幸运儿
4. 超越头脑风暴：把每轮筛选的 IC 结果与因子相关性矩阵回填 prompt，迭代生成（调研报告范式 B）

## 已知局限（诚实声明）

- 全部因子未经验证；约四成是同质变体
- 文献结论基于主流市场，微盘池无现成证据
- LLM（本文档作者）不具备本池实证规律的知识，本文档提供的是结构合理的假设空间
