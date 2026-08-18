# 业界实践与方法论范式

> 调研日期：2026-08-18。来源为公开报道与平台资料；涉及私募内部做法的均为传闻级信息，已标注。

## 1. WorldQuant BRAIN：众包 + 平台化的祖师爷

[WorldQuant BRAIN](https://worldquantbrain.com/zh-hans/consultant) 及其竞赛体系（[Global Alphathon](https://www.worldquant.com/zh-hans/global-alphathon/)、IQC）是公式化 alpha 范式的最大规模实践：
- 研究者在其在线平台用**预定义算子 + 数据字段**写表达式，平台即时回测打分（Sharpe、turnover、fitness 等）。
- 平台强制**自相关性检查**（与已有 alpha 池的相关性超过阈值不收录），本质就是"因子池正交化"的工业化——与 AlphaGen 的"协同集合"目标殊途同归。
- 国内知乎有[因子挖掘框架思路讨论](https://zhuanlan.zhihu.com/p/2020460152789156679)（半年滚动自相关周期等经验）。

**对 quantlab 的启示**：BRAIN 的核心经验是"**因子池的边际贡献比单因子质量重要**"。quantlab 的 `select_factors.py`（IC 排序 + corr>0.75 去冗余）是这个思想的简化版。

## 2. 国内 LLM 因子挖掘的公开实践

- **国产大模型横向评测**：新浪财经报道过六个国产大模型在 AlphaMind 平台进行标准化因子优化竞赛（20 轮迭代、每轮 4 个变体），验证"LLM 迭代优化因子表达式"路线在国内数据上的可行性。报道细节我未逐条核实。
- **QuantaAlpha**（发现报告，[链接](https://www.fxbaogao.com/detail/5339368)）：描述了以 LLM 为核心的"提出假设 → 构建因子 → 回测检验 → 迭代优化 → 因子池维护"一体化体系，与学术侧 Alpha-GPT 循环一致。
- 国内头部私募（九坤、宽德等）使用 LLM 的传闻较多，但**无公开可验证的技术细节**，此处不作引用。

## 3. 四种工程范式（综合学术+业界）

### 范式 A：LLM 一次性头脑风暴（假设生成器）
LLM 无反馈，直接大量产出因子假设（表达式 + 经济直觉），人工/流水线筛选。
- 成本最低；本次 quantlab 实验即此范式。
- 已知缺陷：**同质化严重**（300 个里预计 40%+ 是彼此的变体）、无市场数据 grounding。

### 范式 B：回测反馈迭代（Alpha-GPT 循环）
LLM 生成 → 回测 → 把 IC/相关性等定量反馈喂回 prompt → 修改再生成。
- 对应工程：agent loop，每轮把 top/bottom 表现因子清单写进上下文。
- 关键经验：反馈要包含**因子间相关性矩阵**（否则 LLM 反复产高相关变体）。

### 范式 C：LLM 提议 + 系统性搜索（MCTS/进化）
LLM 做语义合理的提议与变异算子，搜索框架保证覆盖度与多样性（频繁子树规避、表达式聚类去重）。

### 范式 D：LLM 消化另类数据（情绪/事件因子）
LLM 读新闻、公告、研报生成情绪/事件特征（BloombergGPT、FinGPT 一脉）。与本调研的主线（公式化因子）正交，且 quantlab 目前无文本数据源，**不在本次范围**。

## 4. 共同的工程红线（所有范式都强调）

1. **多重检验校正**：批量生成的因子必然有样本内幸运儿。必须时间外样本（walk-forward）复核，必要时按尝试次数惩罚显著性（如 Bonferroni/White reality check 思想）。
2. **同质化控制**：生成阶段做表达式结构聚类（token 序列相似度），筛选阶段做相关性去冗余（quantlab 已有 corr 0.75 阈值）。
3. **过拟合与容量**：因子库越大，模型（LightGBM）越容易把噪声学进去；入模数量应有先验上限（quantlab 当前 30/155 是合理比例）。
4. **经济直觉必要性**：LLM 生成的"故事"不等于因果；保留直觉描述是为了让人能证伪，不是为了说服自己。
5. **数据边界即能力边界**：LLM 不知道微盘池的实证规律（学术全部基于主流市场），它提供的是"结构合理的假设空间"，**有效性完全由本地回测裁决**。

## 5. 对 quantlab 微盘池的适配评估

- 微盘池（主板流通市值 1-20 亿，~1112 只）的已知结构性特征：涨跌停频繁、散户主导、流动性敏感、壳价值残留、风格切换时集体回撤（如年初微盘踩踏）。
- 学术文献**没有**针对此类池子的 FAFM 结论；因此本次 300 因子库的定位是"假设库"，每个因子都标注了依据强度（文献/实践/推测）与数据可得性，走既有 IC 筛选流水线裁决，**不预设任何因子有效**。

## 参考

- [WorldQuant BRAIN 顾问计划](https://worldquantbrain.com/zh-hans/consultant) / [Global Alphathon](https://www.worldquant.com/zh-hans/global-alphathon/)
- [BRAIN 因子挖掘框架（知乎）](https://zhuanlan.zhihu.com/p/2020460152789156679)
- [QuantaAlpha 报告（发现报告）](https://www.fxbaogao.com/detail/5339368)
- 学术侧详见 `01_survey_academic.md`
