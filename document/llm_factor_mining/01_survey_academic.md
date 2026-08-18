# 学术前沿：用大语言模型挖因子（FAFM）调研

> 调研日期：2026-08-18。范围：LLM/RL/搜索式自动化因子挖掘（Formulaic Alpha Factor Mining, FAFM）的代表性学术工作。所有条目均给出可查证链接；未读全文仅凭摘要的部分已标注。

## 1. 问题定义

公式化因子挖掘（FAFM）：在给定算子库（`+ - * /`、`ts_rank`、`ts_corr` 等）与数据字段（open/high/low/close/volume 等）构成的**表达式空间**中，自动搜索对资产收益有预测力的因子公式。该空间组合爆炸，人工穷举不可行。WorldQuant Alpha101 是该范式的人工经典；自动化方法经历了三代：遗传编程（GP）→ 强化学习（RL）→ 大语言模型（LLM）驱动。

## 2. 代表性工作脉络（按时间）

### 2.1 AlphaGen（RL 路线的里程碑）
**"Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning"**，KDD 2023，北京大学 ICT-FinD-Lab。
- 核心思想：不优化单因子 IC，而是**直接以因子集合组合后的预测表现为奖励**（合作式 RL，COMA 多智能体架构），让新挖出的因子与已有池互补而非冗余。
- 因子表示为 token 序列（表达式），策略梯度生成。
- 开源：[GitHub - ICT-FinD-Lab/alphagen](https://github.com/ICT-FinD-Lab/alphagen)；论文：[arXiv 2306.12964](https://arxiv.org/abs/2306.12964) / [ACM DL](https://dl.acm.org/doi/10.1145/3580305.3599831)。
- 后续：[arXiv 2401.02710](https://arxiv.org/abs/2401.02710)（用预训练 alpha 集扩展搜索空间）、AlphaQCM（分布式 RL）。

**对本项目的启示**：`select_factors.py` 目前按"单因子 IC 排序 + 相关性去冗余"逐个筛，本质是贪心；AlphaGen 证明"组合层面协同"这个优化目标更优。即使不上 RL，也可以在筛选时用"加入后组合 IC 增量"替代"单因子 IC"。

### 2.2 Alpha-GPT（LLM 路线的起点）
**"Alpha-GPT: Human-AI Interactive Alpha Mining for Quantitative Investment"**，[arXiv 2308.00016](https://arxiv.org/abs/2308.00016)。
- 范式：把 alpha 挖掘拆成"人类研究员提出想法 ↔ LLM 理解并翻译成因子表达式 ↔ 回测反馈 ↔ 迭代精炼"的**人机交互循环**，核心是一套 prompt 工程框架（alpha 描述 → 表达式的双向翻译）。
- 续作 Alpha-GPT 2.0（[arXiv 2402.09746](https://arxiv.org/abs/2402.09746)）把 human-in-the-loop 系统化。
- 社区开源复现：[alpha-gpt (LangGraph+Zipline)](https://github.com/parthmodi152/alpha-gpt)。

**对本项目的启示**：quantlab 已有 vnpy 表达式 DSL（`factors/utility.py` 的 `calculate_by_expression`），Alpha-GPT 范式可以**零基础设施成本**落地——LLM 只需输出 DSL 字符串即可直接执行回测，这是本项目最顺的接入点。

### 2.3 LLM + MCTS（搜索增强，AAAI 2026）
**"LLM-Powered Monte Carlo Tree Search for Formulaic Alpha Mining"**，[arXiv 2505.11122](https://arxiv.org/abs/2505.11122)（AAAI 2026 收录，作者与清华关联——此为摘要页推断，未读全文确认）。
- LLM 负责在 MCTS 节点扩展时**生成与精炼符号表达式**；每个候选因子经真实回测产生定量反馈，引导 MCTS 的选择/扩展。
- 引入"频繁子树规避"机制防止生成公式同质化。
- 实验称在预测精度与交易表现上优于 GP/RL 基线（基线细节见正文，此处仅凭摘要）。

**对本项目的启示**：纯 LLM 一次性生成 300 个因子会有大量同质变体（本次实验已预期到这一点）；该论文的"子树规避"对应到工程上就是**生成后做表达式结构聚类去重**，可以在 `select_factors.py` 的相关性去冗余之前先做。

### 2.4 CogAlpha（代码级表示，ACL 2026）
**"Cognitive Alpha Mining: LLM-Driven Code-Based Framework"**，[ACL 2026](https://aclanthology.org/2026.acl-long.538/)。
- 用**代码**（而非受限 DSL 表达式）作为 alpha 表示，LLM 推理 + 进化搜索。代码表示的表达力远超算子 DSL（可以写回归、条件逻辑），代价是不可解释性与过拟合风险上升。

### 2.5 AlphaBench（基准）
**"Benchmarking Large Language Models in Formulaic Alpha Mining"**，[OpenReview](https://openreview.net/forum?id=d97Q8r7ZKZ)（PDF 见 [CityU 页面](https://www.cs.cityu.edu.hk/~cliu6444/HomePage/doc/AlphaBench/AlphaBench_PDF.pdf)）。为 FAFM 建立标准化评测。（OpenReview 页面被浏览器验证拦截，细节以 PDF 为准，未逐页核实。）

### 2.6 其他
- Hybrid LLM 范式综述性工作：[Frontiers of Computer Science 2026](https://link.springer.com/article/10.1007/s11704-025-41061-5)。
- 相关背景：BloombergGPT（2023，金融基础模型）、FinGPT（2023）——非因子挖掘本身，但为领域微调提供底座；本文不展开。

## 3. 方法论谱系总结

| 路线 | 搜索空间引导者 | 反馈信号 | 优势 | 弱点 |
|---|---|---|---|---|
| 遗传编程 GP（传统） | 变异/交叉算子 | 单因子 IC | 无需训练 | 表达式膨胀、同质化 |
| 强化学习（AlphaGen） | 策略网络 | **组合表现** | 因子间协同 | 奖励稀疏、算子空间需预定义 |
| LLM 一次性生成（Alpha-GPT 基础形态） | 语言先验 | 人审/回测 | 零基建、语义丰富 | 同质化、无系统性搜索 |
| LLM + 搜索（MCTS/进化） | LLM 提议 + 搜索剪枝 | 回测定量反馈 | 兼顾语义与系统性 | 工程复杂度高 |
| LLM 代码生成（CogAlpha） | LLM 写代码 | 回测 | 表达力最强 | 过拟合与不可解释风险 |

**共识性结论（多篇工作一致）**：LLM 的价值不在"替代回测找规律"（它并没有从数据里学到市场规律），而在 (a) 把人类假设快速翻译成表达式、(b) 在搜索空间中做语义合理的提议、(c) 对挖出的因子给出可解释的叙述。**预测力必须由回测反馈闭环提供**。

## 4. 与 quantlab 的对接评估

| 论文机制 | quantlab 现状 | 落地成本 |
|---|---|---|
| 表达式因子表示 | 已有 vnpy DSL + 155 因子 | 零成本，天然匹配 |
| 回测反馈闭环 | `select_factors.py` IC 筛选 + 相关性去冗余 | 低：把"单因子 IC"换成"组合增量"即 AlphaGen 精神 |
| 人机交互循环 | 本次调研本身即一次实践（LLM 产 300 因子假设，人审后入筛选） | 零成本 |
| MCTS/进化 | 无 | 高，暂不建议 |
| 微盘池特殊性 | 学术工作全部基于主流市场全样本，**无微盘池结论可借鉴** | 需自证 |

## 5. 参考

1. Wang et al., *Generating Synergistic Formulaic Alpha Collections via RL*, KDD 2023. [arXiv](https://arxiv.org/abs/2306.12964) | [code](https://github.com/ICT-FinD-Lab/alphagen)
2. *Alpha-GPT: Human-AI Interactive Alpha Mining*. [arXiv 2308.00016](https://arxiv.org/abs/2308.00016)；2.0: [arXiv 2402.09746](https://arxiv.org/abs/2402.09746)
3. *LLM-Powered MCTS for Formulaic Alpha Mining*, AAAI 2026. [arXiv 2505.11122](https://arxiv.org/abs/2505.11122)
4. *CogAlpha: Cognitive Alpha Mining*, ACL 2026. [ACL](https://aclanthology.org/2026.acl-long.538/)
5. *AlphaBench*. [OpenReview](https://openreview.net/forum?id=d97Q8r7ZKZ)
6. *Synergistic Formulaic Alpha Generation* 扩展. [arXiv 2401.02710](https://arxiv.org/abs/2401.02710)
7. Hybrid FAFM. [FCS 2026](https://link.springer.com/article/10.1007/s11704-025-41061-5)
