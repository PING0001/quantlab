# 双回归模型设计：20d + 6d 开盘中位数收益预测（v2）

日期：2026-08-20（v2 经对抗审查修订；2026-08-21 模型命名 5d→6d，用户裁定）
分支：`feat/regression-dual-model`（从 main 切，未创建）
状态：待用户评审

> v2 修订摘要：对抗审查发现 3 个 P0（退市标签语义描述错误、百分位分数进撮合限价公式导致回测失真、每日生产流水线无切换保护）+ 4 个 P1，全部已并入本版。详见 §9 审查修订记录。
> 命名规则：模型名 = 标签窗口末端交易日（T+16~T+20 → 20d；T+4~T+6 → 6d）。
> 调仓频率：每日调仓（REBALANCE_FREQ=1，2026-08-21 用户裁定，替代原 5 日调仓）。

## 1. 背景与动机

现状：单 LGBMClassifier 三分类（标签 = T+16~T+20 **收盘价**中位数收益，>=+8%→+1 / <=-4%→-1），打分 = 期望收益变换 `p(+1)*0.08 + p(-1)*(-0.04)`。

本设计四个动机：

1. **回归替代分类**：固定阈值丢失收益幅度信息；直接回归预测收益绝对值，预测值可直接解释。
2. **入场基准对齐**：标签基准改为 T+1 开盘价，与回测的次日开盘入场撮合方向一致。注意这是"入场端对齐"，**不是**标签-回测完全等价，残余口径差见 §6.2。
3. **双模型综合买点**：新增 6d 级别模型（T+4~T+6 开盘中位数），与 20d 模型加权综合分判断买点。回测改为**每日调仓**：综合分日频重估买点，6d 模型提供近期收益预期（日频换仓决策的直接输入），20d 模型提供中期选股方向。
4. **日内数据路线落定**：分钟数据调研结论为不可得（Tushare 分钟权限为独立付费商品 2000 元/年、积分买不到；quicksync 中转实测返回"权限不足"）。改为零成本的日线 OHLCV 衍生日内形态因子。

## 2. 决策记录

### 用户已确认

| 决策 | 内容 |
|---|---|
| 20d 模型 | 也改为回归 + 开盘中位数（原三分类退役） |
| 6d 标签 | T+4~T+6 开盘价中位数，回归 |
| 模型命名 | 按标签窗口末端命名：T+4~T+6 → **6d**（2026-08-21 裁定） |
| 回测调仓 | 5 日调仓改为**每日调仓**（REBALANCE_FREQ=1，2026-08-21 裁定） |
| 买入方式 | **阈值准入替代强制 top-N**（2026-08-21 裁定）：exec_score ≥ τ（池化分位校准，--entry-q 0.90/0.95）才可买，max_positions 为上限，允许空仓。实测：按日百分位尺度恒有 ~10% 合格、门槛形同虚设；只有 exec 绝对尺度有日间咬合（P95 时 49/242 个零合格日） |
| 限价口径 | **v3（2026-08-21 终版）**：买入 k=空仓份数、预测排名前 k（剔已持有）挂 `收盘×(1+pred−3%)` 便宜单，成交即价格纪律；卖出 = 每晚全部持仓按目标价 `收盘×(1+pred)` 挂单（无上浮），退出仅两条路径——预测转负（低价出货单）或价格触及目标价（止盈结算）；掉出排名不再触发卖出。阈值准入降级为可选参数（--entry-q，默认关） |
| 综合方式 | 直接加权综合分（不做 gate/择时对比实验） |
| 代码形态 | 参数化统一入口（不复制脚本、不单 bundle 双标签） |
| 日内数据 | 分钟数据暂缓；做日线衍生因子 |

### 代理裁定（用户未及回答，均可配置改）

| 决策 | 裁定 | 理由 |
|---|---|---|
| 标签基准价 | **T+1 开盘价** | 信号 T 收盘后可得，最早可执行买点 = T+1 开盘，与回测次日开盘撮合方向一致。`baseline` 做成配置项（next_open / open / close）。**仍待用户复核** |
| 综合分双通道 | rank_score（百分位）+ exec_score（原始值）分开 | 撮合层限价公式把分数当预期收益率用（`signals.py:64-71`），百分位直接进公式会让卖单永不成交（P0-2，见 §3.5/3.6） |
| 退市标签语义 | 保留"部分窗口中位数 + 窗口全缺→NaN 丢弃"，不实现 -1.0 填充 | 旧 -1.0 填充是死代码（实证见 §3.2）；窗口内可得开盘价是真实可成交价；退市风险由回测组合层处理（delist 排除 + 退市处理） |
| 归一方式 | 按日期横截面百分位归一到 [0,1] 再加权（仅用于排序） | 6d 与 20d 收益量纲不同；排序场景下 rank ensemble 无损（回测只用排名） |
| 综合权重 | 20d:0.6 / 6d:0.4 初始 | 6d IC 预期较低（窗口短噪声大），进配置 |
| 回归 objective | L2 起步，huber 留配置开关 | 临近退市的部分窗口极端负标签（真实崩盘收益）是重尾；是否换 huber 作为实验项 |

## 3. 架构设计

### 3.1 模型注册表 + 统一训练入口

`run_lgb.py` 参数化：`python run_lgb.py --model 20d|6d|all`。模型配置集中在注册表（config.py 或 run_lgb.py 顶部）：

```python
MODEL_CONFIGS = {
    "20d": dict(
        label_window=(16, 20),      # T+16 ~ T+20
        label_price="open",
        baseline="next_open",       # open[T+1]
        label_buffer=20,
        horizon="label_20d",        # pred 列: pred_label_20d
    ),
    "6d": dict(
        label_window=(4, 6),        # T+4 ~ T+6
        label_price="open",
        baseline="next_open",
        label_buffer=6,
        horizon="label_6d",         # pred 列: pred_label_6d
    ),
}
```

路径函数全部加 `model` 参数（config.py）：

| 产物 | 路径 |
|---|---|
| 模型 | `models/{pool}/lgb_{model}.joblib`（lgb_20d.joblib / lgb_6d.joblib） |
| 预测缓存 | `data/predictions__{pool}_lgb_{model}.parquet` |
| 因子清单 | `factors/selected_{pool}_{model}.json` |
| meta | `data/predictions__{pool}_lgb_{model}_meta.json` |

**旧产物不删不改**（`lgb_multi.joblib`、`predictions__{pool}_lgb.parquet`、`selected_{pool}.json`），作为整体回滚保险。旧代码读取旧路径的产物在切换前保持可用（generate_lgb 回退分支依赖此性质，见 §3.7）。

已验证（审查确认）：`buffered_train_end`（`base.py:48-59`）在 buffer=20/窗口端 T+20、buffer=6/窗口端 T+6 下恰好无泄漏且为最小值（训练末行索引 i−buffer−1，窗口末端落在首个测试日前一天）。字符串 horizon 在 `strategies/lgb.py` 的 fit/predict/save/load 全链路即插即用，无分类残留。

### 3.2 标签：labels.py 新增 compute_median_open

```python
compute_median_open(kline_df, start_day, end_day, baseline="next_open", delist_info=None)
# label = median(open[T+s .. T+e]) / open[T+1] - 1
```

- 前复权 open，来自 `daily_kline`（与现有 close 标签同源）；shift 按个股自身交易行移位（停牌股的 T+k 落在复牌后，见 §6.1 残余风险）
- baseline 可选：`next_open`（默认）/ `open`（T 当日）/ `close`（T 当日）

**退市语义（v2 修正，P0-1）**——旧实现的真实行为（已实证，非文档声称的行为）：

- `pd.concat(shift(-d)).median(axis=1)` 默认 skipna → 窗口内 ≥1 个未来开盘价即得**部分窗口中位数**（临近退市/数据末端的行）
- 窗口全缺 → NaN → 被 `run_lgb.py:194` 的 notna 过滤丢出训练集
- 旧 -1.0 填充条件 `date >= delist_date` 在全库 0 行命中（行情最后一笔永远早于摘牌日），**从未生效**
- 新标签沿用该真实行为并如实文档化：部分窗口中位数保留（可得开盘价 = 真实可成交退出价，携带崩盘信号）；不实现 -1.0 假填充；退市风险由回测组合层（delist 排除 + 退市归零处理）承担
- 同步纠正 AGENTS.md 中"退市感知 -1.0"的描述（归入 §7 收尾任务）
- `compute_median_close` 保留不动（旧口径对照）

### 3.3 训练入口改造（run_lgb.py）

- 标签由注册表驱动；`_classify` 闭包删除
- `model_type="regressor"`（LGBStrategy 原生路径，`strategies/lgb.py:229/288-290`）
- 分类专属评估（sign 准确率、label dist）替换为回归口径：pred vs 自身连续标签的 rank IC / IR / hit rate（复用 `rank_ic`/`ic_summary`）+ MAE + 十分位收益单调性
- 训练排除逻辑不变：IsST=1、退市后观测、next-open 涨跌停 mask
- walk_forward 固定测试集协议不变：train < TEST_START − label_buffer，test 2025-06-01 ~ 2026-06-01
- **超参注意（P2）**：现有 LGB_KWARGS 为分类调参（num_leaves=8/depth=4/min_child=2000/colsample 0.3），且分类分支 pop 掉 reg_alpha/reg_lambda 而回归分支保留——正则强度实际有变化。首跑沿用，IC 结果作为回归基线后再调参

### 3.4 因子预筛选按模型（select_factors.py）

- `python -m factors.select_factors --model 20d|6d`，标签改为对应 `compute_median_open`（与训练同一构造，避免选择目标错位）；IC 循环从 4 horizon 缩为单标签
- 输出 `selected_{pool}_{model}.json`；MUST_INCLUDE / corr_threshold=0.75 / MAX_FACTORS 沿用
- 顺带清理：`(labels == -1.0)` 排除条件（`select_factors.py:126`）对 open 标签同样是死条件，参数化时移除；`TRAIN_START=2015-01-01` 与 run_lgb 的 2020-01-01 口径差异在输出 json 中如实记录（不改历史行为）

### 3.5 综合分模块（新 strategies/combine.py）

```python
combine_scores(pred_20d, pred_6d, w20=0.6, w6=0.4) -> pd.DataFrame  # 两列
# rank_score: 每日期横截面 percentile → [0,1] → 加权和（排序/展示/IC 用）
# exec_score: 原始预测值加权和（收益量纲，撮合限价公式用）
```

- **双通道是硬要求（P0-2）**：撮合层 `_buy_limit/_sell_limit = prev_close × (1 ± pred ∓ buffer)` 把 pred 当预期收益率；百分位直接进公式 → 卖限价 ≈ 1.5×前收永不成交、组合只进不出
- 某日期某股票只有一个模型有预测：按可用模型权重重归一（这是**必需**行为而非兜底——6d 标签尾部 NaN 裁剪比 20d 少，6d 预测比 20d 多覆盖测试期尾部约 14 个交易日）
- rank_score 用于：回测排序、HTML 展示、IC 评估；exec_score 用于：撮合限价

### 3.6 回测（backtest/run_lgb.py + signals.py 最小改动）

- `run_portfolio_rebalance` / `run_long_short` 增加可选参数 `rank_scores`（默认用 `predictions`）：**排序用 rank_scores，限价公式用 predictions（exec 量纲）**——这是对 v1"撮合层零改动"声明的修正（P0-2）
- `backtest/run_lgb.py`：加载两份 parquet → `combine_scores` → 两个 Series 分别传入
- IC 参考块：两模型各自 vs 自身开盘中位数标签 + rank_score vs 两个标签（limit/ST 过滤沿用）
- `REBALANCE_FREQ` 5 → **1（每日调仓）**，保持参数化；输出文件独立命名：`equity_lgb_combined_daily_rebalance.csv` + **独立 benchmark 文件**（`benchmark_combined.csv`，勿覆写旧 `benchmark.csv`——`report_strategy.py:15-16` 还在消费旧文件做对照）
- 消费方处置（P1-3）：`backtest/test_holding.py` 读 `pred_5d` 列今天就已是坏代码（现 parquet 只有 pred_label），本 bench 一并更新或显式标记废弃；`report_strategy.py` 指向旧 equity 文件名保持可用（旧文件不删）
- `run_portfolio`（非 rebalance 路径）的 `entry_threshold` 语义依赖收益量纲，不在本 bench 适配范围，标记弃用注释

### 3.7 预测展示（forecast_display/generate_lgb.py）与生产流水线保护

- 加载两个模型 → 最新日期因子 → 两列预测 + rank_score → HTML，表格同时展示 pred_label_20d / pred_label_6d / 综合分供人工对照
- 删除 `NEUTRAL=0.015` 与旧 4-horizon `WEIGHTS`；百分位口径 0.5 为中性
- HTML 文件名加后缀（如 `{date}_forecast_lgb_dual.html`），避免同日新旧报告互相覆盖
- **生产回退分支（P0-3）**：workbuddy 每日流水线（`.workbuddy/automations/automation-1786972402049/run_pipeline.zsh`）19:30 直接调用本脚本。双模型任一缺失时回退到"可用模型单跑 + 报告显著标注 DOWNGRADED"，不 exit 1；仅当两个模型都缺失时才回退旧分类模型路径
- `factors/integrity.py` 软警告 5 读取 `selected_{pool}.json`，参数化为读取 per-model 清单（否则永远检查过时因子表）
- workbuddy 使用独立 venv（`/Users/cui/.workbuddy-ai/quantlab-env`）——本 bench 不引入新第三方依赖，仅用现有库，无需动该 venv

### 3.8 日线衍生日内形态因子（extra_factors.py）

初版 8 列，全部基于 daily_kline 前复权 OHLC：

| 因子 | 公式 |
|---|---|
| UpperShadow | (high − max(open, close)) / close |
| LowerShadow | (min(open, close) − low) / close |
| ClosePos | (close − low) / (high − low)，high==low（一字板）时 NULL |
| OpenPos | (open − low) / (high − low) |
| ShadowRatio | UpperShadow / (UpperShadow + LowerShadow)，分母 0 时 NULL |
| RangeEfficiency | Intraday_range_pct / 当日换手率（换手率 = amount/circ_mv/10，与 Turnover_3d 同源） |
| ClosePos_mean_20d | ClosePos 的 20 日均值 |
| ClosePos_std_20d | ClosePos 的 20 日标准差 |

- 走 `compute_non_alpha_factors` 标准入口 + `config.SELECTED_FACTORS` 注册。**新列必须永久进 compute 计算路径**：全量重建 `store_factor_values` 是 DROP TABLE 重建（仅保护 ai_gz2000_*），表内手工列会在重建时丢失
- **存储迁移顺序约束**：`write_panel` 只写表中已有列（`update.py:141-143`）→ 先 `ALTER TABLE ADD COLUMN` × 8 → 一次性全历史回填（复用 compute_panel 全历史，仅新列 UPDATE 写入，不动其他列所有权）→ 之后 `factors.update` 自动维护。**ALTER 之后、回填完成之前不要跑 `factors.update`**（新列近 250 日 NULL 率 ~100% 会触发 integrity 软警告 7 刷屏，且写入空值）
- 回填勿与 `data.pull`/`factors.update` 并发（DuckDB 写锁互斥）
- 遗留缺口联动：594 只待 backfill 股票对新列同样缺历史，回填时用 `--backfill-stocks` 同机制一并处理
- 预验证用 `factors/test_alpha.py`：注意其 IC 基于 close-to-close 收益，与最终 open 基准标签口径不同，结论仅作初筛参考

## 4. 数据流

```
daily_kline(前复权) ──> compute_median_open ──> label_20d / label_6d
factor_values ──> select_factors --model X ──> selected_{pool}_{model}.json
             └──> run_lgb.py --model X ──> predictions__{pool}_lgb_{model}.parquet
                                          └──> combine_scores(rank/exec) ──> backtest / HTML / signals
```

## 5. 验收标准

1. **标签正确性**：抽样核对 median open return，必含案例：正常股、退市股（部分窗口 + 全缺 NaN）、长期停牌股（T+k 落复牌后）、high==low 一字板、baseline=next_open 与回测入场价对齐
2. **泄漏检查**：`_leak_check.py` 参数化（现硬编码 16/20）后两模型通过；**新增按个股行移位语义的泄漏断言**（全市场日历 buffer 防不住跨界停牌股，见 §6.1），至少对测试期边界前后 30 日的所有训练行验证其标签窗口不引用测试期价格
3. **训练**：两模型对自身标签的测试期 rank IC 显著为正（mean > 0，IR/hit 首跑定标作为后续回归基线）；~~与旧分类 0.26 对照~~（v2 删除：旧参考 IC 是 close 基准 ret_median，与新 open 基准标签口径不同，不可比）
4. **combine**：日期对齐、单模型缺失重归一、rank/exec 双通道量纲各有断言
5. **回测**：综合分回测全流程跑通；抽查卖单限价在合理区间（±10% 带内），确认无"永不成交"持仓；equity/benchmark CSV 产出且旧对照文件未被覆写
6. **展示**：HTML 含双模型分数 + 综合分；删除任一模型文件时回退分支生效（本地演练）
7. **增量**：`python -m factors.update` 后新因子列有当日值且不破坏列所有权（ai 列不受影响）
8. **生产演练**：按 workbuddy 流水线三步在本地手动顺序执行一遍，确认无断流

## 6. 风险与残余口径差

### 6.1 风险

- **基准价 next_open 为代理裁定**，待用户复核（改口径 = 改配置 + 重训）
- **停牌泄漏残差（P1-1）**：标签按个股行移位、buffer 按全市场日历——跨界停牌股的训练行标签窗口可能落入测试期，buffer 防不住，`_leak_check` 日历推演也盲视。发生频率低（池内实证仅个例），验收 2 的个股级断言作部分防护，残余接受并记录
- 6d 标签窗口仅 3 点，回归噪声大，IC 预期低于 20d；0.6/0.4 权重合理性待回测观察后调
- **每日调仓换手率显著上升**：佣金/印花税假设的敏感性变大，T+1 锁定下最短持有约 1 天；回测按现有费率如实计成本，若成本侵蚀明显再评估调仓频率
- 临近退市部分窗口极端负标签（≈−0.8~−0.9，真实崩盘收益）在训练集中：保留（携带真实信号）；若 L2 受重尾影响明显，实验 huber / winsorize
- factor_values 新列全历史回填计算量较大（~1100 股 × 4600 交易日，复用 compute_panel 全历史重算），bench 内一次性跑
- LGB 超参沿用分类调参 + 正则实际增强，首跑后需按回归目标重调

### 6.2 残余口径差（标签 ≠ 回测逐笔等价，如实列出）

- **停牌**：标签用个股下一交易行开盘价（复牌价），回测对 T+1 无交易的买单放弃且当日过期（`signals.py:520-524`）
- **撮合**：日内限价成交价可能劣于/优于开盘价（buy limit 挂 pred−2%），叠加佣金 0.06%×2 + 印花税 0.05%
- **退出端**：实际每日调仓退出，不按 T+16~T+20（20d）中位数卖出——20d 标签是选股信号，不是交易计划
- **分数通道**：排序用 rank_score、执行用 exec_score，两通道在极端日可能给出不一致的方向（可接受，属设计属性）

## 7. 实施顺序（v2 重排：因子迁移提前，避免两模型因子宇宙不一致）

1. `labels.compute_median_open`（含退市语义）+ 标签单元验证（验收 1 案例）
2. config 模型注册表 + 路径函数参数化
3. **8 个日内形态因子**：extra_factors + compute 路径 + SELECTED_FACTORS 注册 + ALTER TABLE + 全历史回填（遵守 §3.8 顺序约束）
4. `select_factors.py` 参数化，20d / 6d 各跑一遍（此时因子池已含新列，两模型候选宇宙一致）
5. `run_lgb.py` 参数化，训练 20d（回归基线定标）
6. 训练 6d
7. `strategies/combine.py`（rank/exec 双通道）
8. `signals.py` 解耦（rank_scores 参数）+ `backtest/run_lgb.py` 改造 + test_holding / report_strategy 处置
9. `generate_lgb.py` 改造（含生产回退分支）+ `integrity.py` selected 清单参数化
10. `_leak_check.py` 参数化 + 个股级泄漏断言
11. 全链路验收（§5）+ AGENTS.md 同步（模型架构、因子数、退市描述纠正、workbuddy 演练记录）

## 8. 部署与回滚

- **合并前置条件**：两模型 joblib + 两 selected_{pool}_{model}.json + 8 列历史回填完成 + generate_lgb 回退分支就绪——四者齐备才可合并到 main（否则 workbuddy 当晚断流）
- **回滚**：旧模型/预测/selected/HTML 路径全部保留未删；generate_lgb 回退分支支持单模型运行；极端情况 git revert + 旧文件直接可用
- workbuddy 流水线本身（zsh 脚本）无需改动，三步命令不变

## 9. 审查修订记录（2026-08-20 对抗审查）

| 级别 | 问题 | 处置 |
|---|---|---|
| P0-1 | "退市 -1.0 填充照抄"描述的是死代码（实证：全库 `date >= delist_date` 行数 0；退市股标签实为部分窗口中位数 + 尾部 NaN 丢弃） | §3.2 重写退市语义；§2/§6 风险项改写；AGENTS.md 纠正列入收尾 |
| P0-2 | 百分位综合分进 `_buy_limit/_sell_limit` 限价公式 → 卖单永不成交、组合只进不出（`signals.py:64-71`，已第一手核实） | §3.5 双通道（rank/exec）；§3.6 signals.py 解耦；验收 5 加限价区间抽查 |
| P0-3 | workbuddy 每日流水线直接调用 generate_lgb.py，双模型缺失即断流（automation 脚本已核实） | §3.7 生产回退分支；§8 合并前置条件；验收 6/8 |
| P1-1 | 停牌股按行移位的标签窗口可越过日历 buffer 泄漏入测试期 | §6.1 残差声明 + 验收 2 个股级断言；确认日历意义 buffer 无 off-by-one |
| P1-2 | 实施顺序导致两模型因子宇宙不一致；验收 3 与 §6 自相矛盾（0.26 跨口径对照无效） | §7 重排；验收 3 改为自身标签显著为正 |
| P1-3 | 遗漏消费方：test_holding.py（已坏）、integrity.py selected 清单、report_strategy.py、HTML 同名覆盖 | §3.6/3.7 逐项处置 |
| P1-4 | "三者口径统一"表述过强（停牌/限价成交/费用/退出端差异仍在） | §1 改"入场基准对齐"；§6.2 残余口径差清单 |
| P2 | 超参迁移、_leak_check 硬编码、select_factors TRAIN_START 口径、test_alpha close 基准、迁移中间态告警、手工列禁令、run_portfolio 弃用 | §3.3/3.4/3.8/3.6 相应条目 |
