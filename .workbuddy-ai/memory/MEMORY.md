# quantlab 项目长期记忆

## 用户原则（重要）
- **AI 做的评测/分析结论不要写入记忆**：用户不默认信任 AI 评测，此类结论每次用时重新验证；只记客观事实（表结构、流程、命令），不记分析判断

## DuckDB 表结构要点（校验 SQL 别再踩坑）
- 行情表名是 `daily_raw`，不是 `daily`；日期列是 `date`，不是 `trade_date`
- `daily` 是 SQL VIEW（前复权），DuckDB 直连时 information_schema 可能不显示，校验请用 `daily_raw`
- 因子表 `factor_values`：日期列为 `date`（字符串日期）
- 筹码表是 `cyq_perf`（不是 cyq）；指数表是 `index_daily`；SHIBOR 在 `macro_daily`（列 date/shibor_on/shibor_1m）
- 补拉状态表 `pending_pulls`（source/date/reason/attempts/status/updated_at）：rolling repull 分片空响应会登记 pending，行级数据可能已完整，判定缺失以行数差集为准

## 每日流水线（automation-1786972402049）
- data.pull 末尾 integrity 硬失败提示「最新开市日 factor_values 无数据」是**预期现象**（因子在第二步 factors.update 才计算），第一步只需确认 daily_raw 当日行数 > 0 即可
- 三步顺序：data.pull → factors.update → forecast_display/generate_lgb.py，解释器统一 `/Users/cui/.workbuddy-ai/quantlab-env/bin/python`
- 长命令放 Bash 前台跑没问题，但**不要在流水线中间插临时校验命令**——若校验 SQL 写错报错，回合会中断，用户以为流水线停了；校验放在步骤间隙或结束后做
- 产出报告：`forecast_display/html_lgb/mainboard_microcap/{date}_forecast_lgb.html`
