<div align="center">

# Quantlab

**A 股量化选股系统**

</div>

## 简介

Quantlab 从 Tushare 拉取行情数据存入 DuckDB，经因子管道计算后，由三个 LightGBM 回归模型分别预测 **2 / 6 / 20 日**前向开盘收益，加权融合为单一分数，驱动每日开盘市价组合。

系统覆盖数据更新、模型训练、滚动折 CV 回测、泄漏断言与 HTML 预测报告的完整流程。
