# 破坏性简化瘦身设计（2026-08-25）

用户指令：合并各部分、允许破坏性删除重写、合并可合并的类与 .py 文件、尽可能删除功能。
背景：用户已手工删除 tmp/、README.md、pools/*.json（旧池历史并集）、report_strategy.py、tmp/archive/。
**手工删除已把 cron 流水线弄断**（见下），本重构同时修复。

## 触发的直接修复（json 池删除的连带）

| 断点 | 现状 | 修复 |
|---|---|---|
| `config.get_pool_codes`/`load_all_pool_stocks` 读已删 json | 多处 import（部分 stale）；integrity 检查 6 静默跳过；compute 入口会崩 | **删除全部 json 池读取**，池代码单源收敛到 `pools.membership`（pool_snapshots 表） |
| `data/sources._pool_union_codes()` 返回空集 | namechange 静默停更（ST/退市事件断流）、行业覆盖触发失效 | 改 `union_codes(con=con)`（传入调用方连接，避免 DuckDB 单写者锁冲突） |
| `pools/build_pool_history.py` 验证门读已删 json | 重建会崩 | 并入 membership.py 时删该验证门 |

## 删除清单（功能删除）

**根**：fold_cv_slim.py（一次性验证，结论已入 AGENTS.md）、_check_median_open.py、_check_combine.py（一次性校验脚本）、_check_pkgs.py
**backtest**：signals.py 的 `run_portfolio`（DEPRECATED）、`run_long_short`（不稳定诊断件）、`compute_benchmark`（非时点版，已被 pit 版取代）、**limit 执行语义全链**（已判死清单在册：`--exec limit`、AUCTION_BUFFER、SELL_MARKUP、rank_scores/rank_threshold 参数）；test_holding.py（DEPRECATED 且读死列）
**strategies**：labels.py 的 4 个死标签函数（peak_high/peak_close/median_close/smoothed）；lgb.py 的 classifier 分支/peak loss/dart/rolling walk-forward 分支（无调用方）；combine_scores 旧双模型版；pearson_ic；BaseStrategy ABC（唯一子类并入 LGBStrategy）
**factors**：migrate_*.py ×5（一次性回填，已完成）、recompute_pool_history.py、registry.py + build_model_factors.py（mf_ 机制空转，注册表已清空；范式铁律文档迁至 extra_factors.py 头注与 AGENTS.md）、build_mf_volchg3.py（用户裁定暂放的实验）、test_causality.py（2026-08 审计已完成并记录）、baseline_alphas.py（并入 baseline_check.py）
**data**：build_cyq.py / build_delist_info.py / build_index_db.py / build_db.py（一次性表构建器，已被 pull 覆盖；build_industry.py 保留——pull._trigger_industry 子进程调用）
**其他**：trade_signals/（三分类时代信号导出，无消费者）、document/archive/、data/fold_cv_report_v7score.json + fold_cv_slim_report.json（旧报告）、旧 predictions__mainboard_microcap_lgb.parquet（三分类遗留）

## 合并清单

| 前 | 后 | 说明 |
|---|---|---|
| strategies/{base,lgb,combine,evaluation}.py | strategies/lgb.py | LGBStrategy（去 ABC）+ walk_forward（仅固定测试集分支）+ buffered_train_end + rank_ic/ic_summary + combine_scores3 |
| strategies/labels.py | 保留（删死函数） | 泄漏断言 C4 依赖其源码可 inspect |
| backtest/{run_lgb,signals}.py | backtest/run_lgb.py | 入口+模拟器合一；market 开盘语义为唯一模式；PRED_COLS/W2D/W6D/W20D 单源不变 |
| pools/{membership,build_pool_history}.py | pools/membership.py | 成员资格查询 + 快照构建器（`python -m pools.membership --from 2015`） |
| factors/{compute,update}.py | factors/update.py | 增量（cron 路径不变）+ `--full/--from/--to` 全量重建 |
| factors/{factor_audit,factor_contribution,test_new_factors}.py | factors/mining.py | 子命令 audit / contribution / batch |
| factors/{baseline_check,baseline_alphas}.py | factors/baseline_check.py | 门禁 + 基准因子一体 |

**cron 四步模块路径全部保持不变**：data.pull / factors.update / factors.build_nn_gap1d / forecast_display/generate_lgb.py。

## 保留（明确不碰）

data/{pull,sources,trading_calendar,lock,_ts,build_industry}.py（摄入层+cron）；factors/{extra_factors,select_factors,integrity,build_gb_gap1d,build_nn_gap1d}.py；run_lgb.py / fold_cv.py / _leak_check.py / config.py；document/{alpha101.md,论文,llm_factor_mining,tushare_md_docs,vnpy}（vnpy 为用户新放参考拷贝，untracked 不动）；docs/superpowers/ 既有 spec/plan；models/（含折产物）；当前回测产物。

## 对抗审查要点（已核）

1. **旧折 joblib 兼容**：LGBStrategy.load 按 inspect.signature 过滤旧 config 键（drop_rate/l1_loss_horizon 等），旧 bundle（纯 regressor）可加载——factor_contribution --fold 消费。
2. **锁冲突**：sources/integrity 在 pull 的写连接进程内取池代码必须 `union_codes(con=con)`（membership 自开只读连接会撞单写者锁）。
3. **namechange 池过滤语义**：快照并集 vs 旧 json 并集——旧 json 也是静态人工更新，新成员缺历史属既有属性；且快照并集正是现役宇宙，语义更诚实。pool_snapshots 缺表时大声失败（fail-fast）。
4. **baseline_check 冻结参考**：2026-08-22 冻结于旧 json 池宇宙，切快照并集后 IC 必漂移——先跑对比确认漂移方向合理，再 `--init` 重冻结并在报告注明（宇宙变更，非评估器回归）。
5. **等价性验证**：重构后重训四模型，parquet md5 + meta IC/calib 对比快照；回测对横幅数字（+48.72%/2.47/−6.70%、基准 +26.17%、超额 +22.55pp）；_leak_check 52 项；fold_cv --skip-train 抽折复跑。
6. **不动 DB**：全程只读库 + 重训写 models/parquet（文件），避开 21:05 cron 窗口。

## 验证后收尾

AGENTS.md 重写（单工作区、新结构、mining 子命令、update --full）；agent memory 更新。
