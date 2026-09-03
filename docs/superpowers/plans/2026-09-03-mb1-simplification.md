# mb1 线简化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按 spec 删除挖矿层、主模型收敛为三（open2d/6d/20d）、ML 因子随折同步训练（防泄漏）、全模型权重单文件不留档。

**Architecture:** 承重墙不动（store/dataset/labels/update），砍增生层（挖矿 4 py + 候审构建器 ×2），fold_cv 重写为"ML 同步训练 → 三主模型 → 泄漏断言 → 回测 → ICIR+收益+夏普汇总"，run_lgb 折模式读主清单、权重写主路径直接覆盖。

**Tech Stack:** Python 3.13 / DuckDB / LightGBM / XGBoost / polars（既有，无新增依赖）

**Spec:** `docs/superpowers/specs/2026-09-03-mb1-simplification-design.md`

## Global Constraints

- 环境变量：一切命令带 `QUANTLAB_DB=/Users/cui/Projects/quantlab/data/ashare.duckdb`；池 `--pool mainboard_all`；解释器 `.venv/bin/python`。
- **DB 单写者锁**：Task 6/7/9/10 会写因子表列（store.update_columns）——执行前 `ps aux | grep -E 'data\.pull|factors\.update'` 确认无流水线，避开工作日 21:05 前后。
- **cron 四步模块路径是硬契约**：`data.pull` / `factors.update` / `factors.build_nn_gap1d` / `forecast_display/generate_lgb.py` 全程不删不改名。
- **DB 只动 mainboard_all 池资产**；因子表七根"保留列"不 DROP；微盘折模型目录 `models/mainboard_microcap/folds/` 不动。
- 本仓无 pytest 基建——每任务的验证 = `py_compile` + 定向 grep + 冒烟跑（仓内既定验证文化）。
- 每任务一提交，提交信息中文、conventional 前缀。

---

### Task 1: 删除清道（代码 + git 产物 + 盘上残留）

**Files:**
- Delete (git rm): `factors/mining.py` `factors/select_factors.py` `factors/baseline_check.py` `factors/ic_probe.py` `factors/build_gb_4d_open2d.py` `factors/build_gb_30d_turn5d.py` `docs/mb2_ideas.md` `factors/folds/`（全部 `selected_mainboard_all_*.json`）`models/mainboard_all/folds/` `data/ic_probe_mainboard_all.json` `data/ic_probe_mainboard_microcap.json` `factors/selected_mainboard_all_gap1d.json` `factors/selected_mainboard_microcap_gap1d.json` `models/mainboard_all/lgb_gap1d.joblib` `models/mainboard_microcap/lgb_gap1d.joblib` `models/mainboard_all/nn_gap1d_state.joblib` `data/predictions__mainboard_all_lgb_gap1d.parquet` `data/predictions__mainboard_all_lgb_gap1d_meta.json` `data/predictions__mainboard_microcap_lgb_gap1d.parquet` `data/predictions__mainboard_microcap_lgb_gap1d_meta.json`
- Delete (盘上，gitignored): `data/folds/F*/predictions__mainboard_all_*`、`backtest/mainboard_all/folds/`、`tmp/` 全部内容、data/ 下未跟踪挖矿报告（factor_audit_*/contribution_report_*/batch 报告，存在才删）

**Interfaces:**
- Produces: 干净树；`_rank_ic_np` 的旧家已删（Task 2 建新家），后续任务无悬挂 import。

- [ ] **Step 1: git rm 已跟踪删除项**

```bash
cd /Users/cui/Projects/quantlab-bench-mainboard-all
git rm -q factors/mining.py factors/select_factors.py factors/baseline_check.py \
  factors/ic_probe.py factors/build_gb_4d_open2d.py factors/build_gb_30d_turn5d.py \
  docs/mb2_ideas.md data/ic_probe_mainboard_all.json data/ic_probe_mainboard_microcap.json \
  factors/selected_mainboard_all_gap1d.json factors/selected_mainboard_microcap_gap1d.json \
  models/mainboard_all/lgb_gap1d.joblib models/mainboard_microcap/lgb_gap1d.joblib \
  models/mainboard_all/nn_gap1d_state.joblib
git rm -rq models/mainboard_all/folds
# 折清单只删 mainboard_all 份（微盘折清单如有属微盘线，边界外保留）；空目录随删
git rm -q factors/folds/F*/selected_mainboard_all_*.json
find factors/folds -type d -empty -delete 2>/dev/null || true
```

- [ ] **Step 2: 盘上清理（gitignored 残留——predictions__* 系未跟踪，用 rm 不用 git rm）**

```bash
rm -f data/predictions__mainboard_all_lgb_gap1d.parquet data/predictions__mainboard_all_lgb_gap1d_meta.json \
      data/predictions__mainboard_microcap_lgb_gap1d.parquet data/predictions__mainboard_microcap_lgb_gap1d_meta.json
rm -f data/folds/F*/predictions__mainboard_all_*
rm -rf backtest/mainboard_all/folds
rm -rf tmp/*
ls data/ | grep -E "factor_audit|contribution|batch" && rm -rf data/factor_audit_* data/contribution_report_* data/*batch_report* || true
```

- [ ] **Step 3: 悬挂引用核查（此刻预期只剩 Task 2-5 将改的文件）**

```bash
grep -rn "select_factors\|factors\.mining\|baseline_check\|ic_probe\|build_gb_4d\|build_gb_30d\|compute_forward_returns" \
  --include='*.py' . | grep -v '\.venv'
```
Expected: `run_lgb.py`（错误信息一处）、`fold_cv.py`（多处，Task 5/7 改）、`strategies/labels.py`（compute_forward_returns 定义，Task 3 删）、`config.py`/`dataset.py`/`factors/__init__.py` 注释（Task 3 改）。出现其他 .py 引用即停下排查。

```bash
grep -rn "gap1d" --include='*.py' config.py run_lgb.py fold_cv.py _leak_check.py | grep -v "gb_gap1d"
```
Expected: `config.py` MODEL_CONFIGS 条目（Task 3 删）、`fold_cv.py` 模型循环（Task 7 改）、`run_lgb.py` 无（模型集来自 sorted(MODEL_CONFIGS)）。

- [ ] **Step 4: 编译全仓确认无 import 断**

```bash
find . -name '*.py' -not -path './.venv/*' -not -path './tmp/*' -print0 | xargs -0 .venv/bin/python -m py_compile && echo COMPILE_OK
```

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor: 挖矿层全删 + 候审构建器退役 + gap1d/nn 产物清 + 折清单/折模型/旧折缓存清（spec 判定表）"
```

---

### Task 2: strategies/lgb.py——积木迁入 + 旧键兼容层删除

**Files:**
- Modify: `strategies/lgb.py`

**Interfaces:**
- Produces: `strategies.lgb._rank_ic_np(f_vals: np.ndarray, l_vals: np.ndarray) -> float`、`strategies.lgb.MIN_STOCKS_PER_DATE = 30`（临时脚本挖矿积木新家）；`LGBStrategy.load` 直载无键过滤。

- [ ] **Step 1: `ic_summary` 之后追加积木（含 scipy import）**

文件头 import 区加 `from scipy.stats import rankdata`；在 `ic_summary` 函数后追加（自 select_factors 逐字迁移）：

```python
# ============================================================================
# 截面 rank IC 积木（2026-09-03 自 factors/select_factors.py 迁入——挖矿层
# 已删，后续挖矿/因子筛选一律 tmp/ 临时脚本，从本处 import）
# ============================================================================

MIN_STOCKS_PER_DATE = 30


def _rank_ic_np(f_vals, l_vals):
    """Compute rank IC (Spearman) using numpy/scipy rankdata."""
    valid = ~np.isnan(f_vals) & ~np.isnan(l_vals)
    n = valid.sum()
    if n < MIN_STOCKS_PER_DATE:
        return np.nan
    f_r = rankdata(f_vals[valid])
    l_r = rankdata(l_vals[valid])
    f_c = f_r - f_r.mean()
    l_c = l_r - l_r.mean()
    denom = np.sqrt(np.dot(f_c, f_c) * np.dot(l_c, l_c))
    if denom == 0:
        return np.nan
    return np.dot(f_c, l_c) / denom
```

- [ ] **Step 2: `LGBStrategy.load` 删旧键兼容过滤**（旧折 bundle 已删，无旧格式消费方）

```python
    @classmethod
    def load(cls, path: str | Path) -> "LGBStrategy":
        path = Path(path)
        bundle = joblib.load(path)
        strategy = cls(
            factor_names=bundle["factor_names"],
            horizons=bundle.get("horizons", (1, 3, 5, 10)),
            **bundle["config"],
        )
        strategy._models = bundle["models"]
        strategy._category_mappings = bundle.get("category_mappings", {})
        strategy._fitted = True
        return strategy
```
同时删除文件头 `import inspect`（仅此处使用）与原注释块。

- [ ] **Step 3: 验证**

```bash
.venv/bin/python -m py_compile strategies/lgb.py && echo OK
QUANTLAB_DB=/Users/cui/Projects/quantlab/data/ashare.duckdb .venv/bin/python -c "
import sys; sys.path.insert(0, '.')
import numpy as np
from strategies.lgb import _rank_ic_np, MIN_STOCKS_PER_DATE, LGBStrategy
x = np.linspace(-1, 1, 100); y = x + np.random.default_rng(0).normal(0, .01, 100)
print('rank_ic_np:', round(_rank_ic_np(x, y), 4), '| MIN:', MIN_STOCKS_PER_DATE)"
```
Expected: rank_ic_np ≈ 1.0；无 import 错误。

- [ ] **Step 4: Commit**

```bash
git add strategies/lgb.py && git commit -m "refactor: _rank_ic_np 积木迁入 strategies/lgb + LGBStrategy.load 删旧键兼容层（旧折 bundle 已清）"
```

---

### Task 3: config gap1d 降位 + 死函数/注释同步

**Files:**
- Modify: `config.py`（删 MODEL_CONFIGS["gap1d"] 条目及前导注释）、`strategies/labels.py`（删 compute_forward_returns + 头注更新）、`dataset.py`（头注收编名单措辞）、`factors/__init__.py`（模块清单更新）

**Interfaces:**
- Produces: `MODEL_CONFIGS` 仅含 20d/6d/open2d；`_leak_check` 的 `for m in sorted(MODEL_CONFIGS)` 自动变 3 模型 ×14 = 42 项。

- [ ] **Step 1: config.py 删 gap1d 条目**

删除以下整块（`"6d"` 条目之后）：

```python
    # 独立实验模型（2026-08-22 用户要求，暂不接入 score/回测）：
    # 预测隔夜跳空 open[T+1]/close[T]−1（= median_open 窗口(1,1)+close 锚），
    # 因子直接复用 6d 清单（selected_*_gap1d.json 为 6d 清单拷贝，不独立筛选）
    "gap1d": dict(
        label_window=(1, 1),
        label_price="open",
        baseline="close",
        label_buffer=1,
        horizon="label_gap1d",      # 预测列: pred_label_gap1d
    ),
```
并在 MODEL_CONFIGS 上方加一行注释：`# 2026-09-03 用户裁定：gap1d 降位——跳空预测归 ML 因子层（gb_gap1d），主模型仅三（open2d/6d/20d）`。
同文件 ~42 行 baseline 注释中 "factors/baseline_check.py" 字样改为 "（挖矿层 2026-09-03 已删，8 基准 alpha 实现随删，git 史可查）"。

- [ ] **Step 2: labels.py 删 compute_forward_returns 整函数**（唯一调用方 baseline_check 已删），头注第 13 行 "仅供评估器回归门禁（factors/baseline_check.py）使用" 一句删除。

- [ ] **Step 3: dataset.py 头注第 4-5 行** "收编此前五个文件…（run_lgb / factors.select_factors / factors.mining / _leak_check / factors.baseline_check）" 改为 "收编此前各文件复制的装载+过滤链（2026-09-03 简化后消费方 = run_lgb / _leak_check / fold_cv / 临时脚本）"。

- [ ] **Step 4: factors/__init__.py** 模块清单 docstring 删去 select_factors/mining/baseline_check/ic_probe/build_gb_4d/build_gb_30d 行，注明"挖矿/筛选 = tmp 临时脚本纪律（2026-09-03 用户裁定）"。

- [ ] **Step 5: 验证 + Commit**

```bash
.venv/bin/python -m py_compile config.py strategies/labels.py dataset.py factors/__init__.py && \
QUANTLAB_DB=/Users/cui/Projects/quantlab/data/ashare.duckdb .venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from config import MODEL_CONFIGS
assert sorted(MODEL_CONFIGS) == ['20d', '6d', 'open2d'], sorted(MODEL_CONFIGS)
print('MODEL_CONFIGS =', sorted(MODEL_CONFIGS))"
git add -A && git commit -m "refactor: gap1d 降位（主模型收敛为三）+ labels 死函数 + 注释同步（挖矿层已删）"
```

---

### Task 4: pools/spec.py 路径族——权重路径去 fold

**Files:**
- Modify: `pools/spec.py:42-47`

**Interfaces:**
- Produces: `model_dir(self) -> Path`（无 fold 参数）、`lgb_model_path(self, model) -> Path`（无 fold 参数）；`lgb_predictions_path`/`lgb_predictions_meta_path`/`backtest_dir` 的 fold 分支**保留**（折评估证据）。

- [ ] **Step 1: 替换 model_dir/lgb_model_path**

```python
    # ---- 路径族（原 config 池路径 helper 迁入；微盘路径逐字节不变）----
    # 2026-09-03 权重单文件纪律：折训练直接覆盖主权重路径，无折模型目录；
    # 折预测/meta/回测仍按折分目录（评估证据，非权重）。
    def model_dir(self) -> Path:
        return ROOT / "models" / self.name

    def lgb_model_path(self, model: str = "20d") -> Path:
        return self.model_dir() / f"lgb_{model}.joblib"
```

- [ ] **Step 2: 全仓核查 fold= 调用方**

```bash
grep -rn "lgb_model_path\|model_dir(" --include='*.py' . | grep -v '\.venv' | grep -v tmp/
```
Expected: `run_lgb.py`（`spec.lgb_model_path(model, fold=fold)` 一处——Task 5 改）、`pools/spec.py` 自身（`_describe` 的 `spec.model_dir()` 无参调用 ✓）、forecast_display（无参主路径 ✓）。

- [ ] **Step 3: 验证 + Commit**

```bash
.venv/bin/python -m py_compile pools/spec.py && QUANTLAB_POOL=mainboard_all .venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from pools.spec import get_pool
s = get_pool('mainboard_all')
print(s.lgb_model_path('6d'))"
# Expected: .../models/mainboard_all/lgb_6d.joblib
git add pools/spec.py && git commit -m "refactor: 权重路径去 fold 分支——单权重文件纪律（折训练覆盖主路径）"
```

---

### Task 5: run_lgb.py 折接缝——读主清单、权重写主路径

**Files:**
- Modify: `run_lgb.py:100-132`（fold 分支与 selected_path）、`:315-317`（model_path）

**Interfaces:**
- Consumes: Task 4 的 `lgb_model_path(model)`（无 fold）。
- Produces: 折模式 = 主清单 + 主权重路径 + 折预测/meta 路径（不变）。

- [ ] **Step 1: selected_path 统一主清单**（替换 116-132 行的 fold/非 fold 双分支）

```python
    # 折模式与主窗口同读主清单（2026-09-03 用户裁定：折清单机制退场，
    # 固定清单折；json 的 train_start/train_end 字段为筛选存档，与折窗无关）
    selected_path = (Path(__file__).resolve().parent / "factors"
                     / f"selected_{spec.name}_{model}.json")
    if selected_path.exists():
        selected_data = json.loads(selected_path.read_text())
        use_factors = selected_data["selected_factors"]
        print(f"  Using {len(use_factors)} pre-selected factors from {selected_path}")
    else:
        use_factors = SELECTED_FACTORS
        print(f"  WARNING: {selected_path.name} not found, falling back to full SELECTED_FACTORS")
```

- [ ] **Step 2: model_path 去 fold**（315-317 行）

```python
    model_path = spec.lgb_model_path(model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    strategy.save(model_path)
    print(f"  model saved: {model_path}"
          f"{'（折模式：覆盖主权重，单文件不留档）' if fold else ''}")
```

- [ ] **Step 3: 验证 + Commit**

```bash
.venv/bin/python -m py_compile run_lgb.py && grep -n "factors.*folds" run_lgb.py
# Expected: 无输出（折清单路径引用清零）
git add run_lgb.py && git commit -m "refactor: run_lgb 折模式读主清单 + 权重覆盖主路径（折清单机制退场）"
```

---

### Task 6: build_gb_gap1d scoped 模式（折同步训练接缝）

**Files:**
- Modify: `factors/build_gb_gap1d.py`

**Interfaces:**
- Consumes: `strategies.lgb.buffered_train_end(all_dates, boundary, label_buffer)`。
- Produces: CLI `--cutoff YYYY-MM-DD --through YYYY-MM-DD`；scoped 模式返回并落盘终态权重 `models/{pool}/gb_gap1d.joblib`（bundle: `{"model", "features", "cutoff"}`）；无参调用行为与现状逐字节一致（全局 OOF）。

**防泄漏规则（spec 铁律）**：段起点 ≥ cutoff 的段不重训；在跨过 cutoff 的第一段做一次"满量冻结拟合"（训练数据 < `buffered_train_end(uniq, cutoff, LABEL_BUFFER)`），此后冻结模型推理到 through——测试窗特征不携带 [cutoff, …) 任何标签信息。

- [ ] **Step 1: walk_forward_oof 加 scoped 参数**（整函数替换）

```python
from strategies.lgb import buffered_train_end   # 文件头 import 区


def walk_forward_oof(X: pd.DataFrame, y: pd.Series,
                     cutoff: str | None = None,
                     through: str | None = None) -> tuple[pd.Series, object | None]:
    """cutoff=None：全局 OOF（行为不变）。scoped 模式（fold_cv 折同步）：
    段起点 >= cutoff 不再重训；跨 cutoff 首段做满量冻结拟合（数据 <
    cutoff − LABEL_BUFFER，复用 buffered_train_end 单源），冻结模型推理到
    through——折测试窗特征零测试期信息。返回 (预测 Series, 终态模型)。"""
    dates = X.index.get_level_values("date")
    uniq_all = dates.unique().sort_values()
    cutoff_ts = pd.Timestamp(cutoff) if cutoff else None
    uniq = uniq_all if through is None else uniq_all[uniq_all <= pd.Timestamp(through)]
    n = len(uniq)
    preds_parts = []
    last = None
    frozen = False
    i = MIN_TRAIN_DAYS
    while i < n:
        cut_ts = uniq[max(i - LABEL_BUFFER, 0)]
        seg_dates = set(uniq[i:i + CADENCE])
        seg_mask = dates.isin(seg_dates)
        yt = y.reindex(X.index)
        if cutoff_ts is None or uniq[i] < cutoff_ts:
            train_mask = np.asarray(dates < cut_ts)
            ok = train_mask & yt.notna().to_numpy()
            if ok.sum() > 10000:
                m = xgb.XGBRegressor(**XGB_PARAMS)
                m.fit(X.loc[ok].values, yt[ok].values)
                last = m
        elif not frozen:
            # 满量冻结拟合：用尽截止前全部安全数据，此后模型冻结
            cut_safe = buffered_train_end(list(uniq_all), cutoff_ts, LABEL_BUFFER)
            ok = np.asarray(dates < cut_safe) & yt.notna().to_numpy()
            if ok.sum() > 10000:
                m = xgb.XGBRegressor(**XGB_PARAMS)
                m.fit(X.loc[ok].values, yt[ok].values)
                last = m
            frozen = True
        if last is not None:
            Xs = X.loc[seg_mask]
            if len(Xs):
                preds_parts.append(pd.Series(last.predict(Xs.values), index=Xs.index))
        i += CADENCE
    preds = pd.concat(preds_parts) if preds_parts else pd.Series(dtype=float)
    return preds, last
```

- [ ] **Step 2: main() 加 CLI 与终态权重落盘**

argparse 增参：

```python
    ap.add_argument("--cutoff", default=None,
                    help="折同步 scoped 模式：训练截止（折 test_start）；缺省=全局 OOF")
    ap.add_argument("--through", default=None,
                    help="scoped 模式产出上界（折 test_end）")
    if (args.cutoff is None) != (args.through is None):
        ap.error("--cutoff 与 --through 必须成对使用")
```

调用与落盘（替换 `preds = walk_forward_oof(X, y)` 一行及其后写表段尾部）：

```python
    preds, final_model = walk_forward_oof(X, y, cutoff=args.cutoff, through=args.through)
    ...  # IC 评估打印与 store.update_columns 写表：原样保留
    if args.cutoff is not None and final_model is not None:
        import joblib
        wpath = spec.model_dir() / f"{COL}.joblib"
        joblib.dump({"model": final_model, "features": FORMULA_FACTORS,
                     "cutoff": args.cutoff}, wpath)
        print(f"  终态权重（折同步，跑后保留供实盘）: {wpath}")
```
（`preds` 现为二元组解包——评估段的 `yv = y.reindex(preds.index)` 等下游用法不变。）

- [ ] **Step 3: 验证（无参模式回归 + scoped 冒烟）**

```bash
.venv/bin/python -m py_compile factors/build_gb_gap1d.py && echo COMPILE_OK
QUANTLAB_DB=/Users/cui/Projects/quantlab/data/ashare.duckdb .venv/bin/python -m factors.build_gb_gap1d \
  --pool mainboard_all --cutoff 2025-06-01 --through 2025-12-31 2>&1 | tail -8
```
Expected: OOS 预测行数正常、写入 matched 行数 > 0、打印"终态权重 … gb_gap1d.joblib"。**此步写库**——先做单写者检查。
（F6 test 上界用 2025-12-31 仅冒烟，正式折窗由 Task 7 驱动。）

- [ ] **Step 4: Commit**

```bash
git add factors/build_gb_gap1d.py && git commit -m "feat: gb_gap1d scoped 折同步模式——--cutoff/--through + 满量冻结拟合防泄漏 + 终态权重落盘"
```

---

### Task 7: fold_cv 重写——ML 同步 + 三主模型 + ICIR 报告

**Files:**
- Modify: `fold_cv.py`（整文件重写为下述结构）

**Interfaces:**
- Consumes: Task 6 的 scoped CLI；Task 5 的 run_lgb 折模式；`config.MODEL_CONFIGS`（3 模型）。
- Produces: 每折顺序 = ML 同步训练 → `run_lgb --model all --fold F*` → leak_checks → 折回测；汇总报告 per_fold 各含 `model_ic`（每模型 mean_ic/ir/n_periods）+ 组合收益/夏普，顶层含 `mean_model_ic`。

- [ ] **Step 1: 整文件重写**

```python
# -*- coding: utf-8 -*-
"""
Rolling fold CV driver（2026-09-03 简化重写，spec: mb1-simplification）。

每折顺序：ML 因子同步训练（防泄漏 scoped）→ 三主模型训练（固定主清单，
权重覆盖主路径）→ 泄漏断言 → market 回测。汇总 = 每模型 test IC/ICIR +
组合收益/夏普，写 data/fold_cv_report_{pool}.json。

防泄漏：run_lgb 的 label_buffer（复用）；ML 因子训练截止 = 折 test_start
（build_gb_gap1d --cutoff，内部自带标签 buffer 回退与测试窗冻结推理）。
折清单机制已退场（2026-09-03 用户裁定：固定主清单折，筛选走 tmp 临时脚本）。

全程 DB 写仅限本池因子表 ML 列（store.update_columns）——避开 cron 窗口。

Usage:
    python fold_cv.py --pool mainboard_all            # 7 折全链
    python fold_cv.py --pool mainboard_all --folds F7 # 单折（冒烟）
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FOLDS, FOLD_TRAIN_START, MODEL_CONFIGS, get_fold
from pools.spec import get_pool

PY = sys.executable
ROOT = Path(__file__).resolve().parent
RF = 0.025  # 与 backtest/run_lgb.py 同口径
# ML 因子列 -> scoped 构建器模块（折同步训练；nn_gap1d 无 scoped 模式，
# 引用即 fail-fast——本简化边界 = mainboard_all，其清单不含 nn）
ML_BUILDERS = {"gb_gap1d": "factors.build_gb_gap1d"}


def run(cmd: list[str], log_path: Path) -> None:
    t0 = time.time()
    with open(log_path, "a") as f:
        f.write(f"\n{'=' * 70}\n$ {' '.join(cmd)}\n")
        f.flush()
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"command failed (exit {r.returncode}): {' '.join(cmd)} — see {log_path}")
    print(f"    done in {time.time() - t0:.0f}s")


def ml_factors_in_lists(spec) -> list[str]:
    """主清单实际引用的 ML 因子列（gb_/nn_ 前缀自动发现，换清单自动跟随）。"""
    cols: set[str] = set()
    for m in sorted(MODEL_CONFIGS):
        p = ROOT / "factors" / f"selected_{spec.name}_{m}.json"
        cols |= {f for f in json.loads(p.read_text())["selected_factors"]
                 if f.startswith(("gb_", "nn_"))}
    return sorted(cols)


def leak_checks(fid: str, spec) -> None:
    """折产物泄漏断言：训练截止<test_start；测试窗=折定义；预测范围⊆折窗。"""
    test_start, test_end = get_fold(fid)
    ts, te = pd.Timestamp(test_start), pd.Timestamp(test_end)
    for m in sorted(MODEL_CONFIGS):
        meta = json.loads(spec.lgb_predictions_meta_path(m, fold=fid).read_text())
        assert pd.Timestamp(meta["train_end"]) < ts, \
            f"{fid}/{m}: meta train_end={meta['train_end']} >= test_start"
        assert meta["test_start"] == test_start and meta["test_end"] == test_end, \
            f"{fid}/{m}: meta test window {meta['test_start']}~{meta['test_end']} != fold def"
        pred = pd.read_parquet(spec.lgb_predictions_path(m, fold=fid))
        dts = pred.index.get_level_values("date")
        assert dts.min() >= ts and dts.max() <= te, \
            f"{fid}/{m}: predictions [{dts.min()}~{dts.max()}] outside fold window"


def fold_model_ic(fid: str, spec) -> dict:
    """每模型折内 test IC/ICIR（读折 meta）。"""
    out = {}
    for m in sorted(MODEL_CONFIGS):
        t = json.loads(spec.lgb_predictions_meta_path(m, fold=fid)
                        .read_text())["results"]["test_ic"]
        out[m] = {"mean_ic": t.get("mean_ic"), "ir": t.get("ir"),
                  "n_periods": t.get("n_periods", 0)}
    return out


def fold_metrics(fid: str, exec_label: str, spec) -> dict:
    """从折回测 CSV 计算：总收益/夏普/回撤/往返/基准 + 每模型 IC/ICIR。"""
    # （原实现逐行保留——equity/夏普/回撤/n_days/trades 往返统计/基准段全部不动；
    #  唯一改动 = return 前插入 model_ic）
    out["model_ic"] = fold_model_ic(fid, spec)
    return out
```

main() 主体改动（其余打印/汇总骨架保留）：

```python
    parser.add_argument("--folds", default=",".join(FOLDS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pool", default=None)
    # --skip-train / --skip-select 已删（2026-09-03：旧折不用兼容、折清单退场）
    ...
    ml_cols = ml_factors_in_lists(spec)
    for f in ml_cols:
        if f not in ML_BUILDERS:
            raise RuntimeError(
                f"主清单引用 ML 因子 {f}，但无 scoped 构建器（nn_gap1d 未适配折同步；"
                f"本简化边界 = mainboard_all）")
    print(f"ML 因子折同步: {ml_cols or '无'}")

    for fid in fids:
        ts, te = get_fold(fid)
        print(f"\n===== {fid} =====")
        for f in ml_cols:
            run([PY, "-m", ML_BUILDERS[f], "--pool", spec.name,
                 "--cutoff", str(ts)[:10], "--through", str(te)[:10]], log_path)
        run([PY, "run_lgb.py", "--model", "all", "--fold", fid, *pool_args], log_path)
        leak_checks(fid, spec)
        print("    leak checks passed")
        run([PY, "-m", "backtest.run_lgb", "--fold", fid, *pool_args], log_path)
```

汇总段新增（report[ex] 字典内追加）：

```python
            "mean_model_ic": {
                m: {"mean_ic": float(np.nanmean([rows[fid]["model_ic"][m]["mean_ic"]
                                                 for fid in fids])),
                    "ir": float(np.nanmean([rows[fid]["model_ic"][m]["ir"]
                                            for fid in fids]))}
                for m in sorted(MODEL_CONFIGS)},
```

dry-run 打印同步为四步新序列（ML 同步 → 训练 → 断言 → 回测）。

- [ ] **Step 2: 验证**

```bash
.venv/bin/python -m py_compile fold_cv.py && \
QUANTLAB_DB=/Users/cui/Projects/quantlab/data/ashare.duckdb .venv/bin/python fold_cv.py --pool mainboard_all --dry-run
```
Expected: 打印 7 折计划 + "ML 因子折同步: ['gb_gap1d']"。

- [ ] **Step 3: Commit**

```bash
git add fold_cv.py && git commit -m "feat: fold_cv 重写——ML 因子折同步训练 + 固定主清单 + 权重覆盖主路径 + ICIR 汇总（--skip-train/--skip-select 删）"
```

---

### Task 8: AGENTS.md 叙事同步

**Files:**
- Modify: `AGENTS.md`（横幅 + 模块树 + 挖矿工作流 + 折 CV 章节）

- [ ] **Step 1: 逐节更新**（要点清单）

1. 横幅"现役清单"行：删"gap1d 0.1747"字样与三 IC 中 gap1d 表述 → 三模型口径（open2d 11 / 6d 10 / 20d 32）。
2. 横幅加一行：**2026-09-03 简化裁定**——挖矿层全删（挖矿/筛选一律 tmp 临时脚本，积木 `strategies.lgb._rank_ic_np`）；主模型收敛为三（gap1d 降位，跳空预测归 ML 因子层）；ML 因子随折同步训练（防泄漏：截止=折 test_start，测试窗冻结推理，权重 `models/{pool}/gb_gap1d.joblib` 跑后保留供实盘）；全模型权重单文件不留档、折训练直接覆盖主路径；DB 七根保留列不 DROP。
3. "三模型 LightGBM 回归"横幅行：删"（+gap1d 独立实验模型）"。
4. 模块结构树：删 mining/select_factors/baseline_check/ic_probe/build_gb_4d/build_gb_30d/folds/ 行；`fold_cv.py` 描述改为"7 折：ML 同步+训练+断言+回测，ICIR+收益+夏普"；`run_lgb.py` 描述去 gap1d；`strategies/lgb.py` 描述加"_rank_ic_np 挖矿积木"。
5. "挖矿工作流（新因子主路径）"整节替换为：**临时脚本纪律**——挖矿/筛选无正式入口；tmp/ 脚本积木（`dataset.load_factors`、`strategies.lgb._rank_ic_np`、`factors.store`）；换清单 = tmp 重筛 → 改唯一 json（note 留痕）→ `run_lgb` + 七折即新基线。
6. "折 CV 验证框架"节：删"每折独立重跑筛选"与折清单表述 → 固定主清单 + ML 同步训练 + 权重覆盖语义；`--skip-train` 字样删除。
7. Key Decisions #6（折 CV）与 #9（gap1d）段同步；"池化边界"入口清单删 mining/select_factors/baseline_check。

- [ ] **Step 2: Commit**

```bash
git add AGENTS.md && git commit -m "docs: AGENTS 同步简化后架构——三主模型/挖矿临时脚本纪律/ML 折同步/权重单文件"
```

---

### Task 9: 冒烟验证（四道门）

**Files:** 无新改动（验证 Task 1-8）

- [ ] **Step 1: 单写者检查 + cron 四步契约核查**（后续步骤会写库）

```bash
date '+%H:%M'; ps aux | grep -E 'data\.pull|factors\.update' | grep -v grep || echo clear
.venv/bin/python -m py_compile data/pull.py factors/update.py factors/build_nn_gap1d.py forecast_display/generate_lgb.py && ls forecast_display/generate_lgb.py && echo CRON_CONTRACT_OK
```

- [ ] **Step 2: 主窗口三模型重训 + 泄漏断言**

```bash
QUANTLAB_DB=/Users/cui/Projects/quantlab/data/ashare.duckdb .venv/bin/python run_lgb.py --model all --pool mainboard_all 2>&1 | tail -12
QUANTLAB_DB=/Users/cui/Projects/quantlab/data/ashare.duckdb .venv/bin/python _leak_check.py 2>&1 | tail -5
```
Expected: Summary 只有三行（20d/6d/open2d）；leak check 42 项 0 失败。

- [ ] **Step 3: F7 单折全链（含 ML 同步，~15 分钟）**

```bash
QUANTLAB_DB=/Users/cui/Projects/quantlab/data/ashare.duckdb .venv/bin/python fold_cv.py --pool mainboard_all --folds F7 2>&1 | tail -25
```
Expected: ML 同步 done → 训练 done → leak checks passed → 回测 done → 汇总表含 model_ic（三模型 IC/ICIR）+ 收益/夏普；`models/mainboard_all/gb_gap1d.joblib` 存在。

- [ ] **Step 4: 主窗回测跑通**

```bash
QUANTLAB_DB=/Users/cui/Projects/quantlab/data/ashare.duckdb .venv/bin/python -m backtest.run_lgb --pool mainboard_all 2>&1 | tail -8
```
Expected: 正常出三件套与超额数字（新基线，无对拍门）。

- [ ] **Step 5: Commit（冒烟产物如有跟踪变化）+ 记录**

```bash
git add -A && git commit -m "chore: 简化后冒烟全绿——三模型重训 + 42 项泄漏断言 + F7 全链（ML 同步）+ 主窗回测" || echo "no tracked changes"
```

---

### Task 10: 七折新基线 + 收尾存档

**Files:**
- Modify: `AGENTS.md`（横幅数字）、agent memory（`bench-mainboard-all-rewrite-status.md` + `MEMORY.md`）

- [ ] **Step 1: 完整七折重跑（后台，~80-100 分钟；含 ML 同步写库，避开 21:05 窗口）**

```bash
QUANTLAB_DB=/Users/cui/Projects/quantlab/data/ashare.duckdb nohup .venv/bin/python fold_cv.py --pool mainboard_all > data/fold_cv_run_mainboard_all.console 2>&1 &
```
完成后检查 `data/fold_cv_report_mainboard_all.json`：7 折 per_fold 均含 model_ic 三模型 + 收益/夏普；顶层 mean_model_ic 存在。

- [ ] **Step 2: AGENTS 横幅写入新基线数字**（三模型主窗 IC + 七折 mean ICIR/收益/夏普/超额，标注"2026-09-03 简化后基线"）。

- [ ] **Step 3: memory 更新**：rewrite-status 加"简化落地"段（提交链、新基线数字、开放项清账：折代际差随机制退场关闭、合并裁定升级为"用户已宣布测试正确即覆盖并入 main"）；MEMORY.md 索引行同步。

- [ ] **Step 4: 终验与提交**

```bash
git add -A && git commit -m "chore: 简化后七折新基线入库 + AGENTS/memory 收尾存档"
git log --oneline -12
```
并向用户汇报：冒烟四门结果 + 七折新基线 + 简化前后体量对照（py 33→26、入口清单），等待其测试与合并指令。
