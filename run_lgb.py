"""
Dual-regression LightGBM walk-forward training (bench 2026-08, spec §3.3).

One LGBMRegressor per model (20d / 6d). Label = median open return over the
model's forward window, next_open baseline (compute_median_open). Fixed
test-set protocol identical to the legacy classifier: train once on data
before TEST_START stepped back label_buffer trading days, predict the whole
test period.

Usage:
    python run_lgb.py                # train all models (20d, 6d)
    python run_lgb.py --model 20d
    python run_lgb.py --model 6d
"""
from __future__ import annotations

import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (DB_PATH, POOL_NAME, get_pool_codes, SELECTED_FACTORS,
                    MODEL_CONFIGS, get_lgb_model_path, get_lgb_predictions_path,
                    get_lgb_predictions_meta_path, FOLDS, get_fold)

from strategies import LGBStrategy, walk_forward, rank_ic, ic_summary
from strategies.base import buffered_train_end
from strategies.labels import compute_median_open, compute_nextopen_limit_mask


# --- config ---
TRAIN_START = pd.Timestamp("2020-01-01")
TEST_START = pd.Timestamp("2025-06-01")
TEST_END = pd.Timestamp("2026-06-01")
WARMUP_DAYS = 90
# 输出校准：训练窗内留出尾段（交易日数）估计 out-of-sample 收缩斜率，见下方 calibration 注释
CALIB_TAIL_DAYS = 60
TRAIN_WINDOW = 252
MIN_TRAIN = 252

# 超参沿用分类时代调参（num_leaves/min_child/colsample 均为分类调出），
# 回归首跑结果即基线，之后按回归目标重调（spec §3.3 超参注意）
LGB_KWARGS = dict(
    # 2026-08-22 用户裁定：L1 损失（条件中位数估计器）——与输出校准的 L1 斜率
    # 同口径自洽（L2 均值模型 × L1 中位数校准会因右偏标签系统性压小斜率），
    # 且对肥尾标签更稳（早停 eval 同步变为 l1）
    objective="regression_l1",
    num_leaves=8,
    max_depth=4,
    learning_rate=0.05,
    n_estimators=3000,
    min_child_samples=2000,
    reg_alpha=3.0,
    reg_lambda=3.0,
    subsample=0.6,
    subsample_freq=1,
    colsample_bytree=0.3,
    model_type="regressor",
    categorical_feature=["sw_l3"],
    early_stopping=True,
    validation_fraction=0.10,
    n_iter_no_change=50,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)


def load_industry_sw_l3(con: duckdb.DuckDBPyConnection) -> tuple[pd.Series, dict[str, int]]:
    """Load SW L3 codes for all stocks, encode into deterministic integers."""
    df = con.execute(
        "SELECT code, sw_l3_code FROM industry WHERE sw_l3_code IS NOT NULL"
    ).fetchdf()
    if df.empty:
        return pd.Series(dtype=int), {}

    categories = sorted(df["sw_l3_code"].astype(str).unique())
    mapping = {code: i for i, code in enumerate(categories)}
    codes = df["sw_l3_code"].map(mapping).fillna(-1).astype(int)
    return pd.Series(codes.values, index=df["code"], name="sw_l3"), mapping


def load_factors(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    pool_codes = get_pool_codes()
    placeholders = ",".join(["?"] * len(pool_codes))
    query = f"SELECT * FROM factor_values WHERE code IN ({placeholders})"
    df = con.execute(query, pool_codes).fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "code"]).sort_index()
    return df


def load_kline(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    pool_codes = get_pool_codes()
    placeholders = ",".join(["?"] * len(pool_codes))
    query = f"SELECT code, date, open, close FROM daily_kline WHERE code IN ({placeholders}) ORDER BY code, date"
    return con.execute(query, pool_codes).fetchdf()


def load_delist_info(con: duckdb.DuckDBPyConnection) -> dict[str, pd.Timestamp]:
    try:
        df = con.execute("SELECT code, delist_date FROM delist_info").fetchdf()
        if df.empty:
            return {}
        return {r["code"]: pd.Timestamp(r["delist_date"]) for _, r in df.iterrows()}
    except Exception:
        return {}


def decile_analysis(pred: pd.Series, label: pd.Series) -> tuple[list[float], float]:
    """Per-date decile buckets -> mean label per decile + Spearman(decile, mean).

    Monotonicity reference: a good ranking signal should show roughly
    increasing mean label from decile 0 (lowest pred) to decile 9.
    """
    df = pd.DataFrame({"p": pred, "y": label}).dropna()
    if df.empty:
        return [], float("nan")
    pct = df.groupby(level="date")["p"].rank(pct=True)
    dec = np.minimum((pct * 10).astype(int), 9)
    means = df.groupby(dec)["y"].mean().tolist()
    rho = float(spearmanr(range(len(means)), means).statistic) if len(means) == 10 else float("nan")
    return means, rho


def train_model(
    model: str,
    factors_raw: pd.DataFrame,
    label: pd.Series,
    st_series: pd.Series | None,
    limit_mask: pd.Series,
    delist_info: dict[str, pd.Timestamp],
    industry_sw_l3: pd.Series,
    sw_l3_mapping: dict[str, int],
    fold: str | None = None,
) -> dict:
    cfg = MODEL_CONFIGS[model]
    h = cfg["horizon"]
    label_buffer = cfg["label_buffer"]

    # 折模式：test 边界取折定义；TRAIN_START 恒 2020（折训练起点与常量一致）
    if fold:
        fold_start, fold_end = get_fold(fold)
        test_start, test_end = pd.Timestamp(fold_start), pd.Timestamp(fold_end)
    else:
        test_start, test_end = TEST_START, TEST_END

    print(f"\n{'=' * 60}")
    print(f"=== Model {model}: label T+{cfg['label_window'][0]}..T+{cfg['label_window'][1]} "
          f"open median, baseline={cfg['baseline']}, buffer={label_buffer} "
          f"{'| fold=' + fold if fold else ''} ===")
    print(f"{'=' * 60}")

    t0 = time.time()

    # ---- factor set: per-model selected list ----
    # 折模式强制读折专属筛选清单（防筛选泄漏：折筛选不得见过折内及以后数据）
    if fold:
        selected_path = (Path(__file__).resolve().parent / "factors" / "folds" / fold
                         / f"selected_{POOL_NAME}_{model}.json")
        if not selected_path.exists():
            raise FileNotFoundError(
                f"fold {fold} selected list not found: {selected_path}\n"
                f"Run first: python -m factors.select_factors --model {model} --fold {fold}")
    else:
        selected_path = Path(__file__).resolve().parent / "factors" / f"selected_{POOL_NAME}_{model}.json"
    if selected_path.exists():
        selected_data = json.loads(selected_path.read_text())
        use_factors = selected_data["selected_factors"]
        print(f"  Using {len(use_factors)} pre-selected factors from {selected_path}")
    else:
        use_factors = SELECTED_FACTORS
        print(f"  WARNING: {selected_path.name} not found, falling back to full SELECTED_FACTORS")

    available = [f for f in use_factors if f in factors_raw.columns]
    missing = [f for f in use_factors if f not in factors_raw.columns]
    if missing:
        print(f"  WARNING: {len(missing)} selected factors not in DB: {missing}")
    factor_cols = available

    X = factors_raw[factor_cols].copy()
    if not industry_sw_l3.empty:
        idx_codes = X.index.get_level_values("code")
        X["sw_l3"] = idx_codes.map(industry_sw_l3).fillna(-1).astype(int)
        if "sw_l3" not in factor_cols:
            factor_cols = factor_cols + ["sw_l3"]

    print(f"  factors: {len(factor_cols)} available")

    # ---- label (regression target, continuous) ----
    y = label.to_frame(h)

    # ---- align + train window ----
    common = X.index.intersection(y.index)
    X, y = X.loc[common], y.loc[common]

    mask = y.notna().all(axis=1)
    X, y = X.loc[mask], y.loc[mask]

    date_level = X.index.get_level_values("date")
    mask = date_level >= TRAIN_START
    X, y = X.loc[mask], y.loc[mask]

    print(f"  aligned samples: {len(X)}")
    print(f"  date range: {date_level.min().date()} ~ {date_level.max().date()}")

    # ---- 训练集排斥（2026-08-24 语义修复：仅从训练剔除，预测行集保持全量）----
    # ① ST/退市/次日开盘封板观测不进训练（原实现整行删除，预测 parquet 连带
    #    缺这些行——回测候选集被 T+1 信息条件化、回避次日开盘跌停的崩盘股，
    #    收益乐观偏）；现在预测照常输出，下游（回测执行层/报告/IC）各自过滤。
    # ② 标签远引用越界：标签按个股自身交易行前移，而 label_buffer 按池轴
    #    回退——停牌股的 T+e0 可落到测试窗内（实测 2025-04-29 有 25 只）。
    #    按"标签窗口内最远实际引用日 < test_start"精确判定（在 X 的行序上
    #    近似 kline 行序，逐股同源）。
    if st_series is not None:
        st_mask = st_series.reindex(X.index, fill_value=False)
    else:
        st_mask = pd.Series(False, index=X.index)
    idx_date = X.index.get_level_values("date")
    idx_code = X.index.get_level_values("code")
    delist_series = pd.Series(delist_info)
    delist_dates = idx_code.map(delist_series)
    delist_mask = pd.Series(idx_date >= delist_dates.values, index=X.index).fillna(False)

    lm = limit_mask.reindex(X.index, fill_value=False)

    date_s = pd.Series(idx_date, index=X.index)
    e0 = cfg["label_window"][1]
    far_max = pd.concat(
        [date_s.groupby(level="code", sort=False).shift(-lag) for lag in range(1, e0 + 1)],
        axis=1).max(axis=1)
    far_cross = (far_max >= test_start).fillna(False)

    train_exclude = (st_mask | delist_mask | lm | far_cross).fillna(False)
    # 计数只统计训练侧（date < test_start）——far_cross 对测试窗行恒真但
    # 对训练无意义（walk_forward 只在 < train_end 上训练），全额计数会虚高
    train_side = pd.Series(idx_date, index=X.index) < test_start
    far_cross_n = int((far_cross & train_side).sum())
    print(f"  train-only exclusions: ST={int(st_mask.sum())} delist={int(delist_mask.sum())} "
          f"limit-next-open={int(lm.sum())} label-far-cross(train-side)={far_cross_n} "
          f"(预测行集不再删行，共 {len(X):,})")

    # ---- strategy ----
    strategy = LGBStrategy(
        factor_names=factor_cols,
        horizons=(h,),
        **LGB_KWARGS,
    )
    if sw_l3_mapping:
        strategy._category_mappings["sw_l3"] = sw_l3_mapping

    # ---- walk-forward (fixed test set) ----
    train_dates_all = sorted(X.index.get_level_values("date").unique())[WARMUP_DAYS:]
    train_end = buffered_train_end(train_dates_all, test_start, label_buffer)
    print(f"  walk-forward: train={train_dates_all[0].date()}~{train_end.date()} "
          f"(label_buffer={label_buffer}), test={test_start.date()}~{test_end.date()}, "
          f"warmup={WARMUP_DAYS}")
    preds = walk_forward(
        strategy,
        X, y,
        train_window=TRAIN_WINDOW,
        min_train=MIN_TRAIN,
        warmup_days=WARMUP_DAYS,
        test_start=test_start,
        test_end=test_end,
        label_buffer=label_buffer,
        train_exclude=train_exclude,
    )

    col = f"pred_{h}"
    results: dict = {}

    # ---- output calibration（2026-08-22 用户裁定：模型输出语义 = 真实相信的到期涨幅）----
    # LGBM 回归 + 正则 + 早停的输出幅度是训练偶然产物（系统性收缩，程度随特征集
    # 漂移），会在 combine 的原始幅度加权里暗中改变有效权重。此处用训练窗内部
    # 留出尾段（不碰测试窗；尾段标签最远只到 buffered train_end 之后、test_start
    # 之前，label_buffer 语义保证）测 out-of-sample 收缩斜率 k（L1 过原点加权
    # 中位数，抗标签肥尾），把输出乘回 k 还原为诚实幅度。k<=0 或无效时退回 1
    # （绝不翻符号）；只缩放本模型输出，融合权重不动。
    calib_dates = [d for d in train_dates_all if d < train_end][-CALIB_TAIL_DAYS:]
    k = 1.0
    if len(calib_dates) >= 20:
        calib_strategy = LGBStrategy(
            factor_names=factor_cols, horizons=(h,), **LGB_KWARGS)
        if sw_l3_mapping:
            calib_strategy._category_mappings["sw_l3"] = sw_l3_mapping
        idx_dates_cal = X.index.get_level_values("date")
        fit_mask = np.asarray(idx_dates_cal < calib_dates[0]) & ~train_exclude.to_numpy()
        calib_strategy.fit(X.loc[fit_mask], y.loc[fit_mask])
        tail_mask = idx_dates_cal.isin(calib_dates)
        x_cal = calib_strategy.predict(X.loc[tail_mask])[col].values
        y_cal = y.loc[tail_mask, h].values
        v = np.isfinite(x_cal) & np.isfinite(y_cal) & (np.abs(x_cal) > 1e-12)
        x_cal, y_cal = x_cal[v], y_cal[v]
        # L1 过原点斜率（用户裁定 2026-08-22）：min_k Σ|y − k·x| 的解是
        # {y/x} 以 |x| 为权的加权中位数；中位数型估计天然抗标签肥尾，不需 winsorize
        ratios = y_cal / x_cal
        wts = np.abs(x_cal)
        order = np.argsort(ratios)
        cw = np.cumsum(wts[order])
        k_raw = float(ratios[order][np.searchsorted(cw, cw[-1] / 2.0)])
        if np.isfinite(k_raw) and k_raw > 0:
            k = k_raw
        if isinstance(preds, pd.DataFrame) and not preds.empty and col in preds.columns:
            preds[col] = preds[col] * k
        # 尾段横截面中位数（校准模型口径）：L1 中位数输出的典型水平为负（池内
        # 典型股票的远期中位收益即为负），回测融合侧用它平移卖出零点——
        # 排序不变，仅把 score<0 的语义从"绝对预期亏损"改为"低于典型预期"
        results["calib_median"] = float(np.median(x_cal * k))
        print(f"  calibration: tail={len(calib_dates)}d "
              f"[{calib_dates[0].date()}~{calib_dates[-1].date()}] slope={k:.3f} "
              f"median={results['calib_median']:+.5f}")
    results["calib_slope"] = k

    # ---- train-set reference (MAE + rank IC) ----
    train_mask = X.index.get_level_values("date") < test_start
    if train_mask.any():
        preds_train = strategy.predict(X.loc[train_mask])
        tr_p = preds_train[col].values
        tr_y = y.loc[train_mask, h].values
        valid = np.isfinite(tr_p) & np.isfinite(tr_y)
        results["train_mae"] = float(np.mean(np.abs(tr_p[valid] - tr_y[valid])))
        tr_ic = rank_ic(preds_train[col], y.loc[train_mask, h])
        tr_s = ic_summary(tr_ic)
        results["train_ic"] = tr_s
        print(f"  train: MAE={results['train_mae']:.4f}  "
              f"IC mean={tr_s['mean_ic']:.4f}  IR={tr_s['ir']:.3f}")

    # ---- test-set evaluation ----
    if isinstance(preds, pd.DataFrame) and not preds.empty and col in preds.columns:
        test_pred = preds[col]
        test_true = y.reindex(preds.index)[h]

        safe = ~limit_mask.reindex(preds.index, fill_value=False)
        if st_series is not None:
            safe = safe & ~st_series.reindex(preds.index, fill_value=False)

        tp = test_pred.loc[safe].values
        tt = test_true.loc[safe].values
        valid = np.isfinite(tp) & np.isfinite(tt)
        mae = float(np.mean(np.abs(tp[valid] - tt[valid])))

        ric = rank_ic(test_pred.loc[safe], test_true.loc[safe])
        s = ic_summary(ric)
        dec_means, dec_rho = decile_analysis(test_pred.loc[safe], test_true.loc[safe])

        n_dates = preds.index.get_level_values("date").nunique()
        print(f"  predictions: {len(preds)} rows over {n_dates} dates ({time.time() - t0:.1f}s)")
        print(f"  TEST  rank IC: mean={s['mean_ic']:.4f}  IR={s['ir']:.3f}  "
              f"hit={s['hit_rate']:.2%}  ({s['n_periods']} dates)")
        print(f"  TEST  MAE={mae:.4f}  decile monotonicity rho={dec_rho:.3f}")
        print(f"  decile mean labels: {[f'{v:+.4f}' for v in dec_means]}")
        print(f"  excluded (limit/ST): {int((~safe).sum())} obs")

        results.update({
            "test_ic": s,
            "test_mae": mae,
            "decile_means": dec_means,
            "decile_rho": dec_rho,
            "n_pred": int(len(preds)),
        })
    else:
        print(f"  predictions: NONE ({time.time() - t0:.1f}s)")

    # ---- persist ----
    model_path = get_lgb_model_path(model, fold=fold)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    strategy.save(model_path)
    print(f"  model saved: {model_path}")

    pred_path = get_lgb_predictions_path(model, fold=fold)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(preds, pd.DataFrame) and not preds.empty:
        preds.to_parquet(pred_path)
        print(f"  predictions saved: {pred_path} ({len(preds)} rows)")

    meta = {
        "model": model,
        "model_type": "LightGBM (regression)",
        "fold": fold,
        "label_fn": "compute_median_open",
        "label_window": list(cfg["label_window"]),
        "baseline": cfg["baseline"],
        "horizons": [h],
        "factor_names": factor_cols,
        "test_start": str(test_start.date()),
        "test_end": str(test_end.date()),
        "train_start": str(train_dates_all[0].date()),
        "train_end": str(train_end.date()),
        "label_buffer": label_buffer,
        "train_exclude_counts": {
            "st": int(st_mask.sum()), "delist": int(delist_mask.sum()),
            "limit_next_open": int(lm.sum()), "label_far_cross": far_cross_n,
        },
        "lgb_kwargs": LGB_KWARGS,
        "results": results,
        "model_path": str(model_path),
        "predictions_path": str(pred_path),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = get_lgb_predictions_meta_path(model, fold=fold)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  meta saved: {meta_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Dual-regression LightGBM training")
    parser.add_argument("--model", default="all",
                        choices=["all"] + sorted(MODEL_CONFIGS),
                        help="train which model (default: all)")
    parser.add_argument("--fold", choices=sorted(FOLDS), default=None,
                        help="滚动折 CV：test 窗=折定义，模型/预测写 fold 路径，"
                             "selected json 强制读 factors/folds/{fid}/")
    args = parser.parse_args()
    models = sorted(MODEL_CONFIGS) if args.model == "all" else [args.model]

    print(f"=== Loading data (pool: {POOL_NAME}, models: {models}"
          f"{', fold: ' + args.fold if args.fold else ''}) ===")
    con = duckdb.connect(str(DB_PATH), read_only=True)

    print("  loading factors ...")
    factors_raw = load_factors(con)

    print("  loading kline ...")
    kline = load_kline(con)

    print("  loading delist_info ...")
    delist_info = load_delist_info(con)
    print(f"  delisted stocks: {len(delist_info)}")

    print("  loading industry sw_l3 ...")
    industry_sw_l3, sw_l3_mapping = load_industry_sw_l3(con)
    print(f"  industry categories: {industry_sw_l3.nunique()}")
    con.close()

    st_series = factors_raw["IsST"].astype(bool) if "IsST" in factors_raw.columns else None
    limit_mask = compute_nextopen_limit_mask(kline, st_series=st_series)

    all_results = {}
    for m in models:
        cfg = MODEL_CONFIGS[m]
        s0, e0 = cfg["label_window"]
        print(f"  computing label {m}: median open T+{s0}..T+{e0}, baseline={cfg['baseline']} ...")
        label = compute_median_open(kline, start_day=s0, end_day=e0, baseline=cfg["baseline"])
        all_results[m] = train_model(
            m, factors_raw, label, st_series, limit_mask, delist_info,
            industry_sw_l3, sw_l3_mapping, fold=args.fold,
        )

    print(f"\n{'=' * 60}")
    print("=== Summary ===")
    print(f"{'=' * 60}")
    for m, r in all_results.items():
        ic = r.get("test_ic", {})
        print(f"  {m}: test IC mean={ic.get('mean_ic', float('nan')):+.4f}  "
              f"IR={ic.get('ir', float('nan')):.3f}  hit={ic.get('hit_rate', float('nan')):.2%}  "
              f"MAE={r.get('test_mae', float('nan')):.4f}  "
              f"decile_rho={r.get('decile_rho', float('nan')):.3f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
