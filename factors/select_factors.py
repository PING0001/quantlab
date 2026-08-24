"""
Factor selection via cluster-first correlation dedup + IC representative pick.

Computes cross-sectional rank IC and pairwise factor correlation in the
training set. 在平均相关矩阵上做 average-linkage 层次聚类（距离 = 1-|corr|，
簇内相关 >= CLUSTER_CORR），每簇选 |IC| 最高者为代表，簇按代表 |IC| 降序
填满 MAX_FACTORS 个名额——相关度定结构（覆盖哪些信号族），IC 只在簇内
选代表（2026-08-21 用户裁定：相关度权重 > IC 权重，取代旧 IC-贪心+0.75 闸门）。

标签 = 目标模型的 compute_median_open（与训练同一构造，spec §3.4）；
每模型一份清单，输出 selected_{pool}_{model}.json。

Usage:
    python -m factors.select_factors --model 20d
    python -m factors.select_factors --model 6d
"""
from __future__ import annotations

import argparse
import sys
import json
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (DB_PATH, POOL_NAME, get_pool_codes, SELECTED_FACTORS,
                    MODEL_CONFIGS, FOLDS, FOLD_TRAIN_START, get_fold)
from strategies.labels import compute_median_open
from pools.membership import union_codes, member_mask
from run_lgb import HISTORY_SINCE   # 成员起点单源（池时点化 2026-08-24）


# 口径说明：2026-08-21 用户裁定筛选与训练对齐——IC 与相关度均自 2020 起
# （此前 2015 长历史口径与 run_lgb 的 2020 训练起点不一致）
TRAIN_START = pd.Timestamp("2020-01-01")
TEST_START = pd.Timestamp("2025-06-01")
MAX_FACTORS = 60
CLUSTER_CORR = 0.7   # 簇优先：average-linkage 距离=1-|corr|，簇内相关 >= 此值合并
EXCLUDE_PREFIXES = ("alpha",)   # 2026-08-21 用户裁定：alpha 开头因子（vnpy 移植 101 个及变体）全部不入筛选
MIN_STOCKS_PER_DATE = 30
MUST_INCLUDE = ["CSI_return_20d"]


def load_factors(con):
    pool_codes = union_codes(since=HISTORY_SINCE)
    placeholders = ",".join(["?"] * len(pool_codes))
    query = f"SELECT * FROM factor_values WHERE code IN ({placeholders})"
    df = con.execute(query, pool_codes).fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "code"]).sort_index()
    return df


def load_kline(con):
    pool_codes = union_codes(since=HISTORY_SINCE)
    placeholders = ",".join(["?"] * len(pool_codes))
    query = f"SELECT code, date, open, close FROM daily_kline WHERE code IN ({placeholders}) ORDER BY code, date"
    return con.execute(query, pool_codes).fetchdf()


def compute_model_label(kline, cfg: dict) -> pd.Series:
    """Label column matching the training target construction exactly."""
    s, e = cfg["label_window"]
    return compute_median_open(kline, start_day=s, end_day=e, baseline=cfg["baseline"])


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


def main():
    parser = argparse.ArgumentParser(description="Factor selection via IC ranking + correlation filtering")
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), default="20d",
                        help="模型（决定标签窗口/基准价，spec §3.4）")
    parser.add_argument("--fold", choices=sorted(FOLDS), default=None,
                        help="滚动折 CV：筛选窗口终点=折 test_start，输出 factors/folds/{fid}/")
    args = parser.parse_args()
    cfg = MODEL_CONFIGS[args.model]
    horizon_name = cfg["horizon"]

    # 折模式：窗口 [FOLD_TRAIN_START, 折 test_start)；非折模式：模块常量
    test_start = pd.Timestamp(get_fold(args.fold)[0]) if args.fold else TEST_START
    train_start = pd.Timestamp(FOLD_TRAIN_START) if args.fold else TRAIN_START

    print(f"Pool: {POOL_NAME} | model: {args.model} "
          f"(label T+{cfg['label_window'][0]}..T+{cfg['label_window'][1]}, baseline={cfg['baseline']})"
          f"{' | fold=' + args.fold if args.fold else ''}")
    con = duckdb.connect(str(DB_PATH), read_only=True)

    print("Loading factors ...")
    factors_raw = load_factors(con)
    available = [f for f in SELECTED_FACTORS
                 if f in factors_raw.columns and not f.startswith(EXCLUDE_PREFIXES)]
    missing = [f for f in SELECTED_FACTORS if f not in factors_raw.columns]
    if missing:
        print(f"  WARNING: {len(missing)} factors missing: {missing[:10]}...")
    factors = factors_raw[available].copy()
    factor_names = available
    n_factors = len(factor_names)
    print(f"  {n_factors} factors available")

    date_level = factors.index.get_level_values("date")
    # 2026-08-24 修复：筛选 IC 窗口补 label_buffer——窗口末 e0 个交易日的标签
    # 引用 test_start 之后的开盘价（原实现裸取 [start, test_start)，尾部 IC
    # 吸收测试窗价格信息）。与训练侧 buffered_train_end 同一机制。
    from strategies.base import buffered_train_end
    panel_dates = sorted(date_level.unique())
    buf_end = buffered_train_end(panel_dates, test_start, cfg["label_buffer"])
    train_mask = (date_level >= train_start) & (date_level < buf_end)
    factors = factors.loc[train_mask]
    # 池时点化：IC 只在当期档成员行上算（横截面口径 = 各档当时的池）
    mm = member_mask(factors.index.get_level_values("date"),
                     factors.index.get_level_values("code"))
    factors = factors.loc[mm]
    print(f"  Training range: {factors.index.get_level_values('date').min().date()} ~ "
          f"{factors.index.get_level_values('date').max().date()} "
          f"(IC 窗口终点回退 label_buffer={cfg['label_buffer']} 至 {buf_end.date()})")
    print(f"  Training rows: {len(factors)}")

    print("Loading kline ...")
    kline = load_kline(con)
    con.close()

    print("Computing labels ...")
    labels = compute_model_label(kline, cfg).to_frame(horizon_name)

    common = factors.index.intersection(labels.index)
    factors = factors.loc[common]
    labels = labels.loc[common]

    exclude = pd.Series(False, index=labels.index)
    exclude |= ~labels.notna().all(axis=1)
    if "IsST" in factors.columns:
        st_series = factors["IsST"].reindex(labels.index, fill_value=False).astype(bool)
        exclude |= st_series
    # 旧 (labels == -1.0) 排除已删：compute_median_open 无 -1.0 填充（spec §3.2，
    # 旧填充是死代码）；窗口缺失行为 NaN，由上一行 notna 排除覆盖

    factors = factors.loc[~exclude]
    labels = labels.loc[~exclude]

    common = factors.index.intersection(labels.index)
    factors = factors.loc[common]
    labels = labels.loc[common]

    n_dates = factors.index.get_level_values("date").nunique()
    print(f"  Aligned: {len(factors)} rows, {n_dates} dates")

    # ---- pre-extract to numpy, clip extremes ----
    print("Preparing arrays ...")
    F_all = factors[factor_names].values.astype(np.float64)
    F_all[~np.isfinite(F_all)] = np.nan
    label_cols = [horizon_name]
    L_all = labels[label_cols].values.astype(np.float64)

    dates_arr = factors.index.get_level_values("date").values
    unique_dates, start_idx, counts = np.unique(dates_arr, return_index=True, return_counts=True)
    all_dates = pd.DatetimeIndex(unique_dates)
    n_unique = len(unique_dates)
    print(f"  {n_unique} unique dates, {len(factor_names)} factors, 1 label ({horizon_name})")

    # ---- daily rank IC ----
    print("Computing daily rank IC ...", flush=True)
    import time as _time
    t_start = _time.time()
    ic_records: list[dict] = []

    for g in range(n_unique):
        s = start_idx[g]
        c = counts[g]
        f_arr = F_all[s:s + c]
        l_arr = L_all[s:s + c]
        date = all_dates[g]

        for f_idx in range(n_factors):
            f_vals = f_arr[:, f_idx]
            l_vals = l_arr[:, 0]
            ic = _rank_ic_np(f_vals, l_vals)
            if not np.isnan(ic):
                ic_records.append({
                    "date": date, "factor": factor_names[f_idx],
                    "horizon": horizon_name, "ic": ic,
                })

        if (g + 1) % 200 == 0:
            elapsed = _time.time() - t_start
            print(f"  ... {g + 1}/{n_unique} dates, {len(ic_records)} ICs, {elapsed:.1f}s", flush=True)

    ic_df = pd.DataFrame(ic_records)
    print(f"  IC records: {len(ic_df)}", flush=True)

    if ic_df.empty:
        print("ERROR: No IC records computed")
        return

    # ---- correlation matrix (pandas .corr, fast C-level) ----
    print("Computing cross-sectional correlation matrix ...", flush=True)
    corr_sum = np.zeros((n_factors, n_factors))
    corr_count = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        for g in range(n_unique):
            s = start_idx[g]
            c = counts[g]
            if c < MIN_STOCKS_PER_DATE:
                continue

            f_arr = F_all[s:s + c]
            f_arr = np.clip(f_arr, -1e15, 1e15)

            corr_mat = pd.DataFrame(f_arr, columns=factor_names).corr(min_periods=MIN_STOCKS_PER_DATE)
            corr_sum += corr_mat.fillna(0.0).values
            corr_count += 1

            if corr_count % 500 == 0:
                elapsed = _time.time() - t_start
                print(f"  ... {corr_count} dates, {elapsed:.1f}s", flush=True)

    if corr_count == 0:
        print("ERROR: No valid dates for correlation computation")
        return

    corr_matrix = corr_sum / corr_count
    corr_df = pd.DataFrame(corr_matrix, index=factor_names, columns=factor_names)
    # pandas 2.x CoW 下 .values 为只读视图，np.fill_diagonal 原地写会抛
    # "underlying array is read-only"——显式拷贝后再写
    corr_vals = corr_df.to_numpy(copy=True)
    np.fill_diagonal(corr_vals, 1.0)
    corr_df = pd.DataFrame(corr_vals, index=factor_names, columns=factor_names)
    print(f"  Correlation averaged over {corr_count} dates")

    # ---- IC summary ----
    ic_primary = (ic_df[ic_df["horizon"] == horizon_name]
                  .groupby("factor")["ic"]
                  .agg(["mean", "std", "count"])
                  .reset_index())
    ic_primary["abs_mean_ic"] = ic_primary["mean"].abs()
    ic_primary = ic_primary.sort_values("abs_mean_ic", ascending=False).set_index("factor")

    ic_all = (ic_df.groupby("factor")["ic"]
              .agg(["mean", "std", "count"])
              .reset_index())
    ic_all["abs_mean_ic"] = ic_all["mean"].abs()
    ic_all = ic_all.set_index("factor")

    # ---- cluster-first selection（2026-08-21 用户裁定：相关度定结构，IC 只选簇内代表）----
    print(f"\nCluster-first: average-linkage on 1-|corr|, intra-cluster corr >= {CLUSTER_CORR}, "
          f"max {MAX_FACTORS} factors")
    corr_abs = corr_df.abs().to_numpy(copy=True)
    np.fill_diagonal(corr_abs, 0.0)
    Z = linkage(squareform(1.0 - corr_abs, checks=False), method="average")
    cluster_labels = fcluster(Z, t=1.0 - CLUSTER_CORR, criterion="distance")
    cluster_of = dict(zip(corr_df.index, cluster_labels))
    n_clusters = len(set(cluster_labels))
    print(f"  {len(corr_df)} factors -> {n_clusters} clusters")

    # MUST_INCLUDE 所在簇由保送因子占代表席（一簇一席）
    selected = [f for f in MUST_INCLUDE if f in cluster_of]
    if selected:
        print(f"  Must-include: {selected}")
    must_clusters = {cluster_of[f] for f in selected}

    # 每簇代表 = 簇内 |IC| 最高者；按 |IC| 降序首见即代表，代表序天然按 IC 降序
    rep_of_cluster = {}
    for f in ic_primary.index:
        if f not in cluster_of or cluster_of[f] in must_clusters:
            continue
        if cluster_of[f] not in rep_of_cluster:
            rep_of_cluster[cluster_of[f]] = f
    reps_ordered = list(rep_of_cluster.values())

    for f in reps_ordered:
        if len(selected) >= MAX_FACTORS:
            break
        selected.append(f)

    # 簇内落选者记 discarded（与同簇所选代表的相关度；average-linkage 下
    # 个别成员与代表的成对相关可低于簇阈值，属正常）
    sel_set = set(selected)
    sel_by_cluster = {cluster_of[f]: f for f in selected}
    discarded_corr = []
    for f in corr_df.index:
        if f in sel_set:
            continue
        rep = sel_by_cluster.get(cluster_of[f])
        if rep is not None:
            discarded_corr.append({
                "factor": f,
                "corr_with": rep,
                "corr": float(corr_df.loc[f, rep]),
            })
    discarded_corr.sort(key=lambda d: -d["corr"])

    # ---- print results ----
    print(f"\n{'='*70}")
    print(f"  Selected: {len(selected)} factors")
    print(f"{'='*70}")
    for i, f in enumerate(selected, 1):
        ic_val = ic_primary.loc[f, "mean"] if f in ic_primary.index else np.nan
        ic_all_val = ic_all.loc[f, "mean"] if f in ic_all.index else np.nan
        print(f"  {i:3d}. {f:35s} |IC_{args.model}|={abs(ic_val):.4f}  IC_all={ic_all_val:+.4f}")

    print(f"\n{'='*70}")
    print(f"  Discarded by cluster (representative kept): {len(discarded_corr)}")
    print(f"{'='*70}")
    for i, d in enumerate(discarded_corr[:30], 1):
        print(f"  {i:3d}. {d['factor']:35s} corr={d['corr']:.3f} with {d['corr_with']}")
    if len(discarded_corr) > 30:
        print(f"  ... and {len(discarded_corr) - 30} more")

    remaining = [f for f in ic_primary.index
                 if f not in selected
                 and f not in {d["factor"] for d in discarded_corr}]
    if remaining:
        print(f"\n  Excluded by cap ({len(remaining)}):")
        for f in remaining[:15]:
            print(f"    {f}")
        if len(remaining) > 15:
            print(f"    ... and {len(remaining) - 15} more")

    # ---- save ----
    if args.fold:
        fold_dir = Path(__file__).resolve().parent / "folds" / args.fold
        fold_dir.mkdir(parents=True, exist_ok=True)
        output_path = fold_dir / f"selected_{POOL_NAME}_{args.model}.json"
    else:
        output_path = Path(__file__).resolve().parent / f"selected_{POOL_NAME}_{args.model}.json"
    result = {
        "pool": POOL_NAME,
        "model": args.model,
        "fold": args.fold,
        "label_fn": "compute_median_open",
        "label_window": list(cfg["label_window"]),
        "baseline": cfg["baseline"],
        "train_start": str(train_start.date()),
        "train_end": str(test_start.date()),
        "train_start_note": "2026-08-21 用户裁定：筛选(IC+相关度)与训练对齐，均自 2020 起（曾用 2015 长历史口径）",
        "algorithm": "cluster-first: average-linkage on 1-|corr|, best-|IC| per cluster, clusters ranked by rep |IC|",
        "cluster_corr_threshold": CLUSTER_CORR,
        "exclude_prefixes": list(EXCLUDE_PREFIXES),
        "n_clusters": n_clusters,
        "max_factors": MAX_FACTORS,
        "n_dates_corr": int(corr_count),
        "selected_factors": selected,
        "factor_metrics": {
            f: {
                "ic_primary_mean": float(ic_primary.loc[f, "mean"]) if f in ic_primary.index else None,
                "ic_primary_abs_mean": float(ic_primary.loc[f, "abs_mean_ic"]) if f in ic_primary.index else None,
                "ic_all_mean": float(ic_all.loc[f, "mean"]) if f in ic_all.index else None,
                "n_ic_dates": int(ic_primary.loc[f, "count"]) if f in ic_primary.index else 0,
            }
            for f in selected
        },
        "discarded_corr": [
            {"factor": d["factor"], "corr_with": d["corr_with"], "corr": d["corr"]}
            for d in discarded_corr
        ],
    }

    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
