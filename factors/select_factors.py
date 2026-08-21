"""
Factor selection via IC ranking + correlation filtering.

Computes cross-sectional rank IC and pairwise factor correlation in the
training set, then greedily selects factors with highest |IC| while capping
pairwise correlation below a threshold.

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
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_PATH, POOL_NAME, get_pool_codes, SELECTED_FACTORS, MODEL_CONFIGS
from strategies.labels import compute_median_open


# 口径说明：此 TRAIN_START(2015) 与 run_lgb 的 TRAIN_START(2020) 历史上就
# 不一致（筛选看更长历史），不改行为，输出 json 如实记录两者
TRAIN_START = pd.Timestamp("2015-01-01")
TEST_START = pd.Timestamp("2025-06-01")
MAX_FACTORS = 60
CORR_THRESHOLD = 0.75
MIN_STOCKS_PER_DATE = 30
MUST_INCLUDE = ["CSI_return_20d"]


def load_factors(con):
    pool_codes = get_pool_codes()
    placeholders = ",".join(["?"] * len(pool_codes))
    query = f"SELECT * FROM factor_values WHERE code IN ({placeholders})"
    df = con.execute(query, pool_codes).fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "code"]).sort_index()
    return df


def load_kline(con):
    pool_codes = get_pool_codes()
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
    args = parser.parse_args()
    cfg = MODEL_CONFIGS[args.model]
    horizon_name = cfg["horizon"]

    print(f"Pool: {POOL_NAME} | model: {args.model} "
          f"(label T+{cfg['label_window'][0]}..T+{cfg['label_window'][1]}, baseline={cfg['baseline']})")
    con = duckdb.connect(str(DB_PATH), read_only=True)

    print("Loading factors ...")
    factors_raw = load_factors(con)
    available = [f for f in SELECTED_FACTORS if f in factors_raw.columns]
    missing = [f for f in SELECTED_FACTORS if f not in factors_raw.columns]
    if missing:
        print(f"  WARNING: {len(missing)} factors missing: {missing[:10]}...")
    factors = factors_raw[available].copy()
    factor_names = available
    n_factors = len(factor_names)
    print(f"  {n_factors} factors available")

    date_level = factors.index.get_level_values("date")
    train_mask = (date_level >= TRAIN_START) & (date_level < TEST_START)
    factors = factors.loc[train_mask]
    print(f"  Training range: {factors.index.get_level_values('date').min().date()} ~ "
          f"{factors.index.get_level_values('date').max().date()}")
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

    # ---- greedy selection ----
    print(f"\nGreedy selection: corr < {CORR_THRESHOLD}, max {MAX_FACTORS} factors")
    selected = [f for f in MUST_INCLUDE if f in corr_df.index]
    if selected:
        print(f"  Must-include: {selected}")
    discarded_corr = []

    for factor in ic_primary.index:
        if factor in selected:
            continue
        if len(selected) >= MAX_FACTORS:
            break

        if factor not in corr_df.index:
            continue

        if not selected:
            selected.append(factor)
            continue

        max_corr = corr_df.loc[factor, selected].abs().max()
        if max_corr < CORR_THRESHOLD:
            selected.append(factor)
        else:
            corr_with_sel = corr_df.loc[factor, selected].abs()
            max_cf = corr_with_sel.idxmax()
            discarded_corr.append({
                "factor": factor,
                "corr_with": max_cf,
                "corr": float(max_corr),
            })

    # ---- print results ----
    print(f"\n{'='*70}")
    print(f"  Selected: {len(selected)} factors")
    print(f"{'='*70}")
    for i, f in enumerate(selected, 1):
        ic_val = ic_primary.loc[f, "mean"] if f in ic_primary.index else np.nan
        ic_all_val = ic_all.loc[f, "mean"] if f in ic_all.index else np.nan
        print(f"  {i:3d}. {f:35s} |IC_{args.model}|={abs(ic_val):.4f}  IC_all={ic_all_val:+.4f}")

    print(f"\n{'='*70}")
    print(f"  Discarded by correlation: {len(discarded_corr)}")
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
    output_path = Path(__file__).resolve().parent / f"selected_{POOL_NAME}_{args.model}.json"
    result = {
        "pool": POOL_NAME,
        "model": args.model,
        "label_fn": "compute_median_open",
        "label_window": list(cfg["label_window"]),
        "baseline": cfg["baseline"],
        "train_start": str(TRAIN_START.date()),
        "train_end": str(TEST_START.date()),
        "train_start_note": "select_factors 自 2015 起算长历史；run_lgb 训练自 2020 起——历史口径差异，如实记录",
        "corr_threshold": CORR_THRESHOLD,
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
