# -*- coding: utf-8 -*-
"""
因子截断一致性测试 (Causality / Look-ahead Bias Test)

对每个因子进行截断测试:
  使用截至日期 T 的数据计算因子，与使用完整数据计算的因子在 [<= T] 范围内做比较。
  任何不一致都说明因子读取了未来信息，或计算结果依赖未来样本范围。

Usage:
    python -m factors.test_causality
    python -m factors.test_causality --n-stocks 30 --n-dates 500 --n-splits 5
"""
from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_PATH, get_pool_codes

# ---------------------------------------------------------------------------
# Factor engine imports (same as compute.py)
# ---------------------------------------------------------------------------
from factors.alpha101 import ALPHA_EXPRESSIONS
from factors.utility import calculate_by_expression
from factors.extra_factors import compute_non_alpha_factors, apply_ind_neutralize
from factors.compute import (
    _load_ohlcv, _load_market_cap, _load_stock_info,
    _load_industry, _load_cyq, _load_index_data, _compute_isst,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_N_STOCKS = 20
DEFAULT_N_DATES = 400
DEFAULT_N_SPLITS = 3
FLOAT_TOLERANCE = 1e-6


def _prepare_factor_data(con, codes: list[str]) -> pl.DataFrame:
    """Load and merge all data needed for factor computation (without running factors)."""
    df = _load_ohlcv(con, codes)
    if df.is_empty():
        return pl.DataFrame()

    mktcap = _load_market_cap(con, codes)
    if not mktcap.is_empty():
        df = df.join(mktcap, on=["datetime", "vt_symbol"], how="left")

    info = _load_stock_info(con)
    if not info.is_empty():
        df = df.join(info, on="vt_symbol", how="left")

    df = df.sort(["vt_symbol", "datetime"])

    if "total_mv" in df.columns:
        df = df.with_columns(pl.col("total_mv").alias("cap"))

    # Pre-compute daily return
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("vt_symbol") - 1).alias("ret")
    )

    return df


def _run_alpha_factors(df: pl.DataFrame) -> pl.DataFrame:
    """Compute all alpha factors (including _v0) on the given data."""
    id_cols = df[["datetime", "vt_symbol"]]
    alpha_df = id_cols.clone()

    for name, expr in sorted(ALPHA_EXPRESSIONS.items()):
        try:
            result_df = calculate_by_expression(df, expr)
            alpha_df = alpha_df.with_columns(result_df["data"].alias(name))
        except Exception as e:
            print(f"    [SKIP] {name}: {e}")

    return alpha_df


def _run_non_alpha_factors(df: pl.DataFrame, con: duckdb.DuckDBPyConnection,
                           codes: list[str]) -> pl.DataFrame:
    """Compute non-alpha factors on the given data."""
    extra_df = compute_non_alpha_factors(df)
    
    # Supplementary factors: chip, market state, IsST
    cyq_df = _load_cyq(con, codes)
    if not cyq_df.is_empty():
        cyq_df = cyq_df.with_columns(
            pl.col("winner_rate").alias("WinnerRate"),
            ((pl.col("cost_50pct") - pl.col("weight_avg")) / 
             (pl.col("cost_95pct") - pl.col("cost_5pct") + 1e-10)).alias("CostPosition"),
            ((pl.col("cost_85pct") - pl.col("cost_15pct")) / 
             (pl.col("cost_50pct") + 1e-10)).alias("ChipDispersion"),
            ((pl.col("cost_50pct") - pl.col("weight_avg")) / 
             (pl.col("cost_85pct") - pl.col("cost_15pct") + 1e-10)).alias("ChipSkew"),
        )
        chip_cols = ["datetime", "vt_symbol", "WinnerRate", "CostPosition",
                      "ChipDispersion", "ChipSkew"]
        cyq_df = cyq_df.select(chip_cols)
        extra_df = extra_df.join(cyq_df, on=["datetime", "vt_symbol"], how="left")

    market_df = _load_index_data(con)
    if not market_df.is_empty():
        symbols = extra_df.select("vt_symbol").unique()
        market_df = symbols.join(market_df, how="cross")
        extra_df = extra_df.join(market_df, on=["datetime", "vt_symbol"], how="left")

    isst_df = _compute_isst(con)
    if not isst_df.is_empty():
        extra_df = extra_df.join(isst_df, on=["datetime", "vt_symbol"], how="left")
        extra_df = extra_df.with_columns(pl.col("IsST").fill_null(0).cast(pl.Int32))
    else:
        extra_df = extra_df.with_columns(pl.lit(0).cast(pl.Int32).alias("IsST"))

    return extra_df


def _run_ind_neutralize(alpha_df: pl.DataFrame, con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """Apply industry neutralization."""
    industry_map = _load_industry(con)
    if industry_map.is_empty():
        return alpha_df
    return apply_ind_neutralize(alpha_df, industry_map)


def _compare_factors(ref: pl.DataFrame, test: pl.DataFrame,
                     factor_names: list[str],
                     margin_dates: int = 30) -> dict:
    """
    Compare factor values between reference and test DataFrames.

    Parameters
    ----------
    ref : pl.DataFrame
        Reference factors computed on full data.
    test : pl.DataFrame
        Test factors computed on truncated data.
    factor_names : list[str]
        Factor columns to compare.
    margin_dates : int
        Number of trading days before the truncation boundary to exclude.
        This avoids false positives from rolling windows that span the boundary.

    Returns
    -------
    dict with:
      - passed: list of factor names that passed
      - failed: list of (name, max_diff, sample_diffs) for factors that failed
      - skipped: list of factor names not in both DataFrames
    """
    # Align on (datetime, vt_symbol)
    ref_sel = ref.select(["datetime", "vt_symbol"] + 
                         [c for c in factor_names if c in ref.columns])
    test_sel = test.select(["datetime", "vt_symbol"] + 
                          [c for c in factor_names if c in test.columns])

    if ref_sel.is_empty() or test_sel.is_empty():
        return {"passed": [], "failed": [], "skipped": factor_names}

    # Find common symbol-date pairs
    ref_keys = ref_sel.select(["datetime", "vt_symbol"])
    test_keys = test_sel.select(["datetime", "vt_symbol"])
    common = ref_keys.join(test_keys, on=["datetime", "vt_symbol"], how="inner")

    # Remove boundary dates (last N trading days)
    all_dates = sorted(common["datetime"].unique())
    cutoff_date = all_dates[-margin_dates - 1] if len(all_dates) > margin_dates else all_dates[0]
    common = common.filter(pl.col("datetime") <= cutoff_date)

    if common.is_empty():
        return {"passed": [], "failed": [], "skipped": factor_names}

    # Join back to get values
    ref_sel = common.join(ref_sel, on=["datetime", "vt_symbol"], how="inner")
    test_sel = common.join(test_sel, on=["datetime", "vt_symbol"], how="inner")

    passed = []
    failed = []
    skipped = []

    for name in factor_names:
        if name not in ref_sel.columns or name not in test_sel.columns:
            skipped.append(name)
            continue

        ref_vals = ref_sel[name].fill_nan(None)
        test_vals = test_sel[name].fill_nan(None)

        # Both must be null or both must be non-null
        ref_null = ref_vals.is_null()
        test_null = test_vals.is_null()
        null_mismatch = (ref_null != test_null).sum()
        
        if null_mismatch > 0:
            failed.append((name, float("nan"), 
                          [f"{null_mismatch} NaN mismatches"]))
            continue

        # Both non-null
        non_null = ~ref_null
        if non_null.sum() == 0:
            passed.append(name)
            continue

        ref_nonnull = ref_vals.filter(non_null).to_numpy()
        test_nonnull = test_vals.filter(non_null).to_numpy()

        # Replace infinities
        ref_nonnull = np.nan_to_num(ref_nonnull, nan=0.0, posinf=1e10, neginf=-1e10)
        test_nonnull = np.nan_to_num(test_nonnull, nan=0.0, posinf=1e10, neginf=-1e10)

        diffs = np.abs(ref_nonnull - test_nonnull)
        max_diff = float(diffs.max())

        if max_diff > FLOAT_TOLERANCE:
            # Find worst offenders
            worst_idx = np.argsort(-diffs)[:5]
            samples = [f"  diff={diffs[i]:.6e}  ref={ref_nonnull[i]:.6f}  test={test_nonnull[i]:.6f}"
                       for i in worst_idx if diffs[i] > FLOAT_TOLERANCE]
            failed.append((name, max_diff, samples))
        else:
            passed.append(name)

    return {"passed": passed, "failed": failed, "skipped": skipped}


def main():
    parser = argparse.ArgumentParser(description="因子截断一致性测试")
    parser.add_argument("--n-stocks", type=int, default=DEFAULT_N_STOCKS,
                        help="样本股票数量")
    parser.add_argument("--n-dates", type=int, default=DEFAULT_N_DATES,
                        help="每只股票使用多少交易日")
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS,
                        help="测试几个截断点")
    parser.add_argument("--margin", type=int, default=30,
                        help="截断边界的安全边际（交易日数）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    args = parser.parse_args()

    print("=" * 70)
    print("  因子截断一致性测试 (Causality Test)")
    print(f"  stocks={args.n_stocks}  dates={args.n_dates}  splits={args.n_splits}")
    print("=" * 70)

    np.random.seed(args.seed)

    # ---- 1. Load data ----
    t0 = time.time()
    pool_codes = get_pool_codes()
    selected = sorted(np.random.choice(pool_codes, size=min(args.n_stocks, len(pool_codes)),
                                        replace=False))
    print(f"\n[1/4] Loading data for {len(selected)} stocks ...")

    con = duckdb.connect(str(DB_PATH), read_only=True)

    df = _prepare_factor_data(con, selected)
    if df.is_empty():
        print("  ERROR: No data loaded.")
        con.close()
        return

    # Keep last N dates per stock
    all_symbols = df.select("vt_symbol").unique()
    result_parts = []
    for sym_row in all_symbols.iter_rows():
        sym = sym_row[0]
        sym_df = df.filter(pl.col("vt_symbol") == sym).sort("datetime")
        if len(sym_df) > args.n_dates:
            sym_df = sym_df.tail(args.n_dates)
        result_parts.append(sym_df)
    df = pl.concat(result_parts)
    df = df.sort(["vt_symbol", "datetime"])

    all_dates = sorted(df["datetime"].unique())
    n_dates = len(all_dates)
    n_stocks = df["vt_symbol"].n_unique()
    print(f"  {n_stocks} stocks × {n_dates} dates = {len(df)} rows")
    print(f"  date range: {all_dates[0]} ~ {all_dates[-1]}")

    # ---- 2. Compute reference factors on FULL data ----
    print(f"\n[2/4] Computing reference factors on FULL data ...")
    t_ref = time.time()

    alpha_ref = _run_alpha_factors(df)
    alpha_ref = _run_ind_neutralize(alpha_ref, con)

    extra_ref = _run_non_alpha_factors(df, con, selected)
    extra_fact_cols = [c for c in extra_ref.columns 
                       if c not in ("datetime", "vt_symbol", "ret", "cap")
                       and not c.startswith("_")]

    # Merge alpha + extra
    ref = alpha_ref.join(
        extra_ref.select(["datetime", "vt_symbol"] + extra_fact_cols),
        on=["datetime", "vt_symbol"], how="left"
    )

    alpha_names = sorted(ALPHA_EXPRESSIONS.keys())
    all_factor_names = alpha_names + extra_fact_cols
    print(f"  Reference: {len(alpha_names)} alpha + {len(extra_fact_cols)} non-alpha = "
          f"{len(all_factor_names)} factors ({time.time() - t_ref:.1f}s)")

    # ---- 3. Run truncation tests ----
    print(f"\n[3/4] Running truncation tests at {args.n_splits} points ...")
    
    split_indices = np.linspace(n_dates // 2, n_dates - args.margin - 1, 
                                args.n_splits, dtype=int)
    all_results = []

    for split_idx in split_indices:
        trunc_date = all_dates[split_idx]
        print(f"\n  --- Truncation at {trunc_date} (date {split_idx}/{n_dates}) ---")
        t_trunc = time.time()

        df_trunc = df.filter(pl.col("datetime") <= trunc_date)

        # Compute alpha factors on truncated data
        alpha_trunc = _run_alpha_factors(df_trunc)
        alpha_trunc = _run_ind_neutralize(alpha_trunc, con)

        # Compute non-alpha on truncated data
        extra_trunc = _run_non_alpha_factors(df_trunc, con, selected)
        extra_trunc_cols = [c for c in extra_trunc.columns 
                            if c in extra_fact_cols]

        trunc = alpha_trunc.join(
            extra_trunc.select(["datetime", "vt_symbol"] + extra_trunc_cols),
            on=["datetime", "vt_symbol"], how="left"
        )

        # Compare
        result = _compare_factors(ref, trunc, all_factor_names, margin_dates=args.margin)
        n_pass = len(result["passed"])
        n_fail = len(result["failed"])
        n_skip = len(result["skipped"])
        elapsed = time.time() - t_trunc

        print(f"    PASS: {n_pass}  FAIL: {n_fail}  SKIP: {n_skip}  ({elapsed:.1f}s)")

        if result["failed"]:
            for name, max_diff, samples in result["failed"]:
                print(f"    [FAIL] {name}: max_diff={max_diff:.4e}")
                for s in samples[:3]:
                    print(s)

        all_results.append((trunc_date, result))

    con.close()

    # ---- 4. Summary ----
    print(f"\n[4/4] Summary (tolerance={FLOAT_TOLERANCE})")
    print("=" * 70)

    # Aggregate: a factor is "clean" if it passes ALL truncation splits
    all_factors = set(all_factor_names)
    failed_by_factor: dict[str, list] = {}
    
    for trunc_date, result in all_results:
        for name, max_diff, samples in result["failed"]:
            if name not in failed_by_factor:
                failed_by_factor[name] = []
            failed_by_factor[name].append((trunc_date, max_diff, samples))

    clean = all_factors - set(failed_by_factor)
    
    print(f"\n  CLEAN (pass all splits): {len(clean)} factors")
    print(f"  FAILED: {len(failed_by_factor)} factors")
    
    if failed_by_factor:
        print(f"\n  Failed factors:")
        for name in sorted(failed_by_factor):
            entries = failed_by_factor[name]
            n_fail = len(entries)
            max_d = max(e[1] for e in entries)
            print(f"    {name}: failed {n_fail}/{args.n_splits} splits, "
                  f"max_abs_diff={max_d:.2e}")

    print(f"\n  Total elapsed: {time.time() - t0:.1f}s")

    if failed_by_factor:
        print(f"\n  *** WARNING: {len(failed_by_factor)} factors show look-ahead bias! ***")
        return 1
    
    print(f"\n  *** ALL CLEAN: No look-ahead bias detected ***")
    return 0


if __name__ == "__main__":
    sys.exit(main())
