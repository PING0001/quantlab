# -*- coding: utf-8 -*-
"""
Factor correlation analysis and diversity selection.

Loads pre-computed factor values from the factor_values table,
computes cross-sectional rank correlation, and prints a summary.
Optionally exports the correlation matrix to JSON in %TMP%.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_PATH, POOL_NAME

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# load factor panel from DuckDB
# ---------------------------------------------------------------------------

def load_factor_panel(
    db_path: str | Path | None = None,
    factors: list[str] | None = None,
    max_stocks: int | None = None,
    max_dates: int | None = None,
) -> pd.DataFrame:
    """Load factor values from the factor_values table as a MultiIndex panel.

    Parameters
    ----------
    db_path : path, optional
        Path to DuckDB database. Defaults to DB_PATH from config.
    factors : list[str], optional
        Factor columns to load.  If None, loads all factor columns.
    max_stocks : int, optional
        Limit number of stocks (random sample) for faster analysis.
    max_dates : int, optional
        Limit to the most recent N dates for faster analysis.

    Returns
    -------
    panel : DataFrame
        MultiIndex (date, code) with factor columns.
    """
    con = duckdb.connect(str(db_path or DB_PATH), read_only=True)

    # discover factor columns if not specified
    all_cols = con.execute("DESCRIBE factor_values").fetchall()
    factor_cols = [c[0] for c in all_cols if c[0] not in ("code", "date")]
    if factors is None:
        factors = factor_cols
    else:
        factors = [f for f in factors if f in factor_cols]
        missing = set(factors) - set(factor_cols)
        if missing:
            log.warning("Factors not in table: %s", missing)

    if not factors:
        log.error("No valid factor columns to load.")
        con.close()
        return pd.DataFrame()

    quoted = ", ".join(f'"{f}"' for f in factors)

    # build SQL
    wheres = []
    if max_dates is not None:
        wheres.append(
            f"date IN (SELECT DISTINCT date FROM factor_values ORDER BY date DESC LIMIT {max_dates})"
        )
    if max_stocks is not None:
        wheres.append(
            f"code IN (SELECT DISTINCT code FROM factor_values USING SAMPLE {max_stocks})"
        )
    sql = f'SELECT code, date, {quoted} FROM factor_values'
    if wheres:
        sql += "\n  WHERE " + " AND ".join(wheres)
    sql += " ORDER BY date, code"
    log.info("Loading factor panel...")
    df = con.execute(sql).df()
    con.close()

    log.info(
        "Loaded %d rows × %d factors, %d stocks, %d dates",
        len(df), len(factors),
        df["code"].nunique(), df["date"].nunique(),
    )
    return df.set_index(["date", "code"])


# ---------------------------------------------------------------------------
# correlation computation
# ---------------------------------------------------------------------------

def cross_sectional_rank_corr(factor_panel: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-sectional Spearman rank correlation matrix.

    At each date, compute Spearman correlation between all factor pairs,
    then average across dates.

    Parameters
    ----------
    factor_panel : DataFrame
        MultiIndex (date, code) with one column per factor.

    Returns
    -------
    corr_matrix : DataFrame
        Average cross-sectional rank correlation matrix.
    """
    dates = factor_panel.index.get_level_values("date").unique()
    factor_names = list(factor_panel.columns)
    n_factors = len(factor_names)
    log.info("Computing cross-sectional rank correlation across %d dates...", len(dates))

    sum_corr = np.zeros((n_factors, n_factors))
    n_valid_dates = 0

    for i, dt in enumerate(dates):
        xs = factor_panel.xs(dt, level="date")
        xs = xs.dropna()
        if len(xs) < 10:
            continue
        corr = xs.corr(method="spearman").values
        corr = np.nan_to_num(corr, nan=0.0)
        sum_corr += corr
        n_valid_dates += 1
        if (i + 1) % 500 == 0:
            log.info("  processed %d/%d dates", i + 1, len(dates))

    avg_corr = sum_corr / max(n_valid_dates, 1)
    log.info("Correlation matrix built from %d valid dates.", n_valid_dates)
    return pd.DataFrame(avg_corr, index=factor_names, columns=factor_names)


def time_series_corr(factor_panel: pd.DataFrame) -> pd.DataFrame:
    """Compute average time-series Pearson correlation across stocks.

    For each stock, computes Pearson correlation between factor values
    over time, then averages across stocks.
    """
    stocks = factor_panel.index.get_level_values("code").unique()
    factor_names = list(factor_panel.columns)
    n = len(factor_names)
    log.info("Computing time-series correlation across %d stocks...", len(stocks))

    sum_corr = np.zeros((n, n))
    n_valid = 0

    for i, code in enumerate(stocks):
        ts = factor_panel.xs(code, level="code")
        ts = ts.dropna()
        if len(ts) < 20:
            continue
        corr = ts.corr(method="pearson").values
        corr = np.nan_to_num(corr, nan=0.0)
        sum_corr += corr
        n_valid += 1
        if (i + 1) % 200 == 0:
            log.info("  processed %d/%d stocks", i + 1, len(stocks))

    avg_corr = sum_corr / max(n_valid, 1)
    log.info("Correlation matrix built from %d valid stocks.", n_valid)
    return pd.DataFrame(avg_corr, index=factor_names, columns=factor_names)


# ---------------------------------------------------------------------------
# greedy selection
# ---------------------------------------------------------------------------

def greedy_diverse_selection(
    corr_matrix: pd.DataFrame,
    n_select: int = 10,
    max_corr_threshold: float = 0.7,
    start_factor: str | None = None,
    mandatory: list[str] | None = None,
) -> list[str]:
    """Greedy select factors with minimal pairwise redundancy.

    1. Start with mandatory factors, then the factor with lowest
       average absolute correlation (or specified start_factor).
    2. At each step, add the unselected factor whose max absolute
       correlation with already-selected factors is lowest.
    3. Stop when n_select reached or all remaining factors exceed
       max_corr_threshold.
    """
    factors = list(corr_matrix.columns)
    abs_corr = corr_matrix.abs().values

    selected = list(mandatory) if mandatory is not None else []
    for f in selected:
        if f not in factors:
            raise ValueError(f"Mandatory factor '{f}' not found in correlation matrix")

    if start_factor is not None and start_factor not in selected:
        selected.append(start_factor)

    if not selected:
        avg_abs_corr = abs_corr.sum(axis=1) / len(factors)
        start_idx = int(np.argmin(avg_abs_corr))
        selected = [factors[start_idx]]

    remaining = [f for f in factors if f not in selected]

    while len(selected) < n_select and remaining:
        best_score = float("inf")
        best_factor = None
        for f in remaining:
            f_idx = factors.index(f)
            sel_indices = [factors.index(s) for s in selected]
            max_corr = max(abs_corr[f_idx, s_idx] for s_idx in sel_indices)
            if max_corr < best_score:
                best_score = max_corr
                best_factor = f
        if best_factor is None or best_score > max_corr_threshold:
            break
        selected.append(best_factor)
        remaining.remove(best_factor)

    return selected


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def print_correlation_summary(corr_matrix: pd.DataFrame, selected: list[str] | None = None):
    """Print a human-readable summary of the correlation matrix."""
    selected = selected or []
    sel_set = set(selected)

    print("\n" + "=" * 60)
    print("  FACTOR CORRELATION ANALYSIS")
    print("=" * 60)
    print(f"  Factors analyzed : {len(corr_matrix)}")
    print(f"  Pool             : {POOL_NAME}")
    print(f"  Time             : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if selected:
        print(f"  Greedy selected  : {len(selected)}/{len(corr_matrix)}")
    print()

    # per-factor average absolute correlation
    abs_corr = corr_matrix.abs()
    avg_corr = abs_corr.mean(axis=1).sort_values()
    print("  Factors ranked by average |r| (lower = more unique):")
    print(f"  {'Factor':<28} {'Avg |r|':>8}")
    print("  " + "-" * 38)
    for f in avg_corr.index:
        marker = "  <--" if f in sel_set else ""
        print(f"  {f:<28} {avg_corr[f]:>8.3f}{marker}")
    print()

    # selected-set internal correlation
    if len(selected) > 1:
        sel_corr = corr_matrix.loc[selected, selected]
        vals = sel_corr.where(~np.eye(len(selected), dtype=bool))
        max_pair = vals.max().max()
        print(f"  Max pairwise |r| within selected set: {max_pair:.3f}")
        print()

    # highly correlated pairs
    print("  Highly correlated pairs (|r| > 0.8):")
    printed: set[tuple] = set()
    for i, f1 in enumerate(corr_matrix.columns):
        for j, f2 in enumerate(corr_matrix.columns):
            if i >= j:
                continue
            r_val = abs_corr.iloc[i, j]
            if r_val > 0.8:
                key = (f1, f2) if f1 < f2 else (f2, f1)
                if key not in printed:
                    printed.add(key)
                    m1 = " [selected]" if f1 in sel_set else ""
                    m2 = " [selected]" if f2 in sel_set else ""
                    print(f"    r = {r_val:.3f}   {f1}{m1}  <->  {f2}{m2}")
    if not printed:
        print("    (none)")
    print()


def export_corr_json(corr_matrix: pd.DataFrame, output_dir: str | Path | None = None):
    """Export correlation matrix as JSON to a temp directory."""
    out_dir = Path(output_dir) if output_dir else Path(os.environ.get("TMP", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"factor_correlation_{POOL_NAME}_{ts}.json"

    corr_json = {
        "pool": POOL_NAME,
        "n_factors": len(corr_matrix),
        "factors": list(corr_matrix.columns),
        "correlation": corr_matrix.to_dict(orient="records"),
        "generated_at": datetime.now().isoformat(),
    }
    out_path.write_text(json.dumps(corr_json, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Correlation matrix exported to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def analyze(
    db_path: str | Path | None = None,
    factors: list[str] | None = None,
    method: str = "cross_sectional",
    n_select: int = 0,
    max_corr: float = 0.7,
    mandatory: list[str] | None = None,
    export_json: bool = False,
    max_stocks: int | None = None,
    max_dates: int | None = None,
):
    """Load factors, compute correlation, print summary.

    Parameters
    ----------
    method : str
        "cross_sectional" (default) or "time_series".
    n_select : int
        If > 0, run greedy selection to pick this many diverse factors.
    export_json : bool
        If True, export correlation matrix as JSON to %TMP%.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    # 1. load
    panel = load_factor_panel(
        db_path=db_path, factors=factors,
        max_stocks=max_stocks, max_dates=max_dates,
    )
    if panel.empty:
        return

    # 2. drop factors with too many NaNs
    nan_frac = panel.isna().mean(axis=0)
    bad = nan_frac[nan_frac > 0.5].index.tolist()
    if bad:
        log.warning("Dropping %d factors with >50%% NaN: %s", len(bad), bad)
        panel = panel.drop(columns=bad)

    log.info("Analyzing %d factors, %d rows", len(panel.columns), len(panel))

    # 3. compute correlation
    if method == "time_series":
        corr = time_series_corr(panel)
    else:
        corr = cross_sectional_rank_corr(panel)

    # 4. greedy selection (optional)
    selected: list[str] = []
    if n_select > 0:
        selected = greedy_diverse_selection(
            corr, n_select=n_select, max_corr_threshold=max_corr,
            mandatory=mandatory,
        )
        log.info("Greedy selected %d factors: %s", len(selected), selected)

    # 5. print summary
    print_correlation_summary(corr, selected)

    # 6. export (optional)
    if export_json:
        export_corr_json(corr)

    return corr, selected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Analyze factor correlations from factor_values table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m factors.selection
  python -m factors.selection --method ts
  python -m factors.selection --factors Return_1d,Return_5d,Volatility,ATR --select 3
  python -m factors.selection --json
        """,
    )
    ap.add_argument("--factors", type=str, default=None,
                    help="Comma-separated factor names to analyze (default: all in table)")
    ap.add_argument("--method", choices=["cs", "ts"], default="cs",
                    help="Correlation method: cs=cross-sectional Spearman (default), ts=time-series Pearson")
    ap.add_argument("--select", type=int, default=0, metavar="N",
                    help="Run greedy selection to pick N diverse factors")
    ap.add_argument("--max-corr", type=float, default=0.7,
                    help="Max correlation threshold for greedy selection (default: 0.7)")
    ap.add_argument("--mandatory", type=str, default=None,
                    help="Comma-separated mandatory factors for greedy selection")
    ap.add_argument("--json", action="store_true", default=False,
                    help="Export correlation matrix as JSON to %%TMP%%")
    ap.add_argument("--max-stocks", type=int, default=None,
                    help="Limit to N random stocks for faster analysis")
    ap.add_argument("--max-dates", type=int, default=None,
                    help="Limit to most recent N dates for faster analysis")

    args = ap.parse_args()

    factor_list = None
    if args.factors:
        factor_list = [s.strip() for s in args.factors.split(",")]

    mandatory_list = None
    if args.mandatory:
        mandatory_list = [s.strip() for s in args.mandatory.split(",")]

    method_map = {"cs": "cross_sectional", "ts": "time_series"}
    analyze(
        factors=factor_list,
        method=method_map[args.method],
        n_select=args.select,
        max_corr=args.max_corr,
        mandatory=mandatory_list,
        export_json=args.json,
        max_stocks=args.max_stocks,
        max_dates=args.max_dates,
    )
