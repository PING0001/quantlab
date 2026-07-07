# -*- coding: utf-8 -*-
"""
Factor correlation analysis and diversity selection.
Selects a subset of factors with minimal pairwise redundancy.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .compute import compute_panel, store_factors

log = logging.getLogger(__name__)


def cross_sectional_rank_corr(factor_panel: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-sectional rank correlation matrix between factors.

    At each date, compute Spearman rank correlation between all factor pairs,
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
    n_factors = len(factor_panel.columns)
    factor_names = list(factor_panel.columns)

    sum_corr = np.zeros((n_factors, n_factors))
    n_valid_dates = 0

    for dt in dates:
        xs = factor_panel.xs(dt, level="date")
        # Rank each factor cross-sectionally
        ranked = xs.rank(pct=True)
        # Drop rows with any NaN
        ranked = ranked.dropna()
        if len(ranked) < 10:
            continue
        corr = ranked.corr(method="spearman").values
        sum_corr += corr
        n_valid_dates += 1

    avg_corr = sum_corr / max(n_valid_dates, 1)
    return pd.DataFrame(avg_corr, index=factor_names, columns=factor_names)


def time_series_corr(factor_panel: pd.DataFrame) -> pd.DataFrame:
    """Compute average time-series correlation between factors across stocks.

    For each stock, computes the Pearson correlation between factor values
    over time, then averages across stocks.

    Returns correlation matrix averaged across stocks.
    """
    stocks = factor_panel.index.get_level_values("code").unique()
    factor_names = list(factor_panel.columns)
    n = len(factor_names)

    sum_corr = np.zeros((n, n))
    n_valid = 0

    for code in stocks:
        ts = factor_panel.xs(code, level="code")
        ts = ts.dropna()
        if len(ts) < 20:
            continue
        corr = ts.corr(method="pearson").values
        corr = np.nan_to_num(corr, nan=0.0)
        sum_corr += corr
        n_valid += 1

    avg_corr = sum_corr / max(n_valid, 1)
    return pd.DataFrame(avg_corr, index=factor_names, columns=factor_names)


def greedy_diverse_selection(
    corr_matrix: pd.DataFrame,
    n_select: int = 10,
    max_corr_threshold: float = 0.7,
    start_factor: str | None = None,
    mandatory: list[str] | None = None,
) -> list[str]:
    """Greedy select factors with minimal pairwise redundancy.

    Algorithm:
    1. Start with mandatory factors (if any), then the factor with lowest
       average absolute correlation (or specified start_factor).
    2. At each step, add the unselected factor whose max absolute correlation
       with already-selected factors is lowest.
    3. Stop when n_select reached or all remaining factors exceed max_corr_threshold.

    Parameters
    ----------
    mandatory : list[str], optional
        Factors that must be included regardless of correlation.
    """
    factors = list(corr_matrix.columns)
    abs_corr = corr_matrix.abs().values

    selected = list(mandatory) if mandatory is not None else []

    # Validate mandatory factors exist
    for f in selected:
        if f not in factors:
            raise ValueError(f"Mandatory factor '{f}' not found in correlation matrix")

    if start_factor is not None and start_factor not in selected:
        selected.append(start_factor)

    if not selected:
        # Start with factor having lowest average absolute correlation
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
            # Max absolute correlation with any selected factor
            max_corr = max(abs_corr[f_idx, s_idx] for s_idx in sel_indices)
            if max_corr < best_score:
                best_score = max_corr
                best_factor = f

        if best_factor is None or best_score > max_corr_threshold:
            # Remaining factors are all too correlated with the set
            break

        selected.append(best_factor)
        remaining.remove(best_factor)

    return selected


def print_correlation_summary(corr_matrix: pd.DataFrame, selected: list[str]):
    """Print a human-readable summary of the correlation matrix and selection."""
    sel_set = set(selected)
    sel_indices = [list(corr_matrix.columns).index(s) for s in selected]

    print("=== Factor Correlation Summary ===")
    print(f"Total factors: {len(corr_matrix)}")
    print(f"Selected: {len(selected)}/{len(corr_matrix)}")
    print()

    # Per-factor average cross-correlation
    abs_corr = corr_matrix.abs()
    avg_corr = abs_corr.mean(axis=1).sort_values()
    print("Factors ranked by average absolute correlation (lower = more unique):")
    print(f"{'Factor':<25} {'Avg |r|':>8} {'Selected':>10}")
    print("-" * 45)
    for f in avg_corr.index:
        marker = " [SELECTED]" if f in sel_set else ""
        print(f"{f:<25} {avg_corr[f]:>8.3f} {marker}")
    print()

    # Pairwise correlations within selected set
    if len(selected) > 1:
        sel_corr = corr_matrix.loc[selected, selected]
        max_pair = sel_corr.where(~np.eye(len(selected), dtype=bool)).max().max()
        print(f"Max pairwise |r| among selected: {max_pair:.3f}")
        print()

    # Check for near-duplicate pairs
    print("Highly correlated pairs (|r| > 0.8):")
    printed = set()
    for i, f1 in enumerate(corr_matrix.columns):
        for j, f2 in enumerate(corr_matrix.columns):
            if i >= j:
                continue
            r = abs_corr.iloc[i, j]
            if r > 0.8:
                key = tuple(sorted([f1, f2]))
                if key not in printed:
                    printed.add(key)
                    sel1 = " [selected]" if f1 in sel_set else ""
                    sel2 = " [selected]" if f2 in sel_set else ""
                    print(f"  {f1:<25} ~ {f2:<25}  r = {r:.3f}{sel1}{sel2}")
    if not printed:
        print("  (none)")


def main(
    db_path: str | Path | None = None,
    n_select: int = 12,
    max_corr: float = 0.7,
    max_stocks: int | None = None,
    store: bool = True,
    mandatory_factors: list[str] | None = None,
):
    """
    Full pipeline: compute factors → correlation analysis → select diverse subset → store.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    # Step 1: Compute factor panel
    log.info("Step 1: Computing factor panel...")
    panel = compute_panel(db_path=db_path, max_stocks=max_stocks)
    if panel.empty:
        log.error("No data available. Is the database built?")
        return

    log.info("  Panel shape: %s, %d stocks, %d dates",
             panel.shape, panel.index.get_level_values("code").nunique(),
             panel.index.get_level_values("date").nunique())

    # Remove factors with too many NaNs
    nan_frac = panel.isna().mean(axis=0)
    bad_factors = nan_frac[nan_frac > 0.5].index.tolist()
    if bad_factors:
        log.warning("  Dropping %d factors with >50%% NaN: %s", len(bad_factors), bad_factors)
        panel = panel.drop(columns=bad_factors)

    # Step 2: Cross-sectional rank correlation
    log.info("Step 2: Computing cross-sectional rank correlation matrix...")
    corr_xs = cross_sectional_rank_corr(panel)
    log.info("  Done.")

    # Step 3: Greedy selection
    log.info("Step 3: Greedy selecting %d diverse factors (max_corr=%.2f)...", n_select, max_corr)
    selected = greedy_diverse_selection(corr_xs, n_select=n_select, max_corr_threshold=max_corr, mandatory=mandatory_factors)
    log.info("  Selected: %s", selected)

    # Step 4: Summary
    print_correlation_summary(corr_xs, selected)

    # Step 5: Store to DB
    if store and selected:
        log.info("Step 4: Storing selected factors into DuckDB table 'factor_values'...")
        selected_panel = panel[selected]
        con = duckdb.connect(str(db_path or Path(__file__).resolve().parent.parent / "data" / "ashare.duckdb"))
        store_factors(selected_panel, table_name="factor_values", con=con)
        con.close()
        log.info("  Done.")

    # Also store full factor correlation matrix for reference
    if store:
        con = duckdb.connect(str(db_path or Path(__file__).resolve().parent.parent / "data" / "ashare.duckdb"))
        con.execute("DROP TABLE IF EXISTS factor_correlation")
        con.execute("CREATE TABLE factor_correlation AS SELECT * FROM corr_xs")
        con.close()
        log.info("Correlation matrix stored in 'factor_correlation' table.")

    return selected



if __name__ == "__main__":
    # Hardcoded 22 curated factors -- skip greedy selection
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)

    from .compute import compute_panel, store_factors

    panel = compute_panel(max_stocks=None)
    if panel.empty:
        log.error("No data available")
        sys.exit(1)

    CURATED_FACTORS = [
        # 动量 (4)
        "Return_1d", "Return_5d", "Return_20d", "Reversal_60d",
        # 波动率 (5)
        "ATR", "Volatility", "Volatility_60d",
        "Bollinger_width", "alpha060",
        # 价格位置 & 其他 (6)
        "Price_position_252d", "Stochastic_K",
        "Return_skew_20d", "Trend_strength",
        "SMA", "MACD_signal",
        # 形态/日内 (3)
        "Gap_pct", "Body_pct", "Intraday_range_pct",
        # 量/流动性 (2)
        "Volume_ratio", "Amihud_illiquidity",
        # Alpha 复合 (13)
        "alpha001", "alpha002", "alpha003", "alpha006", "alpha009",
        "alpha012", "alpha013", "alpha014", "alpha019", "alpha020",
        "alpha050", "alpha101", "alpha191",
    ]

    available = [f for f in CURATED_FACTORS if f in panel.columns]
    missing = [f for f in CURATED_FACTORS if f not in panel.columns]
    if missing:
        log.warning("Factors not in panel: %s", missing)

    selected_panel = panel[available]
    log.info("Storing %d/%d factors (%d rows)", len(available), len(CURATED_FACTORS), selected_panel.shape[0])

    con = duckdb.connect(str(Path(__file__).resolve().parent.parent / "data" / "ashare.duckdb"))
    store_factors(selected_panel, table_name="factor_values", con=con)
    con.close()
    log.info("Done.")
