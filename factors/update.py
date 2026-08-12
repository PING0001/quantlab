# -*- coding: utf-8 -*-
"""
Incremental factor update: compute factors for new dates only,
then INSERT into factor_values table.

Rolling windows used in alpha expressions (e.g. ts_sum(returns, 250),
ts_std(close, 250)) require historical context. We load data from
(LOOKBACK_DAYS) before the last computed date, then only write
the NEW date rows to factor_values.

Usage:
    python -m factors.update
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import duckdb
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_PATH, get_pool_codes
from .compute import compute_panel

log = logging.getLogger(__name__)

LOOKBACK_DAYS = 260  # trading days (~1 year, covers 250d windows + margin)


def get_latest_date(con: duckdb.DuckDBPyConnection) -> str | None:
    """Get the maximum date currently in factor_values."""
    try:
        result = con.execute("SELECT max(date) FROM factor_values").fetchone()
        if result and result[0]:
            return str(result[0])
    except Exception:
        pass
    return None


def get_lookback_start(con: duckdb.DuckDBPyConnection, latest: str) -> str:
    """Get the date LOOKBACK_DAYS trading days before *latest*."""
    result = con.execute(
        "SELECT date FROM daily_kline WHERE date <= ? "
        "ORDER BY date DESC LIMIT 1 OFFSET ?",
        [latest, LOOKBACK_DAYS - 1],
    ).fetchone()
    if result and result[0]:
        return str(result[0])
    return latest


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    codes = get_pool_codes()
    log.info("Pool: %d stocks", len(codes))

    con = duckdb.connect(str(DB_PATH))
    con.execute("SET threads = 4")

    latest = get_latest_date(con)
    if not latest:
        log.warning("factor_values table is empty. Run full compute first.")
        con.close()
        return

    log.info("Last factor date: %s", latest)

    # Check for new dates in daily_kline beyond factor_values
    max_kline = con.execute("SELECT max(date) FROM daily_kline").fetchone()
    if not max_kline or not max_kline[0]:
        log.info("No kline data available.")
        con.close()
        return

    max_kline_date = str(max_kline[0])
    if max_kline_date <= latest:
        log.info("Factors are up to date (kline: %s).", max_kline_date)
        con.close()
        return

    # Compute from lookback start to latest kline date
    lookback_start = get_lookback_start(con, latest)

    log.info("Incremental range: lookback %s → kline max %s", lookback_start, max_kline_date)

    panel = compute_panel(con, codes, start_date=lookback_start)

    if panel.is_empty():
        log.info("No new data to compute.")
        con.close()
        return

    # Filter to only NEW dates (beyond *latest*)
    panel = panel.filter(pl.col("date") > latest)

    if panel.is_empty():
        log.info("No new factor rows beyond %s.", latest)
        con.close()
        return

    # Deduplicate on (code, date): compute_panel can emit duplicate rows when
    # a stock has overlapping ST/*ST namechange records (both multiply through
    # the IsST join). store_factor_values (full path) dedups via panel.unique();
    # the incremental path must do the same.
    n_before = len(panel)
    panel = panel.unique(subset=["code", "date"], keep="last")
    n_after = len(panel)
    if n_after < n_before:
        log.warning("Deduplicated new panel: %d -> %d rows (-%d duplicates)",
                    n_before, n_after, n_before - n_after)

    n_new = len(panel)
    n_dates = panel["date"].n_unique() if "date" in panel.columns else 0
    log.info("New factor rows: %d rows, %d dates", n_new, n_dates)

    # Convert to pandas and insert
    pdf = panel.to_pandas()
    pdf = pdf.sort_values(["date", "code"])

    # Remove any rows already in factor_values (safety check)
    existing = con.execute(
        "SELECT date, code FROM factor_values WHERE date > ?", [latest]
    ).fetchdf()
    if not existing.empty:
        existing_set = set(
            zip(existing["date"].astype(str), existing["code"])
        )
        pdf = pdf[
            ~pdf.apply(
                lambda r: (str(r["date"]), r["code"]) in existing_set, axis=1
            )
        ]

    if pdf.empty:
        log.info("All new rows already exist in factor_values.")
    else:
        # Insert only columns that exist in both the table and new data
        existing_cols = {r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='factor_values'"
        ).fetchall()}
        insert_cols = [c for c in pdf.columns if c in existing_cols]
        pdf_sel = pdf[insert_cols]
        cols_str = ", ".join(insert_cols)
        con.execute(f"INSERT INTO factor_values ({cols_str}) SELECT * FROM pdf_sel")
        con.execute("CHECKPOINT")
        log.info("Inserted %d rows into factor_values.", len(pdf_sel))

    con.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
