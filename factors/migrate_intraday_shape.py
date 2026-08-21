# -*- coding: utf-8 -*-
"""
One-time migration: add the 8 intraday-shape factor columns to factor_values
and backfill them over full history (bench 2026-08, plan Task 3).

- ALTER TABLE ... ADD COLUMN IF NOT EXISTS 语义（缺列才加，幂等）
- 回填走轻量路径：仅 OHLCV + daily_basic + stock_info 三源，调
  compute_non_alpha_factors（公式单一来源，与 compute/update 增量路径同源），
  不算 Alpha101/行业/筹码
- 只 UPDATE 这 8 列，不触碰其他因子列与 ai_gz2000_*（列所有权）
- 表内已有行才更新：股票级历史缺口（池扩容新成员）的行不存在，仍由
  --backfill-stocks 机制负责——届时 8 列已在 compute 路径内自动带上

Usage:
    python -m factors.migrate_intraday_shape --dry-run   # 只打印计划
    python -m factors.migrate_intraday_shape             # 执行 ALTER + 回填
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_PATH, get_pool_codes
from .compute import _load_ohlcv, _load_market_cap, _load_stock_info
from .extra_factors import compute_non_alpha_factors

log = logging.getLogger(__name__)

NEW_COLUMNS = [
    "UpperShadow", "LowerShadow", "ClosePos", "OpenPos",
    "ShadowRatio", "RangeEfficiency", "ClosePos_mean_20d", "ClosePos_std_20d",
]


def build_new_columns(con: duckdb.DuckDBPyConnection, codes: list[str]):
    """Full-history polars panel of just the 8 new columns for `codes`."""
    import polars as pl

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

    extra = compute_non_alpha_factors(df)
    keep = ["datetime", "vt_symbol"] + NEW_COLUMNS
    missing = [c for c in keep if c not in extra.columns]
    if missing:
        raise RuntimeError(f"compute_non_alpha_factors did not produce: {missing}")
    panel = extra.select(keep).rename({"vt_symbol": "code", "datetime": "date"})
    return panel.unique(subset=["code", "date"], keep="last")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印计划，不 ALTER 不计算不写库")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    codes = get_pool_codes()
    log.info("Pool: %d stocks", len(codes))

    con = duckdb.connect(str(DB_PATH))
    con.execute("SET threads = 4")
    try:
        existing = {r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='factor_values'").fetchall()}
        to_add = [c for c in NEW_COLUMNS if c not in existing]
        n_fv = con.execute("SELECT COUNT(*) FROM factor_values").fetchone()[0]
        n_ai = con.execute(
            "SELECT COUNT(ai_gz2000_20d) FROM factor_values").fetchone()[0]
        log.info("factor_values: %d rows; columns to add: %s", n_fv, to_add or "(none)")

        if args.dry_run:
            log.info("[dry-run] would ALTER ADD %d columns, then backfill 8 cols "
                     "over pool full history (~%d existing rows)", len(to_add), n_fv)
            return

        for c in to_add:
            con.execute(f"ALTER TABLE factor_values ADD COLUMN {c} DOUBLE")
            log.info("ALTER TABLE factor_values ADD COLUMN %s DOUBLE", c)

        t0 = time.time()
        panel = build_new_columns(con, codes)
        if panel.is_empty():
            log.warning("Empty panel, nothing to backfill.")
            return
        log.info("panel built: %d rows in %.1fs", len(panel), time.time() - t0)

        pdf = panel.to_pandas()
        pdf["date"] = pdf["date"].astype(str).str[:10]
        pdf = pdf.sort_values(["date", "code"])
        con.execute("CREATE OR REPLACE TEMP TABLE _shape_upd AS SELECT * FROM pdf")

        n_match = con.execute("""
            SELECT COUNT(*) FROM _shape_upd p
            JOIN factor_values f ON f.code = p.code AND f.date = p.date
        """).fetchone()[0]
        n_fv_panel_only = con.execute("""
            SELECT COUNT(*) FROM _shape_upd p
            WHERE NOT EXISTS (SELECT 1 FROM factor_values f
                              WHERE f.code = p.code AND f.date = p.date)
        """).fetchone()[0]

        set_clause = ", ".join(f"{c} = p.{c}" for c in NEW_COLUMNS)
        con.execute(f"""
            UPDATE factor_values f SET {set_clause}
            FROM _shape_upd p
            WHERE f.code = p.code AND f.date = p.date
        """)
        con.execute("CHECKPOINT")

        # ---- post-verify：行数不变、ai 列不动、8 列非空率 ----
        n_fv2 = con.execute("SELECT COUNT(*) FROM factor_values").fetchone()[0]
        n_ai2 = con.execute(
            "SELECT COUNT(ai_gz2000_20d) FROM factor_values").fetchone()[0]
        assert n_fv2 == n_fv, f"row count changed: {n_fv} -> {n_fv2}"
        assert n_ai2 == n_ai, f"ai_gz2000_20d non-null count changed: {n_ai} -> {n_ai2}"
        log.info("updated matched rows: %d (panel-only rows skipped: %d)", n_match, n_fv_panel_only)
        for c in NEW_COLUMNS:
            nn = con.execute(f"SELECT COUNT({c}) FROM factor_values").fetchone()[0]
            log.info("  %-18s non-null %7d (%.2f%%)", c, nn, nn / n_fv2 * 100)
    finally:
        con.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
