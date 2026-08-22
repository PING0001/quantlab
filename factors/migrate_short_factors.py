# -*- coding: utf-8 -*-
"""
One-time migration: add the 6 short-window factor variants to factor_values
and backfill full history (bench 2026-08-22, 用户要求"加入 6 个左右原有因子
的小周期形式"供 open2d 短 horizon 模型筛选取用).

NEW_COLUMNS = Return_3d, Volatility_3d, Amihud_3d, AvgAmount_3d,
              ClosePos_mean_3d, Price_position_5d

- 公式单一来源 compute_non_alpha_factors（compute/update 增量路径同源自动带上）
- ALTER 缺列才加（幂等）；只 UPDATE 这 6 列，不触碰其他因子列与 ai_gz2000_*
- 前置条件：无其他进程持有 DB（读连接也不行，DuckDB 单写者锁）

Usage:
    python -m factors.migrate_short_factors --dry-run
    python -m factors.migrate_short_factors
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

NEW_COLUMNS = ["Return_3d", "Volatility_3d", "Amihud_3d",
               "AvgAmount_3d", "ClosePos_mean_3d", "Price_position_5d"]


def build_new_columns(con: duckdb.DuckDBPyConnection, codes: list[str]):
    """Full-history polars panel of just the new columns for `codes`."""
    df = _load_ohlcv(con, codes)
    if df.is_empty():
        return None
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
    log.info("Pool: %d stocks | new columns: %s", len(codes), NEW_COLUMNS)

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
            log.info("[dry-run] would ALTER ADD %d columns then backfill 6 cols", len(to_add))
            return

        for c in to_add:
            con.execute(f"ALTER TABLE factor_values ADD COLUMN {c} DOUBLE")
            log.info("ALTER TABLE factor_values ADD COLUMN %s DOUBLE", c)

        t0 = time.time()
        panel = build_new_columns(con, codes)
        if panel is None or panel.is_empty():
            log.warning("Empty panel, nothing to backfill.")
            return
        log.info("panel built: %d rows in %.1fs", len(panel), time.time() - t0)

        pdf = panel.to_pandas()
        pdf["date"] = pdf["date"].astype(str).str[:10]
        pdf = pdf.sort_values(["date", "code"])
        con.execute("CREATE OR REPLACE TEMP TABLE _short_upd AS SELECT * FROM pdf")

        n_match = con.execute("""
            SELECT COUNT(*) FROM _short_upd p
            JOIN factor_values f ON f.code = p.code AND f.date = p.date
        """).fetchone()[0]

        set_clause = ", ".join(f"{c} = p.{c}" for c in NEW_COLUMNS)
        con.execute(f"""
            UPDATE factor_values f SET {set_clause}
            FROM _short_upd p
            WHERE f.code = p.code AND f.date = p.date
        """)
        con.execute("CHECKPOINT")

        # ---- post-verify：行数不变、ai 列不动、新列非空率 ----
        n_fv2 = con.execute("SELECT COUNT(*) FROM factor_values").fetchone()[0]
        n_ai2 = con.execute(
            "SELECT COUNT(ai_gz2000_20d) FROM factor_values").fetchone()[0]
        assert n_fv2 == n_fv, f"row count changed: {n_fv} -> {n_fv2}"
        assert n_ai2 == n_ai, f"ai_gz2000_20d non-null count changed: {n_ai} -> {n_ai2}"
        log.info("updated matched rows: %d", n_match)
        for c in NEW_COLUMNS:
            nn = con.execute(f"SELECT COUNT({c}) FROM factor_values").fetchone()[0]
            log.info("  %-20s non-null %7d (%.2f%%)", c, nn, nn / n_fv2 * 100)
    finally:
        con.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
