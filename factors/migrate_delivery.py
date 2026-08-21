# -*- coding: utf-8 -*-
"""
One-time migration: add DaysToDelivery to factor_values and backfill full
history (bench 2026-08-21).

- 纯日期因子：值 = 距下一个股指期货交割日（每月第三个周五）的自然日数，
  交割日当天 = 0；公式单一来源 extra_factors.days_to_delivery
- ALTER ADD 缺列才加（幂等）；只 UPDATE 这 1 列，不触碰其他因子列与
  ai_gz2000_*（列所有权）
- 比 intraday_shape 迁移轻：无需 OHLCV 面板，按 distinct date 建映射直接 UPDATE

Usage:
    python -m factors.migrate_delivery --dry-run   # 只打印计划
    python -m factors.migrate_delivery             # 执行 ALTER + 回填
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_PATH
from .extra_factors import days_to_delivery

log = logging.getLogger(__name__)

COLUMN = "DaysToDelivery"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印计划，不 ALTER 不写库")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    con = duckdb.connect(str(DB_PATH))
    try:
        existing = {r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='factor_values'").fetchall()}
        n_fv = con.execute("SELECT COUNT(*) FROM factor_values").fetchone()[0]
        n_ai = con.execute(
            "SELECT COUNT(ai_gz2000_20d) FROM factor_values").fetchone()[0]
        n_dates = con.execute(
            "SELECT COUNT(DISTINCT date) FROM factor_values").fetchone()[0]
        log.info("factor_values: %d rows, %d distinct dates; column to add: %s",
                 n_fv, n_dates, COLUMN if COLUMN not in existing else "(none)")

        if args.dry_run:
            log.info("[dry-run] would ALTER ADD %s, then UPDATE by date mapping", COLUMN)
            return

        if COLUMN not in existing:
            con.execute(f"ALTER TABLE factor_values ADD COLUMN {COLUMN} DOUBLE")
            log.info("ALTER TABLE factor_values ADD COLUMN %s DOUBLE", COLUMN)

        dates = [r[0] for r in con.execute(
            "SELECT DISTINCT date FROM factor_values ORDER BY date").fetchall()]
        rows = [(d, days_to_delivery(date.fromisoformat(d))) for d in dates]
        con.execute(
            f"CREATE OR REPLACE TEMP TABLE _deliv(d VARCHAR, v DOUBLE)")
        con.executemany("INSERT INTO _deliv VALUES (?, ?)", rows)

        n_upd = con.execute(f"""
            UPDATE factor_values f SET {COLUMN} = t.v
            FROM _deliv t WHERE f.date = t.d
        """).fetchone()[0]
        con.execute("CHECKPOINT")

        # ---- post-verify：行数不变、ai 列不动、新列全覆盖 ----
        n_fv2 = con.execute("SELECT COUNT(*) FROM factor_values").fetchone()[0]
        n_ai2 = con.execute(
            "SELECT COUNT(ai_gz2000_20d) FROM factor_values").fetchone()[0]
        nn = con.execute(f"SELECT COUNT({COLUMN}) FROM factor_values").fetchone()[0]
        assert n_fv2 == n_fv, f"row count changed: {n_fv} -> {n_fv2}"
        assert n_ai2 == n_ai, f"ai_gz2000_20d non-null count changed: {n_ai} -> {n_ai2}"
        log.info("updated rows: %d; %s non-null %d (%.2f%%)",
                 n_upd, COLUMN, nn, nn / n_fv2 * 100)
        assert nn == n_fv2, f"non-null {nn} != total {n_fv2}（存在未覆盖日期）"

        sample = con.execute(f"""
            SELECT DISTINCT date, {COLUMN} FROM factor_values
            WHERE date IN ('2026-08-14','2026-08-21','2026-08-22',
                           '2026-09-18','2026-01-30','2026-02-20')
            ORDER BY date
        """).fetchall()
        log.info("sample (expect 7/0/27/0/21/0): %s", sample)
    finally:
        con.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
