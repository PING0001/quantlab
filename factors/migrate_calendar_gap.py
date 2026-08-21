# -*- coding: utf-8 -*-
"""
One-time migration: add DaysToNextTrading to factor_values and backfill full
history (bench 2026-08-21).

- 值 = 今天到下一交易日之间的休市天数（明天开市=0；普通周五=2；节前最后
  交易日=假期长度）；公式单一来源 compute._load_days_to_next_trading，
  trading_calendar（is_open 序列相邻差-1）唯一真相源
- ALTER ADD 缺列才加（幂等）；只 UPDATE 这 1 列，不触碰其他因子列与
  ai_gz2000_*（列所有权）
- 日历末位交易日（当前 2026-08-21）无下一日 → NULL，属预期诚实留空，
  日历随 pull 延展后由 compute/update 增量路径自然补上

Usage:
    python -m factors.migrate_calendar_gap --dry-run   # 只打印计划
    python -m factors.migrate_calendar_gap             # 执行 ALTER + 回填
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_PATH
from .compute import _load_days_to_next_trading

log = logging.getLogger(__name__)

COLUMN = "DaysToNextTrading"


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
        log.info("factor_values: %d rows; column to add: %s",
                 n_fv, COLUMN if COLUMN not in existing else "(none)")

        gap = _load_days_to_next_trading(con)
        log.info("calendar mapping: %d dates (max %s)",
                 len(gap), gap["datetime"].max())
        if args.dry_run:
            log.info("[dry-run] would ALTER ADD %s, then UPDATE by date mapping", COLUMN)
            return

        if COLUMN not in existing:
            con.execute(f"ALTER TABLE factor_values ADD COLUMN {COLUMN} DOUBLE")
            log.info("ALTER TABLE factor_values ADD COLUMN %s DOUBLE", COLUMN)

        con.execute("CREATE OR REPLACE TEMP TABLE _gap(d VARCHAR, v DOUBLE)")
        con.executemany(
            "INSERT INTO _gap VALUES (?, ?)",
            [(r["datetime"], float(r["DaysToNextTrading"])) for r in gap.to_dicts()],
        )

        n_upd = con.execute(f"""
            UPDATE factor_values f SET {COLUMN} = t.v
            FROM _gap t WHERE f.date = t.d
        """).fetchone()[0]
        con.execute("CHECKPOINT")

        # ---- post-verify：行数不变、ai 列不动、非空=全表-日历末日行 ----
        n_fv2 = con.execute("SELECT COUNT(*) FROM factor_values").fetchone()[0]
        n_ai2 = con.execute(
            "SELECT COUNT(ai_gz2000_20d) FROM factor_values").fetchone()[0]
        nn = con.execute(f"SELECT COUNT({COLUMN}) FROM factor_values").fetchone()[0]
        # 自洽校验：非空数应等于"日期出现在映射表中的行数"（日历真末位交易日
        # 无映射 → 若有因子行则留 NULL；当前 08-21 无因子行 → 全表非空）
        n_mapped = con.execute("""
            SELECT COUNT(*) FROM factor_values f
            WHERE EXISTS (SELECT 1 FROM _gap t WHERE t.d = f.date)
        """).fetchone()[0]
        n_last = n_fv2 - n_mapped
        assert n_fv2 == n_fv, f"row count changed: {n_fv} -> {n_fv2}"
        assert n_ai2 == n_ai, f"ai_gz2000_20d non-null count changed: {n_ai} -> {n_ai2}"
        log.info("updated rows: %d; %s non-null %d (无映射行 %d，日历末位交易日属预期)",
                 n_upd, COLUMN, nn, n_last)
        assert nn == n_mapped, f"non-null {nn} != mapped {n_mapped}"

        sample = con.execute(f"""
            SELECT DISTINCT date, {COLUMN} FROM factor_values
            WHERE date IN ('2026-08-14','2026-08-20','2025-09-30','2026-02-13','2026-08-13')
            ORDER BY date
        """).fetchall()
        log.info("sample (expect 0/8/2/10/0 按日期序): %s", sample)
    finally:
        con.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
