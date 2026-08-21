# -*- coding: utf-8 -*-
"""
One-time migration: backfill DaysToDelivery in factor_values (bench 2026-08-21).

- v2（日历感知）：交割日 = 每月第三个周五，逢法定假日顺延至下一交易日
  （CFFEX 规则；2008-2026 共 9 个月顺延，均为春节/中秋撞期）。公式单源
  compute._load_days_to_delivery，trading_calendar 唯一真相源
- 首版为纯日期规则（不识别顺延），本次重刷修正 9 个受影响月份
- 只 UPDATE 这 1 列，不触碰其他因子列与 ai_gz2000_*（列所有权）

Usage:
    python -m factors.migrate_delivery --dry-run   # 只打印计划
    python -m factors.migrate_delivery             # 执行回填
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_PATH
from .compute import _load_days_to_delivery

log = logging.getLogger(__name__)

COLUMN = "DaysToDelivery"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印计划，不写库")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    con = duckdb.connect(str(DB_PATH))
    try:
        n_fv = con.execute("SELECT COUNT(*) FROM factor_values").fetchone()[0]
        n_ai = con.execute(
            "SELECT COUNT(ai_gz2000_20d) FROM factor_values").fetchone()[0]
        log.info("factor_values: %d rows", n_fv)

        deliv = _load_days_to_delivery(con)
        log.info("delivery mapping: %d dates (含 9 个顺延月)", len(deliv))
        if args.dry_run:
            log.info("[dry-run] would UPDATE %s by date mapping", COLUMN)
            return

        con.execute("CREATE OR REPLACE TEMP TABLE _deliv(d VARCHAR, v DOUBLE)")
        con.executemany(
            "INSERT INTO _deliv VALUES (?, ?)",
            [(r["datetime"], float(r["DaysToDelivery"])) for r in deliv.to_dicts()],
        )

        n_upd = con.execute(f"""
            UPDATE factor_values f SET {COLUMN} = t.v
            FROM _deliv t WHERE f.date = t.d
        """).fetchone()[0]
        con.execute("CHECKPOINT")

        # ---- post-verify：行数不变、ai 列不动、非空=有映射行 ----
        n_fv2 = con.execute("SELECT COUNT(*) FROM factor_values").fetchone()[0]
        n_ai2 = con.execute(
            "SELECT COUNT(ai_gz2000_20d) FROM factor_values").fetchone()[0]
        nn = con.execute(f"SELECT COUNT({COLUMN}) FROM factor_values").fetchone()[0]
        n_mapped = con.execute("""
            SELECT COUNT(*) FROM factor_values f
            WHERE EXISTS (SELECT 1 FROM _deliv t WHERE t.d = f.date)
        """).fetchone()[0]
        assert n_fv2 == n_fv and n_ai2 == n_ai, "row/ai counts changed"
        assert nn == n_mapped, f"non-null {nn} != mapped {n_mapped}"
        log.info("updated rows: %d; non-null %d (无映射 %d，日历尾月属预期)",
                 n_upd, nn, n_fv2 - nn)

        sample = con.execute(f"""
            SELECT DISTINCT date, {COLUMN} FROM factor_values
            WHERE date IN ('2026-02-13','2026-02-24','2010-02-12','2010-02-22',
                           '2026-07-17','2025-08-15','2026-06-19','2026-06-22')
            ORDER BY date
        """).fetchall()
        log.info("sample (顺延月: 13→11/24→0/12→10/22→0; 正常月: 17→0/15→0; "
                 "假日第三周五 19 无行, 22→0): %s", sample)
    finally:
        con.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
