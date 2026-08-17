# -*- coding: utf-8 -*-
"""One-off: 回补 2026-07-14~2026-08-10 期间因 cyq_perf 缺失而为 NULL 的筹码因子。

背景：pull_adj 时代 cyq 16 个开市日拉空无人补救，期间 factor_values 的
WinnerRate/CostPosition/ChipDispersion/ChipSkew 为 NULL。现 cyq_perf 已由
`python -m data.pull --reconcile` 补齐，本脚本重算这 16 天的因子并经
write_panel 的 UPDATE 路径写回（保留 ai_gz2000_* 列）。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DB_PATH, get_pool_codes
from factors.compute import compute_panel
from factors.update import get_lookback_start, write_panel

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATES = ["2026-07-14", "2026-07-17", "2026-07-22", "2026-07-23", "2026-07-24",
         "2026-07-27", "2026-07-28", "2026-07-29", "2026-07-30", "2026-07-31",
         "2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07",
         "2026-08-10"]


def main():
    con = duckdb.connect(str(DB_PATH))
    con.execute("SET threads = 4")
    try:
        codes = get_pool_codes()
        lookback = get_lookback_start(con, min(DATES))
        log.info("Backfilling %d dates (%s ~ %s), lookback from %s",
                 len(DATES), DATES[0], DATES[-1], lookback)

        before = con.execute(
            f"SELECT COUNT(*) FROM factor_values "
            f"WHERE date IN ({','.join('?' * len(DATES))}) AND WinnerRate IS NULL",
            DATES).fetchone()[0]

        panel = compute_panel(con, codes, start_date=lookback)
        panel = (panel.filter(pl.col("date").is_in(DATES))
                      .unique(subset=["code", "date"], keep="last"))
        log.info("Panel rows to update: %d", len(panel))

        pdf = panel.to_pandas().sort_values(["date", "code"])
        n_ins, n_upd = write_panel(con, pdf)
        con.execute("CHECKPOINT")

        after = con.execute(
            f"SELECT COUNT(*) FROM factor_values "
            f"WHERE date IN ({','.join('?' * len(DATES))}) AND WinnerRate IS NULL",
            DATES).fetchone()[0]
        ai = con.execute(
            f"SELECT COUNT(*) FROM factor_values "
            f"WHERE date IN ({','.join('?' * len(DATES))}) AND ai_gz2000_20d IS NOT NULL",
            DATES).fetchone()[0]
        log.info("Done: WinnerRate NULL %d -> %d (ins=%d upd=%d, ai kept=%d)",
                 before, after, n_ins, n_upd, ai)
    finally:
        con.close()


if __name__ == "__main__":
    main()
