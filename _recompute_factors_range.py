# -*- coding: utf-8 -*-
"""One-off: 重算 2026-07-13~2026-08-17 被 lookback 窗口 bug 污染的因子。

事故：factors/update.py 的 get_lookback_start 原实现 `LIMIT 1 OFFSET 259`
作用在行级 daily_kline（约5500行/天）上，返回 from_date 当天而非 260 个
交易日前——增量路径每天只装 1 天历史，长窗口因子全 NULL、min_samples=1
类因子短窗错值。同时 GZ2000 九列特征的计算代码已丢失（从未入 git），
入模的 3 个 GZ2000 因子自 07-13 起全 NULL。

本脚本用修复后的代码（DISTINCT 日历 lookback + 重建的 GZ2000 计算）
走 write_panel 的 UPDATE 路径重写目标区间（保留 ai_gz2000_* 列）。

用法：
    python _recompute_factors_range.py --from 2026-07-13 --to 2026-08-17
"""
from __future__ import annotations

import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", required=True)
    ap.add_argument("--to", dest="end", required=True)
    args = ap.parse_args()

    con = duckdb.connect(str(DB_PATH))
    con.execute("SET threads = 4")
    try:
        codes = get_pool_codes()
        dates = [str(r[0])[:10] for r in con.execute(
            "SELECT DISTINCT date FROM factor_values WHERE date BETWEEN ? AND ? ORDER BY date",
            [args.start, args.end]).fetchall()]
        assert dates, "target range empty"
        lookback = get_lookback_start(con, dates[0])
        log.info("Recompute %d dates (%s ~ %s), lookback from %s",
                 len(dates), dates[0], dates[-1], lookback)
        assert lookback < dates[0], f"lookback must precede range: {lookback}"

        panel = compute_panel(con, codes, start_date=lookback)
        panel = (panel.filter(pl.col("date").is_in(dates))
                      .unique(subset=["code", "date"], keep="last"))
        log.info("Panel rows to update: %d", len(panel))

        pdf = panel.to_pandas().sort_values(["date", "code"])
        n_ins, n_upd = write_panel(con, pdf)
        con.execute("CHECKPOINT")
        log.info("Done: ins=%d upd=%d", n_ins, n_upd)
    finally:
        con.close()


if __name__ == "__main__":
    main()
