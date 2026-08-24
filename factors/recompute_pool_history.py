# -*- coding: utf-8 -*-
"""阶段 1：因子按历史池成员重算（池时点化工程，2026-08-24）。

裁定：回补范围 B——只重算各档成员在其档期内的行。横截面类因子（排名/
个股-池协同等）的历史值此前按"并集宇宙"计算，必须按各档当时的池成员
截面重算才时点；非成员期的行保留旧值（阶段 2 的成员资格过滤会把它们
挡在训练/回测之外）。

窗口：2019-12-02 档起（模型训练 2020-01 起，再早无人消费）。
机制：逐档 compute_panel(成员, lookback→档期末) + 截到档期 + write_panel
（行级 INSERT 缺失成员 / 列所有权 UPDATE 横截面列；nn_/gb_/mf_ 等他方
列不受影响）。

Usage:
    python -m factors.recompute_pool_history
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DB_PATH
from factors.compute import compute_panel
from factors.update import get_lookback_start, write_panel

START_EFFECTIVE = "2019-12-02"   # 首档（覆盖训练窗 2020-01 起）


def main():
    t0 = time.time()
    con = duckdb.connect(str(DB_PATH))

    periods = con.execute("""
        SELECT effective_date, cutoff_date, list(code) AS codes
        FROM pool_snapshots
        WHERE effective_date >= ?
        GROUP BY effective_date, cutoff_date
        ORDER BY effective_date
    """, [START_EFFECTIVE]).fetchall()
    if not periods:
        raise SystemExit("pool_snapshots 无档期，先跑 pools/build_pool_history.py")

    kline_max = str(con.execute("SELECT max(date) FROM daily_kline").fetchone()[0])[:10]
    log_rows = []
    for i, (eff, cut, codes) in enumerate(periods):
        next_eff = periods[i + 1][0] if i + 1 < len(periods) else "9999-12-31"
        members = list(codes)
        lookback = get_lookback_start(con, eff)
        panel = compute_panel(con, members, start_date=lookback, end_date=next_eff)
        if panel.is_empty():
            print(f"  [SKIP] {eff}: 空面板")
            continue
        panel = panel.filter(
            (pl.col("date") >= eff) & (pl.col("date") < next_eff))
        n_ins, n_upd = write_panel(con, panel.to_pandas())
        print(f"  {eff} ~ {min(next_eff, kline_max)}: 成员 {len(members)} 只, "
              f"写入 INSERT {n_ins:,} / UPDATE {n_upd:,} 行 "
              f"({time.time() - t0:.0f}s 累计)")
        log_rows.append((eff, len(members), n_ins, n_upd))

    con.execute("CHECKPOINT")
    con.close()
    tot_ins = sum(r[2] for r in log_rows)
    tot_upd = sum(r[3] for r in log_rows)
    print(f"\n完成 {len(log_rows)} 档：INSERT {tot_ins:,} + UPDATE {tot_upd:,} 行，"
          f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
