# -*- coding: utf-8 -*-
"""股票池时点化：沪深300式半年度快照序列构建器（2026-08-24 用户裁定）。

规则（四项裁定 + 次新参数）：
  生效日   = 每年 6 月 / 12 月的首个交易日（简化版，不抄第二个周五）
  选样截止 = 生效日前一个月的最后一个交易日（5 月/11 月月末；模拟指数
             "提前定样"，生效日当天不依赖当天数据）
  成员规则 = 现行微盘带规则原样继承：主板（00/60 前缀兜底）+ 流通市值
             通胀调整带（1-20 亿 @2026，DECAY_RATIO 年衰减）+ 截止日
             当日有 daily_basic 行（幸存者感知：退市股自然缺席）
  次新排除 = 选样截止日时上市不满 252 个交易日（用户裁定，按交易日历计）
  基准     = 消费端另行改造（快照生效日重置等权）

产出：
  - DB 表 pool_snapshots(effective_date, cutoff_date, code)——消费端按日期取档
  - pools/mainboard_microcap_history.json（人读版，含各档人数/带域）

与现行池 json 的关系：现行 json 是历史检查点的"并集"（非时点的又一来源），
本序列的**最新一档**将取代它成为 LIVE/增量的依据（阶段 2 同步）。

Usage:
    python pools/build_pool_history.py            # 全序列重建（幂等）
    python pools/build_pool_history.py --from 2024   # 只重建指定年份起的档
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import duckdb
from dateutil.relativedelta import relativedelta

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "ashare.duckdb"
HISTORY_PATH = ROOT / "pools" / "mainboard_microcap_history.json"

# ---- 与 pools/build_microcap.py 同源的带规则（单源复制，勿漂移）----
# 2026-08-24 用户裁定：带宽放宽至 1~40 亿（流通市值；原 1~20 亿时点池仅
# 150-400 只/档，用户认为过窄）
BASE_YEAR = 2026
BASE_LOW_YI = 1.0
BASE_HIGH_YI = 40.0
TOTAL_YEARS = 11.0
DECAY_RATIO = 11.0 / 20.0
MIN_LISTED_TRADING_DAYS = 252   # 次新股门槛（用户裁定 2026-08-24）


def _decay_factor(cp_year: float) -> float:
    t = max(0.0, BASE_YEAR - cp_year)
    return DECAY_RATIO ** (t / TOTAL_YEARS)


def _decimal_year(d) -> float:
    dt = d if isinstance(d, datetime) else datetime.strptime(str(d)[:10], "%Y-%m-%d")
    days = 366 if (dt.year % 4 == 0 and dt.year % 100 != 0) or dt.year % 400 == 0 else 365
    return dt.year + (dt.timetuple().tm_yday - 1) / days


def _periods(con: duckdb.DuckDBPyConnection, from_year: int) -> list[tuple[str, str]]:
    """[(effective_date, cutoff_date)]：生效=6/12 月首个交易日，截止=前一月末
    最后一个交易日。从 from_year-06 到最新有选样数据的档。"""
    opens = [str(r[0]) for r in con.execute(
        "SELECT date FROM trading_calendar WHERE is_open = true ORDER BY date"
    ).fetchall()]
    openset = set(opens)
    out = []
    now = datetime.now()
    for y in range(from_year, now.year + 1):
        for m in (6, 12):
            # 生效日 = 该月首个交易日
            eff = next((d for d in opens if d.startswith(f"{y:04d}-{m:02d}")), None)
            if eff is None or eff > now.strftime("%Y-%m-%d"):
                continue
            # 截止日 = 生效月前一月的最后一个交易日
            pm = m - 1 if m == 12 else m - 1  # 6->5, 12->11
            py = y if m == 12 else y
            pm_str = f"{py:04d}-{pm:02d}"
            cut = max((d for d in opens if d.startswith(pm_str)), default=None)
            if cut is None:
                continue
            # 选样数据须已存在（daily_basic 截止日有行）
            has = con.execute(
                "SELECT count(*) FROM daily_basic WHERE date = ?", [cut]).fetchone()[0]
            if has == 0:
                log.warning("  %s: 选样截止 %s 无 daily_basic 数据，跳过", eff, cut)
                continue
            out.append((eff, cut))
    return out


def _listed_trading_days_before(con, opens: list[str]) -> dict[str, int]:
    """每只股票的上市日。次新判定用日历计数（SQL 侧算，避免逐股循环）。"""
    return con.execute("SELECT code, list_date FROM stock_info WHERE list_date IS NOT NULL").fetchall()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_year", type=int, default=2015,
                    help="起始年份（默认 2015，即首个档 2015-06）")
    args = ap.parse_args()

    con = duckdb.connect(str(DB_PATH))

    periods = _periods(con, args.from_year)
    log.info("档期 %d 个: %s ... %s", len(periods), periods[0][0], periods[-1][0])

    opens = [str(r[0]) for r in con.execute(
        "SELECT date FROM trading_calendar WHERE is_open = true ORDER BY date"
    ).fetchall()]

    # 次新判定：上市日起至截止日的交易日数（对每档用 SQL 数日历）
    snapshots = []
    for eff, cut in periods:
        factor = _decay_factor(_decimal_year(cut))
        low_wan = round(BASE_LOW_YI * factor * 10000, 0)
        high_wan = round(BASE_HIGH_YI * factor * 10000, 0)
        codes = [r[0] for r in con.execute(
            """
            SELECT DISTINCT b.code
            FROM daily_basic b
            LEFT JOIN stock_info s ON b.code = s.code
            WHERE COALESCE(
                      s.market,
                      CASE WHEN b.code LIKE '00%' OR b.code LIKE '60%'
                           THEN '主板' END) = '主板'
              AND b.date = ?
              AND b.circ_mv > ?
              AND b.circ_mv < ?
              AND s.list_date IS NOT NULL
              AND (SELECT count(*) FROM trading_calendar c
                   WHERE c.is_open = true
                     AND c.date > s.list_date AND c.date <= ?::DATE) >= ?
            """,
            [cut, low_wan, high_wan, cut, MIN_LISTED_TRADING_DAYS],
        ).fetchall()]
        snapshots.append({"effective_date": eff, "cutoff_date": cut,
                          "codes": sorted(codes)})
        log.info("  生效 %s（选样 %s，带 %.2f~%.2f 亿）: %d 只",
                 eff, cut, BASE_LOW_YI * factor, BASE_HIGH_YI * factor, len(codes))

    # ---- 写 DB 表 ----
    con_w = duckdb.connect(str(DB_PATH))
    con_w.execute("DROP TABLE IF EXISTS pool_snapshots")
    con_w.execute("""CREATE TABLE pool_snapshots (
        effective_date VARCHAR NOT NULL,
        cutoff_date     VARCHAR NOT NULL,
        code            VARCHAR NOT NULL,
        PRIMARY KEY (effective_date, code))""")
    for snap in snapshots:
        if snap["codes"]:
            con_w.execute(
                "INSERT INTO pool_snapshots SELECT ?, ?, unnest(?)",
                [snap["effective_date"], snap["cutoff_date"], snap["codes"]])
    con_w.execute("CHECKPOINT")
    n = con_w.execute("SELECT count(*) FROM pool_snapshots").fetchone()[0]
    con_w.close()

    # ---- 写 JSON（人读版）----
    meta = {
        "rule": "半年度快照：生效=6/12月首个交易日；选样=前一月末最后交易日；"
                "主板+通胀调整带(1-20亿@2026, decay 0.55/11y)；"
                f"次新排除=上市不满 {MIN_LISTED_TRADING_DAYS} 交易日",
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_periods": len(snapshots),
        "total_rows": n,
        "periods": [{k: v for k, v in s.items() if k != "codes"} | {"n": len(s["codes"])}
                    for s in snapshots],
        "snapshots": snapshots,
    }
    HISTORY_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    # ---- 验证门 ----
    latest = snapshots[-1]
    pool = json.load(open(ROOT / "pools" / "mainboard_microcap.json"))
    cur = {s["code"] for s in pool["stocks"]}
    last_set = set(latest["codes"])
    log.info("=" * 60)
    log.info("DB pool_snapshots 共 %d 行 / %d 档；JSON -> %s", n, len(snapshots), HISTORY_PATH)
    log.info("最新档 %s: %d 只；现行池 json（历史并集）: %d 只；重合 %d；"
             "最新档不在现行池: %d；现行池不在最新档: %d",
             latest["effective_date"], len(last_set), len(cur),
             len(last_set & cur), len(last_set - cur), len(cur - last_set))
    sizes = [len(s["codes"]) for s in snapshots]
    log.info("各档人数: min=%d max=%d 中位=%d", min(sizes), max(sizes),
             sorted(sizes)[len(sizes) // 2])
    con.close()


if __name__ == "__main__":
    main()
