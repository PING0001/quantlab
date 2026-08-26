# -*- coding: utf-8 -*-
"""池成员资格（时点）单源模块：查询 API + 快照构建器（2026-08-24 池时点化，
2026-08-25 吸收原 pools/build_pool_history.py）。

数据源：pool_snapshots(effective_date, cutoff_date, code)--半年度快照。
查询语义：日期 d 的池 = 满足 eff <= d < next_eff 的档的成员集；d 早于首档
-> 空集。消费端（训练/筛选/泄漏断言/回测基准/LIVE 报告/日更/数据拉取范围）
一律经本模块取成员，不再读 pools/*.json 的历史并集（并集宇宙已废，
2026-08-25 起物理删除）。

快照构建（沪深300式，2026-08-24 用户四项裁定 + 次新参数）：
  生效日   = 每年 6 / 12 月的首个交易日（简化版，不抄第二个周五）
  选样截止 = 生效日前一个月的最后一个交易日（模拟指数"提前定样"，
             生效日当天不依赖当天数据）
  成员规则 = 主板（00/60 前缀兜底）+ 流通市值通胀调整带（1-40 亿 @2026，
             DECAY_RATIO 年衰减）+ 截止日当日有 daily_basic 行（幸存者感知：
             退市股自然缺席）
  次新排除 = 选样截止日时上市不满 252 个交易日（按交易日历计）

Usage（重建快照，幂等；写 DB 后自动清进程缓存）：
    python -m pools.membership              # 全序列重建（默认 2015 起）
    python -m pools.membership --from 2024  # 只重建指定年份起的档
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from config import DB_PATH

log = logging.getLogger(__name__)

_IVS = None   # 进程级缓存：[(eff, next_eff|None, frozenset)]

# 人读版 JSON（构建器副产物，regenerable，gitignore；勿复用已删的旧池历史并集文件名）
HISTORY_PATH = DB_PATH.parent.parent / "pools" / "mainboard_microcap_snapshots.json"

# ---- 带规则（原 pools/build_microcap.py 同源复制，勿漂移）----
# 2026-08-24 用户裁定：带宽放宽至 1~40 亿（流通市值；原 1~20 亿时点池仅
# 150-400 只/档，用户认为过窄）
BASE_YEAR = 2026
BASE_LOW_YI = 1.0
BASE_HIGH_YI = 40.0
TOTAL_YEARS = 11.0
DECAY_RATIO = 11.0 / 20.0
MIN_LISTED_TRADING_DAYS = 252   # 次新股门槛（用户裁定 2026-08-24）


# ============================================================================
# 查询 API
# ============================================================================

def _load(con: duckdb.DuckDBPyConnection | None = None):
    global _IVS
    if _IVS is not None:
        return _IVS
    own = con is None
    c = con or duckdb.connect(str(DB_PATH), read_only=True)
    try:
        rows = c.execute(
            "SELECT effective_date, list(code) FROM pool_snapshots "
            "GROUP BY effective_date ORDER BY effective_date").fetchall()
    finally:
        if own:
            c.close()
    effs = [r[0] for r in rows]
    ivs = [(r[0], effs[i + 1] if i + 1 < len(rows) else None, frozenset(r[1]))
           for i, r in enumerate(rows)]
    _IVS = ivs
    return ivs


def refresh():
    """快照重建后清缓存。"""
    global _IVS
    _IVS = None


def _norm_date(d) -> str:
    if isinstance(d, str):
        return d[:10]
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def member_mask(dates, codes, con=None) -> np.ndarray:
    """逐行判定 (date, code) 是否当期池成员。向量化（merge），无 python 行循环。"""
    ivs = _load(con)
    effs = np.array([e for e, _, _ in ivs])
    d_str = pd.Series(dates).map(_norm_date).to_numpy()
    iv_id = np.searchsorted(effs, d_str, side="right") - 1
    valid = iv_id >= 0
    members = pd.DataFrame(
        {"iv": np.repeat(np.arange(len(ivs)), [len(s) for _, _, s in ivs]),
         "code": np.concatenate([sorted(s) for _, _, s in ivs]) if ivs else []})
    rows = pd.DataFrame({"iv": np.where(valid, iv_id, -1), "code": list(codes)})
    if members.empty:
        return np.zeros(len(rows), dtype=bool)
    hit = rows.merge(members.drop_duplicates(), on=["iv", "code"], how="left",
                     indicator=True)
    return (hit["_merge"] == "both").to_numpy()


def union_codes(since: str | None = None, con=None) -> list[str]:
    """since 起各档成员并集（数据加载用：标签 K 线、因子行、拉取范围）。"""
    ivs = _load(con)
    s = set()
    for eff, _, codes in ivs:
        if since is None or eff >= since:
            s |= codes
    return sorted(s)


def latest_codes(con=None) -> frozenset:
    return _load(con)[-1][2]


def codes_on(date, con=None) -> frozenset:
    d = _norm_date(date)
    for eff, nxt, codes in reversed(_load(con)):
        if d >= eff and (nxt is None or d < nxt):
            return codes
    return frozenset()


def reset_points(test_start, test_end, con=None) -> list[tuple[str, frozenset]]:
    """基准半年重置点（用户裁定 A）：测试首日 + 窗口内各生效日 ->
    [(date, members)] 升序。"""
    ts, te = _norm_date(test_start), _norm_date(test_end)
    pts = [(ts, codes_on(ts, con))]
    for eff, _, codes in _load(con):
        if ts < eff <= te:
            pts.append((eff, codes))
    return pts


# ============================================================================
# 快照构建器
# ============================================================================

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
    out = []
    now = datetime.now()
    for y in range(from_year, now.year + 1):
        for m in (6, 12):
            # 生效日 = 该月首个交易日
            eff = next((d for d in opens if d.startswith(f"{y:04d}-{m:02d}")), None)
            if eff is None or eff > now.strftime("%Y-%m-%d"):
                continue
            # 截止日 = 生效月前一月的最后一个交易日（6->5, 12->11）
            pm = m - 1
            py = y
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


def rebuild_snapshots(from_year: int = 2015) -> None:
    """全序列重建 pool_snapshots（幂等，DROP+CREATE）+ 人读版 JSON。"""
    con = duckdb.connect(str(DB_PATH))

    periods = _periods(con, from_year)
    if not periods:
        raise SystemExit("无可构建档期（检查 trading_calendar / daily_basic）")
    log.info("档期 %d 个: %s ... %s", len(periods), periods[0][0], periods[-1][0])

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

    con.execute("DROP TABLE IF EXISTS pool_snapshots")
    con.execute("""CREATE TABLE pool_snapshots (
        effective_date VARCHAR NOT NULL,
        cutoff_date     VARCHAR NOT NULL,
        code            VARCHAR NOT NULL,
        PRIMARY KEY (effective_date, code))""")
    for snap in snapshots:
        if snap["codes"]:
            con.execute(
                "INSERT INTO pool_snapshots SELECT ?, ?, unnest(?)",
                [snap["effective_date"], snap["cutoff_date"], snap["codes"]])
    con.execute("CHECKPOINT")
    n = con.execute("SELECT count(*) FROM pool_snapshots").fetchone()[0]
    con.close()
    refresh()

    # ---- 写 JSON（人读版）----
    meta = {
        "rule": "半年度快照：生效=6/12月首个交易日；选样=前一月末最后交易日；"
                "主板+通胀调整带(1-40亿@2026, decay 0.55/11y)；"
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

    sizes = [len(s["codes"]) for s in snapshots]
    log.info("=" * 60)
    log.info("DB pool_snapshots 共 %d 行 / %d 档；JSON -> %s", n, len(snapshots), HISTORY_PATH)
    log.info("最新档 %s: %d 只；各档人数: min=%d max=%d 中位=%d",
             snapshots[-1]["effective_date"], len(snapshots[-1]["codes"]),
             min(sizes), max(sizes), sorted(sizes)[len(sizes) // 2])


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="重建池时点快照 pool_snapshots")
    ap.add_argument("--from", dest="from_year", type=int, default=2015,
                    help="起始年份（默认 2015，即首个档 2015-06）")
    args = ap.parse_args()
    rebuild_snapshots(from_year=args.from_year)


if __name__ == "__main__":
    main()
