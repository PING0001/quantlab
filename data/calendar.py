# -*- coding: utf-8 -*-
"""
交易日历 trading_calendar：trade_cal API 是唯一真相源。

所有增量拉取的日期判断基于本表，而非业务表的 MAX(date)——后者无法发现
中间交易日的永久空洞。刷新失败时降级使用已缓存日历（只覆盖到缓存上限，
安全），绝不用业务表（如 daily_raw）兜底：业务表本身是对账对象，拿它当
日历会让缺口永不可见。

日期统一以 'YYYY-MM-DD' 字符串对外（与 factor_values 的 VARCHAR date
约定一致，字典序即时间序）。

用法：
    python -m data.calendar     # 全量刷新日历（单次 trade_cal 调用）
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_PATH
from data._ts import init_pro, retry_api

log = logging.getLogger(__name__)

CAL_START = "20080101"

_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trading_calendar (
    date    DATE PRIMARY KEY,
    is_open BOOLEAN
)
"""


def ensure_table(con: duckdb.DuckDBPyConnection):
    con.execute(_TABLE_SQL)


def refresh(con: duckdb.DuckDBPyConnection, pro, start: str = CAL_START) -> str:
    """全量刷新日历（幂等）。返回刷新到的最晚日期（YYYY-MM-DD）。"""
    ensure_table(con)
    end = datetime.now().strftime("%Y%m%d")
    df = retry_api(pro.trade_cal, exchange="SSE",
                   start_date=start, end_date=end,
                   fields="cal_date,is_open")
    if df is None or df.empty:
        raise RuntimeError("trade_cal returned empty")
    con.execute(
        "INSERT OR REPLACE INTO trading_calendar "
        "SELECT strptime(cal_date, '%Y%m%d')::DATE, is_open FROM df"
    )
    con.execute("CHECKPOINT")
    latest = str(df["cal_date"].max())
    log.info("trading_calendar refreshed: %d rows (%s ~ %s)",
             len(df), df["cal_date"].min(), latest)
    return f"{latest[:4]}-{latest[4:6]}-{latest[6:]}"


def refresh_safe(con: duckdb.DuckDBPyConnection, pro) -> str:
    """刷新日历；API 失败时降级用已缓存日历，返回可用的最晚日期。

    降级只影响"发现新交易日"的能力，不影响缺口对账（对账用缓存内日期）。
    """
    try:
        return refresh(con, pro)
    except Exception as e:
        log.warning("trade_cal refresh failed (%s), falling back to cached calendar", e)
        cached = latest(con)
        if cached is None:
            raise RuntimeError("trading_calendar is empty and refresh failed; "
                               "cannot proceed safely")
        return cached


def _fmt(d) -> str:
    return str(d)[:10]


def trading_days(con, start: str = None, end: str = None) -> list[str]:
    """开市日列表（升序，'YYYY-MM-DD'）。"""
    ensure_table(con)
    sql = "SELECT date FROM trading_calendar WHERE is_open"
    params = []
    if start:
        sql += " AND date >= ?::DATE"
        params.append(start)
    if end:
        sql += " AND date <= ?::DATE"
        params.append(end)
    sql += " ORDER BY date"
    return [_fmt(r[0]) for r in con.execute(sql, params).fetchall()]


def latest(con, on_or_before: str = None) -> str | None:
    """最近的一个开市日（默认为表内最晚）。"""
    ensure_table(con)
    if on_or_before:
        row = con.execute(
            "SELECT MAX(date) FROM trading_calendar WHERE is_open AND date <= ?::DATE",
            [on_or_before],
        ).fetchone()
    else:
        row = con.execute(
            "SELECT MAX(date) FROM trading_calendar WHERE is_open"
        ).fetchone()
    return _fmt(row[0]) if row and row[0] else None


def recent(con, n: int, on_or_before: str = None) -> list[str]:
    """最近 n 个开市日（升序），用于滚动窗口重拉。"""
    ensure_table(con)
    if on_or_before:
        rows = con.execute(
            "SELECT date FROM trading_calendar WHERE is_open AND date <= ?::DATE "
            "ORDER BY date DESC LIMIT ?",
            [on_or_before, n],
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT date FROM trading_calendar WHERE is_open "
            "ORDER BY date DESC LIMIT ?", [n],
        ).fetchall()
    return sorted(_fmt(r[0]) for r in rows)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    con = duckdb.connect(str(DB_PATH))
    try:
        pro = init_pro()
        latest_dt = refresh(con, pro)
        n_open = con.execute(
            "SELECT count(*) FROM trading_calendar WHERE is_open"
        ).fetchone()[0]
        log.info("Calendar OK: %d open days, latest %s", n_open, latest_dt)
    finally:
        con.close()


if __name__ == "__main__":
    main()
