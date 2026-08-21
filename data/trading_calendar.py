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
    python -m data.trading_calendar   # 全量刷新日历（单次 trade_cal 调用）
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
    """全量刷新日历（幂等）。返回增量拉取边界 = 今天（含）之前最晚的开市日。

    表内可包含未来日期（trade_cal 返回官方预公布的次年安排，供
    DaysToNextTrading/DaysToDelivery 等日历因子取"下一交易日"），但**返回值
    永远 ≤ 今天**——pull 的增量目标、integrity 的"当日应有"都以它为界，
    绝不因表内未来日期而把未来当增量。"""
    ensure_table(con)
    today = datetime.now()
    # 拉到次年年底：国务院每年 11~12 月预公布次年假期，API 已含官方安排；
    # 更远的年份未公布，不拉（防臆测数据）
    end = f"{today.year + 1}1231"
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
    latest = latest(con, on_or_before=today.strftime("%Y-%m-%d"))
    log.info("trading_calendar refreshed: %d rows (%s ~ %s); "
             "increment boundary (latest open <= today): %s",
             len(df), df["cal_date"].min(), df["cal_date"].max(), latest)
    return latest


def refresh_safe(con: duckdb.DuckDBPyConnection, pro) -> str:
    """刷新日历；API 失败时降级用已缓存日历，返回可用的最晚日期。

    降级只影响"发现新交易日"的能力，不影响缺口对账（对账用缓存内日期）。
    """
    try:
        return refresh(con, pro)
    except Exception as e:
        log.warning("trade_cal refresh failed (%s), falling back to cached calendar", e)
        cached = latest(con, on_or_before=datetime.now().strftime("%Y-%m-%d"))
        if cached is None:
            raise RuntimeError("trading_calendar is empty and refresh failed; "
                               "cannot proceed safely")
        return cached


def _table_exists(con) -> bool:
    return con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name='trading_calendar'"
    ).fetchone()[0] > 0


def _fmt(d) -> str:
    return str(d)[:10]


def trading_days(con, start: str = None, end: str = None) -> list[str]:
    """开市日列表（升序，'YYYY-MM-DD'）。表不存在时返回空（只读连接安全）。"""
    if not _table_exists(con):
        return []
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
    """最近的一个开市日（默认为表内最晚）。表不存在时返回 None。"""
    if not _table_exists(con):
        return None
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
    """最近 n 个开市日（升序），用于滚动窗口重拉。表不存在时返回空。"""
    if not _table_exists(con):
        return []
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
