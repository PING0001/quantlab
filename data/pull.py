# -*- coding: utf-8 -*-
"""统一数据拉取入口（全量与增量同一套 fetcher）。

用法：
    python -m data.pull                # 日常：pending 补拉 + 滚动重拉近5交易日 + 新增
                                       #       当日 cyq 拉空则每30分钟重试至 21:00
    python -m data.pull --reconcile    # 深对账：各表全历史 vs 交易日历，缺口入 pending 后补拉（月度）
    python -m data.pull --full         # 全量 == 原 build_db.py（2~4 小时，勿轻易运行）
    python -m data.pull --dry-run      # 只打印各源目标日期，不调 API 不写库

机制：
- 滚动窗口幂等重拉为主（覆盖 99% 瞬时故障），pending 只记例外；
- 行级后验：行数漂移 ±30% / adj_factor NULL 率 >1% -> 该日入 pending；
- trading_calendar 是日期判断唯一真相（trade_cal），不用业务表 MAX 兜底。
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_PATH
from data import trading_calendar as cal
from data import sources
from data._ts import init_pro
from data.lock import file_lock

log = logging.getLogger(__name__)

LOG_PATH = Path(__file__).parent / "pull.log"

ROLLING_TRADING_DAYS = 5    # 滚动重拉窗口（开市日）
ROLLING_CALENDAR_DAYS = 7   # calendar 粒度滚动窗口（日历日）
CYQ_RETRY_UNTIL = (21, 0)   # 当日 cyq 拉空的重试截止（时, 分）；实证 20:52 首次就绪
CYQ_RETRY_INTERVAL_MIN = 30
MAX_ATTEMPTS = 5            # pending 超过转 dead，等人工处置
FULL_START = "2008-01-01"


# ---- pending_pulls ----

def ensure_pending_table(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS pending_pulls (
            source     VARCHAR,
            date       DATE,
            reason     VARCHAR,
            attempts   INTEGER DEFAULT 0,
            status     VARCHAR DEFAULT 'pending',
            updated_at TIMESTAMP,
            PRIMARY KEY (source, date))
    """)


def mark_pending(con, source: str, date_str: str, reason: str):
    con.execute("""
        INSERT INTO pending_pulls (source, date, reason, attempts, status, updated_at)
        VALUES (?, ?::DATE, ?, 1, 'pending', now())
        ON CONFLICT (source, date) DO UPDATE SET
            attempts = attempts + 1, reason = excluded.reason,
            status = 'pending', updated_at = now()
    """, [source, date_str, reason])
    log.warning("pending: %s %s (%s)", source, date_str, reason)


def clear_pending(con, source: str, date_str: str):
    con.execute("DELETE FROM pending_pulls WHERE source=? AND date=?::DATE",
                [source, date_str])


def load_pending(con, source: str) -> list[str]:
    return [str(r[0])[:10] for r in con.execute(
        "SELECT date FROM pending_pulls WHERE source=? AND status='pending' ORDER BY date",
        [source]).fetchall()]


def reap_dead_pending(con):
    rows = con.execute(
        "SELECT source, date FROM pending_pulls "
        "WHERE status='pending' AND attempts >= ?", [MAX_ATTEMPTS]).fetchall()
    for source, d in rows:
        con.execute("UPDATE pending_pulls SET status='dead', updated_at=now() "
                    "WHERE source=? AND date=?", [source, d])
        log.error("pending DEAD after %d attempts: %s %s "
                  "(manual intervention needed)", MAX_ATTEMPTS, source, d)


# ---- 目标日期计算 ----

def _source_latest(con, source: str) -> str | None:
    """源主表的当前最大日期（trading 粒度）。"""
    table = sources.SOURCES[source]["tables"][0]
    row = con.execute(f"SELECT MAX(date) FROM {table}").fetchone()
    return str(row[0])[:10] if row and row[0] else None


def compute_targets(con, latest_open: str, full: bool) -> dict[str, list[str]]:
    """每源目标日期：pending ∪ 滚动窗口 ∪ 新增（full 则全部历史）。"""
    targets = {}
    rolling = cal.recent(con, ROLLING_TRADING_DAYS, on_or_before=latest_open)
    for name, spec in sources.SOURCES.items():
        if spec["grain"] == "snapshot" or spec["grain"] == "event":
            targets[name] = []  # 忽略日期，每次都跑
            continue
        if full:
            targets[name] = cal.trading_days(con, start=None, end=latest_open) \
                if spec["grain"] == "trading" else _calendar_days(FULL_START, latest_open)
            continue
        pending = load_pending(con, name)
        latest_db = _source_latest(con, name)
        if latest_db is None:
            # 防呆：空表时“新增区间”会退化为全部历史（=静默全量，2~4 小时）。
            # 全量必须显式 --full。
            raise RuntimeError(
                f"source '{name}' table is empty on daily-incremental path; "
                f"run `python -m data.pull --full` first")
        if spec["grain"] == "trading":
            new = [d for d in cal.trading_days(con, start=latest_db, end=latest_open)
                   if d > latest_db]
            base = set(rolling)
        else:  # calendar
            next_day = (datetime.strptime(latest_db, "%Y-%m-%d")
                        + timedelta(days=1)).strftime("%Y-%m-%d")
            new = _calendar_days(next_day, latest_open)
            base = set(_calendar_days(
                (datetime.now() - timedelta(days=ROLLING_CALENDAR_DAYS - 1)).strftime("%Y-%m-%d"),
                datetime.now().strftime("%Y-%m-%d")))
        targets[name] = sorted(base | set(new) | set(pending))
    return targets


def _calendar_days(start: str, end: str) -> list[str]:
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    return [(s + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range((e - s).days + 1)]


# ---- reconcile：全历史对账 ----

def reconcile_to_pending(con, latest_open: str) -> int:
    """各 trading 源表内日期 vs 交易日历，缺口入 pending。返回缺口总数。"""
    n_total = 0
    for name, spec in sources.SOURCES.items():
        if spec["grain"] != "trading":
            continue
        table = spec["tables"][0]
        tdates = {str(r[0])[:10] for r in con.execute(
            f"SELECT DISTINCT date FROM {table}").fetchall()}
        if not tdates:
            continue
        lo, hi = min(tdates), max(tdates)
        expected = set(cal.trading_days(con, start=lo, end=hi))
        for d in sorted(expected - tdates):
            mark_pending(con, name, d, f"calendar gap found by reconcile")
            n_total += 1
    log.info("reconcile: %d gap dates -> pending", n_total)
    return n_total


# ---- 拉取执行 ----

def _pull_one_day_with_checks(con, pro, source: str, date_str: str) -> bool:
    """拉取单日（trading 粒度）+ 行级后验。返回是否成功。"""
    n = sources.SOURCES[source]["pull"](con, pro, [date_str])
    if n == 0:
        mark_pending(con, source, date_str, "empty response")
        return False
    for table in sources.SOURCES[source]["tables"]:
        reason = sources.postcheck(con, source, table, date_str)
        if reason:
            mark_pending(con, source, date_str, f"postcheck[{table}]: {reason}")
            return False
    clear_pending(con, source, date_str)
    return True


def _pull_cyq(con, pro, dates: list[str], latest_open: str):
    """cyq 专属：历史日期常规拉；当日拉空进入延迟重试（至 21:00）。"""
    today_str = datetime.now().strftime("%Y-%m-%d")
    history = [d for d in dates if d != today_str]
    for d in history:
        _pull_one_day_with_checks(con, pro, "cyq", d)

    if today_str in dates and today_str == latest_open:
        while True:
            if _pull_one_day_with_checks(con, pro, "cyq", today_str):
                return
            now = datetime.now()
            deadline = now.replace(hour=CYQ_RETRY_UNTIL[0], minute=CYQ_RETRY_UNTIL[1],
                                   second=0, microsecond=0)
            if now >= deadline:
                log.error("cyq %s: still failing at deadline %s, left pending",
                          today_str, deadline.strftime("%H:%M"))
                return
            wait_s = min(CYQ_RETRY_INTERVAL_MIN * 60, (deadline - now).total_seconds())
            log.info("cyq %s empty (source updates ~18-19h, relay lag until ~20:52), "
                     "retrying in %.0f min", today_str, wait_s / 60)
            time.sleep(wait_s)


def _trigger_industry(con):
    """池内股票在 industry 表缺失覆盖时触发增量（复用 build_industry.py，子进程隔离）。

    以 industry 缺失为准而非 stock_info 新 code：新股通常半年后才进池定义，
    彼时 stock_info 已收录，按 new_codes 触发会漏。

    DuckDB 文件锁跨进程互斥：父进程必须先释放连接子进程才能写库，跑完重连。
    返回（可能重连过的）连接，调用方需接住返回值。
    """
    pool_codes = sorted(sources._pool_union_codes())
    if not pool_codes:
        return con
    ph = ",".join(["?"] * len(pool_codes))
    missing = con.execute(
        f"""SELECT count(*) FROM stock_info
            WHERE code IN ({ph}) AND code NOT IN (SELECT code FROM industry)""",
        pool_codes,
    ).fetchone()[0]
    if not missing:
        return con
    log.info("%d pool stocks missing industry coverage -> build_industry "
             "(parent releases DB connection for the subprocess)", missing)
    con.close()
    try:
        # 用 -m 模块方式运行（cwd=项目根）：脚本直跑会把 data/ 顶到 sys.path[0]，
        # data/ 下的模块名可能遮蔽 stdlib（calendar 事故的教训）
        subprocess.run(
            [sys.executable, "-m", "data.build_industry"],
            cwd=Path(__file__).resolve().parent.parent,
            check=True, timeout=1800)
    except Exception as e:
        log.error("industry refresh failed (run `python -m data.build_industry` "
                  "manually): %s", e)
    finally:
        con = duckdb.connect(str(DB_PATH))
        con.execute("SET memory_limit='2GB'")
        con.execute("SET threads=4")
    return con


def run(full: bool = False, reconcile: bool = False, dry_run: bool = False) -> int:
    with file_lock(DB_PATH.parent / ".pull.lock", "data-pull"):
        con = duckdb.connect(str(DB_PATH))
        con.execute("SET memory_limit='2GB'")
        con.execute("SET threads=4")
        try:
            sources.ensure_tables(con)
            ensure_pending_table(con)
            pro = init_pro()

            latest_open = cal.refresh_safe(con, pro)
            log.info("latest open day: %s", latest_open)

            if reconcile:
                reconcile_to_pending(con, latest_open)
            reap_dead_pending(con)

            targets = compute_targets(con, latest_open, full)

            if dry_run:
                log.info("[dry-run] targets per source:")
                for name, dates in targets.items():
                    spec = sources.SOURCES[name]
                    if spec["grain"] in ("snapshot", "event"):
                        log.info("  %-11s (%s): always runs", name, spec["grain"])
                    else:
                        preview = f"{dates[0]} ~ {dates[-1]}" if dates else "none"
                        log.info("  %-11s (%s): %d days [%s]",
                                 name, spec["grain"], len(dates), preview)
                return 0

            # 1. stock_info 快照（先跑：识别新股票）
            n_info, new_codes = sources.pull_stock_info(con, pro, [])
            log.info("stock_info: %d rows, %d new codes", n_info, len(new_codes))

            # 2. trading 粒度源：daily 逐日后验；cyq 专属重试
            for d in targets["daily"]:
                _pull_one_day_with_checks(con, pro, "daily", d)
            _pull_cyq(con, pro, targets["cyq"], latest_open)

            n_idx = sources.pull_index(con, pro, targets["index"])
            log.info("index: %d rows", n_idx)
            n_shibor = sources.pull_shibor(con, pro, targets["shibor"])
            log.info("shibor: %d rows", n_shibor)
            n_nc = sources.pull_namechange(con, pro, [])
            log.info("namechange: %d rows merged", n_nc)

            con.execute("CHECKPOINT")

            # 3. 池内行业覆盖缺口 -> 触发增量
            con = _trigger_industry(con)

            # 4. 收尾对账报告（软警告记录，不阻断）
            from factors import integrity
            report = integrity.check(con)
            log.info("Pull complete. hard_fail=%s, soft_warnings=%d",
                     report["hard_fail"], len(report["soft_warnings"]))
            return 0
        finally:
            con.close()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )
    ap = argparse.ArgumentParser(description="Unified data pull (incremental & full)")
    ap.add_argument("--full", action="store_true",
                    help="Full rebuild from 2008 (equivalent to legacy build_db.py; 2-4 hours!)")
    ap.add_argument("--reconcile", action="store_true",
                    help="Deep reconcile: full-history calendar gaps -> pending, then pull")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print target dates per source without API calls")
    args = ap.parse_args()
    if args.full:
        log.warning("--full performs ~13800 API calls over 2-4 hours. Continuing.")
    sys.exit(run(full=args.full, reconcile=args.reconcile, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
