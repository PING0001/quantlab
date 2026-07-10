# -*- coding: utf-8 -*-
"""
A股筹码分布数据 - 全量构建 + 增量更新

数据来源：Tushare cyq_perf 接口（通过 api.quicksync.cn 中转）
数据起始：2018-01-01
更新频率：每天18~19点更新

用法：
    python data/build_cyq.py           # 全量拉取（2018-至今）
    python data/build_cyq.py --incr    # 增量拉取（最近缺失交易日）
"""
from __future__ import annotations
import logging, sys, time
from pathlib import Path
from datetime import datetime, timedelta

import duckdb
import pandas as pd
import tushare as ts
import tushare.pro.client as client
client.DataApi._DataApi__http_url = "http://api.quicksync.cn"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from config import DB_PATH, get_pool_codes

LOG_PATH = Path(__file__).parent / "build_cyq.log"

CYQ_START = "20180101"
REQ_INTERVAL = 0.0
PROGRESS_EVERY = 200

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

CYQ_FIELDS = (
    "ts_code,trade_date,his_low,his_high,"
    "cost_5pct,cost_15pct,cost_50pct,cost_85pct,cost_95pct,"
    "weight_avg,winner_rate"
)
CYQ_NUMERIC = [
    "his_low", "his_high",
    "cost_5pct", "cost_15pct", "cost_50pct", "cost_85pct", "cost_95pct",
    "weight_avg", "winner_rate",
]
CYQ_ALL_COLS = ["code", "date"] + CYQ_NUMERIC


def _init_pro():
    global _PRO_API
    if _PRO_API is not None:
        return _PRO_API
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path, encoding="utf-8-sig")
    import os
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN not found in .env")
    ts.set_token(token)
    pro = ts.pro_api()
    pro._DataApi__http_timeout = 120
    _PRO_API = pro
    return pro
_PRO_API = None


def _retry_api(fn, *args, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 * (2 ** attempt)
            log.warning("Retry %d/%d after error: %s, waiting %ds...",
                        attempt + 1, max_retries, e, wait)
            time.sleep(wait)


def _code_to_ts(code):
    """Convert 6-digit code to ts_code format."""
    if not isinstance(code, str) or len(code) != 6:
        return code
    if code.startswith(('6', '9')):
        return f"{code}.SH"
    elif code.startswith(('8', '4')):
        return f"{code}.BJ"
    else:
        return f"{code}.SZ"


def _ensure_table(con):
    exists = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name='cyq_perf'"
    ).fetchone()[0]
    if exists:
        has_pk = con.execute("""
            SELECT count(*) FROM information_schema.table_constraints
            WHERE table_name='cyq_perf' AND constraint_type='PRIMARY KEY'
        """).fetchone()[0]
        if not has_pk:
            con.execute("DROP TABLE cyq_perf")
            exists = False
    if not exists:
        con.execute("""
            CREATE TABLE cyq_perf (
                code        VARCHAR NOT NULL,
                date        DATE    NOT NULL,
                his_low     DOUBLE,
                his_high    DOUBLE,
                cost_5pct   DOUBLE,
                cost_15pct  DOUBLE,
                cost_50pct  DOUBLE,
                cost_85pct  DOUBLE,
                cost_95pct  DOUBLE,
                weight_avg  DOUBLE,
                winner_rate DOUBLE,
                PRIMARY KEY (code, date)
            )
        """)


def _get_latest_cal_date(pro):
    """Get the latest trading day from Tushare trade_cal."""
    today_str = datetime.now().strftime("%Y%m%d")
    try:
        df = _retry_api(pro.trade_cal, exchange='SSE',
                        start_date="20180101", end_date=today_str,
                        fields='cal_date,is_open')
        if df is not None and not df.empty:
            open_days = df[df['is_open'] == 1]['cal_date'].tolist()
            if open_days:
                return max(open_days)
    except Exception as e:
        log.warning("trade_cal failed: %s", e)
    return today_str


def pull_cyq_for_stock(pro, ts_code, start_date, end_date):
    """Pull cyq_perf for one stock over a date range."""
    df = _retry_api(pro.cyq_perf, ts_code=ts_code,
                    start_date=start_date, end_date=end_date,
                    fields=CYQ_FIELDS)
    if df is None or df.empty:
        return pd.DataFrame()
    df["code"] = df["ts_code"].str[:6]
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df.rename(columns={"trade_date": "date"}, inplace=True)
    for col in CYQ_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CYQ_ALL_COLS:
        if col not in df.columns:
            df[col] = None
    return df[CYQ_ALL_COLS]


def build_full(con, pro):
    """Full backfill: pull cyq_perf for all POOL stocks from 2018 to now."""
    codes = get_pool_codes()
    end_date = datetime.now().strftime("%Y%m%d")

    n_stocks = len(codes)
    log.info("Full cyq_perf build: %d pool stocks, %s ~ %s",
             n_stocks, CYQ_START, end_date)

    total_rows = 0
    t_start = time.time()
    for i, code in enumerate(codes, 1):
        ts_code = _code_to_ts(code)

        try:
            df = pull_cyq_for_stock(pro, ts_code, CYQ_START, end_date)
        except Exception as e:
            log.warning("Failed to pull cyq_perf for %s: %s", code, e)
            continue

        if df.empty:
            continue

        n_rows = len(df)
        con.execute("DELETE FROM cyq_perf WHERE code = ?", (code,))
        con.execute("INSERT INTO cyq_perf SELECT * FROM df")
        total_rows += n_rows

        if i % PROGRESS_EVERY == 0:
            elapsed = time.time() - t_start
            rate = i / elapsed
            eta = (n_stocks - i) / rate
            log.info("  [%d/%d] %.1f stock/s, ETA %.0fs, rows so far: %d",
                     i, n_stocks, rate, eta, total_rows)

    log.info("Full build complete: %d rows in %.0fs", total_rows,
             time.time() - t_start)


def build_incremental(con, pro):
    """Incremental: pull new rows for stocks missing the latest trading day."""
    codes = get_pool_codes()
    latest_cal = _get_latest_cal_date(pro)

    # Check what's the max date per stock already in DB
    existing = con.execute("""
        SELECT code, MAX(date) FROM cyq_perf GROUP BY code
    """).fetchall()
    existing_map = {r[0]: r[1] for r in existing}

    # Find stocks that need updating
    need_update = []
    for code in codes:
        last_date = existing_map.get(code)
        if last_date is None:
            need_update.append((code, CYQ_START))
        else:
            last_dt = pd.Timestamp(last_date)
            if last_dt.strftime("%Y%m%d") < latest_cal:
                start = (last_dt + timedelta(days=1)).strftime("%Y%m%d")
                need_update.append((code, start))

    if not need_update:
        log.info("cyq_perf already up to date (%d stocks).", len(codes))
        return

    n = len(need_update)
    end_date = datetime.now().strftime("%Y%m%d")
    log.info("Incremental cyq_perf: %d stocks need update", n)

    total_rows = 0
    t_start = time.time()
    for i, (code, start) in enumerate(need_update, 1):
        ts_code = _code_to_ts(code)

        try:
            df = pull_cyq_for_stock(pro, ts_code, start, end_date)
        except Exception as e:
            log.warning("Failed to pull cyq_perf for %s: %s", code, e)
            continue

        if df.empty:
            continue

        n_rows = len(df)
        con.execute("DELETE FROM cyq_perf WHERE code = ? AND date >= ?",
                    (code, pd.to_datetime(start).date()))
        con.execute("INSERT INTO cyq_perf SELECT * FROM df")
        total_rows += n_rows

        if i % PROGRESS_EVERY == 0:
            elapsed = time.time() - t_start
            rate = i / elapsed
            eta = (n - i) / rate
            log.info("  [%d/%d] %.1f stock/s, ETA %.0fs, new rows: %d",
                     i, n, rate, eta, total_rows)

    log.info("Incremental complete: %d new rows in %.0fs", total_rows,
             time.time() - t_start)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build/update cyq_perf chip data")
    ap.add_argument("--incr", action="store_true",
                    help="Incremental update (only missing dates)")
    args = ap.parse_args()

    con = duckdb.connect(str(DB_PATH))
    con.execute("SET memory_limit = '2GB'")
    con.execute("SET threads = 4")

    _ensure_table(con)
    pro = _init_pro()

    if args.incr:
        build_incremental(con, pro)
    else:
        build_full(con, pro)

    row_count = con.execute("SELECT count(*) FROM cyq_perf").fetchone()[0]
    stock_count = con.execute("SELECT count(DISTINCT code) FROM cyq_perf").fetchone()[0]
    log.info("cyq_perf table: %d rows, %d stocks", row_count, stock_count)

    con.close()


if __name__ == "__main__":
    main()
