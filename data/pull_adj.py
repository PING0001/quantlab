# -*- coding: utf-8 -*-
"""
A股日线数据库 - 增量更新

从 relay 拉取缺失交易日的最新数据，合并到 daily_raw。
VIEW daily_kline 自动反映更新，无需额外操作。
"""
from __future__ import annotations
import json, logging, sys, time
from pathlib import Path
from datetime import datetime, timedelta

import duckdb
import pandas as pd
import tushare as ts
import tushare.pro.client as client
client.DataApi._DataApi__http_url = "http://api.quicksync.cn"
from dotenv import load_dotenv

DB_PATH = Path(__file__).parent / "ashare.duckdb"
STOCK_POOL = Path(__file__).parent.parent / "mainboard_microcap.json"
LOG_PATH = Path(__file__).parent / "pull_adj.log"
BATCH_SIZE = 2
REQ_INTERVAL = 0.35

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def _init_pro():
    global _PRO_API
    if _PRO_API is not None:
        return _PRO_API
    env_path = Path(__file__).parent.parent / ".env"
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


def load_stock_pool(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["stocks"]


def batch_ts_codes(stocks, size=2):
    for i in range(0, len(stocks), size):
        batch = stocks[i:i+size]
        ts_codes = ",".join(s["full_code"] for s in batch)
        codes = [s["code"] for s in batch]
        yield ts_codes, codes


def fetch_daily(pro, ts_codes, start, end):
    df = pro.daily(ts_code=ts_codes, start_date=start, end_date=end)
    if df is None or df.empty:
        return pd.DataFrame()
    df["code"] = df["ts_code"].str.replace(r"[.](SH|SZ|BJ)$", "", regex=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df.rename(columns={"trade_date": "date", "vol": "volume"}, inplace=True)
    for col in ["open", "high", "low", "close", "volume", "amount", "pct_chg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["code", "date", "open", "high", "low", "close", "volume", "amount", "pct_chg"]]


def fetch_adj_factor(pro, ts_codes, start, end):
    df = pro.adj_factor(ts_code=ts_codes, start_date=start, end_date=end)
    if df is None or df.empty:
        return pd.DataFrame()
    df["code"] = df["ts_code"].str.replace(r"[.](SH|SZ|BJ)$", "", regex=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce")
    return df[["code", "trade_date", "adj_factor"]].rename(columns={"trade_date": "date"})


def ensure_view(con):
    try:
        con.execute("SELECT count(*) FROM daily_kline LIMIT 1")
    except Exception:
        log.info("daily_kline VIEW 不存在，创建中 ...")
        con.execute("DROP VIEW IF EXISTS daily_kline")
        con.execute("""
            CREATE OR REPLACE VIEW daily_kline AS
            WITH latest_adj AS (
                SELECT code, MAX_BY(adj_factor, date) AS latest_adj
                FROM daily_raw
                WHERE adj_factor IS NOT NULL AND adj_factor > 0
                GROUP BY code
            )
            SELECT
                r.code, r.date,
                r.open   * (l.latest_adj / NULLIF(r.adj_factor, 0)) AS open,
                r.high   * (l.latest_adj / NULLIF(r.adj_factor, 0)) AS high,
                r.low    * (l.latest_adj / NULLIF(r.adj_factor, 0)) AS low,
                r.close  * (l.latest_adj / NULLIF(r.adj_factor, 0)) AS close,
                r.volume, r.amount, r.pct_chg, r.turn
            FROM daily_raw r
            LEFT JOIN latest_adj l ON r.code = l.code
        """)
        log.info("daily_kline VIEW 已创建")


def ensure_data_gaps_table(con):
    con.execute("""CREATE TABLE IF NOT EXISTS data_gaps (
        code VARCHAR NOT NULL,
        date DATE NOT NULL,
        reason VARCHAR DEFAULT 'suspension',
        PRIMARY KEY (code, date)
    )""")


def main():
    con = duckdb.connect(str(DB_PATH))
    ensure_data_gaps_table(con)
    stocks = load_stock_pool(STOCK_POOL)
    pro = _init_pro()

    today = datetime.now().date()
    all_daily, all_adj = [], []
    total_batches = (len(stocks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx, (ts_codes, codes) in enumerate(batch_ts_codes(stocks), 1):
        log.info("[批 %d/%d] %s ~ %s (%d只)",
                 batch_idx, total_batches, codes[0], codes[-1], len(codes))

        # 按股查最大日期，跳过已记录的缺口
        batch_start = None
        for code in codes:
            max_d = con.execute(
                "SELECT MAX(date) FROM daily_raw WHERE code=?", (code,)
            ).fetchone()[0]
            if max_d is None:
                s = datetime(2015, 1, 1)
            else:
                s = max_d + timedelta(days=1)
            g = con.execute(
                "SELECT MAX(date) FROM data_gaps WHERE code=? AND date>=?", (code, s)
            ).fetchone()[0]
            if g is not None:
                s = g + timedelta(days=1)
            if s <= today:
                if batch_start is None or s < batch_start:
                    batch_start = s

        if batch_start is None:
            log.info("  %s 都已完整或标记为缺口，跳过", codes)
            continue

        start_str = batch_start.strftime("%Y%m%d")
        end_str = today.strftime("%Y%m%d")
        log.info("  拉取: %s ~ %s", start_str, end_str)

        df_d = fetch_daily(pro, ts_codes, start_str, end_str)
        if not df_d.empty:
            all_daily.append(df_d)
            log.info("  daily: %d 行", len(df_d))
        time.sleep(REQ_INTERVAL)

        df_a = fetch_adj_factor(pro, ts_codes, start_str, end_str)
        if not df_a.empty:
            all_adj.append(df_a)
            log.info("  adj_factor: %d 行", len(df_a))
        time.sleep(REQ_INTERVAL)

        # 拉完后检查哪些日期还缺失 → 记入 data_gaps
        for code in codes:
            max_after = con.execute(
                "SELECT MAX(date) FROM daily_raw WHERE code=?", (code,)
            ).fetchone()[0]
            if max_after is None:
                continue
            first_miss = max_after + timedelta(days=1)
            if first_miss <= today:
                dates = [(code, d, "suspension") for d in pd.date_range(first_miss, today, freq="D")]
                if dates:
                    con.executemany("INSERT OR IGNORE INTO data_gaps VALUES (?, ?, ?)", dates)
                    log.info("  %s: 标记 %d 个缺口（%s ~ %s）", code, len(dates), first_miss, today)

    if not all_daily:
        log.info("无新数据")
        con.close()
        return

    daily = pd.concat(all_daily, ignore_index=True)
    adj = pd.concat(all_adj, ignore_index=True) if all_adj else pd.DataFrame()
    if not adj.empty:
        daily = daily.merge(adj, on=["code", "date"], how="left")
    else:
        daily["adj_factor"] = None
    if "turn" not in daily.columns:
        daily["turn"] = None

    log.info("合并写入 daily_raw（%d 行）...", len(daily))
    con.execute("""
        INSERT OR REPLACE INTO daily_raw
            (code, date, open, high, low, close, volume, amount, pct_chg, turn, adj_factor)
        SELECT code, date, open, high, low, close,
               volume, amount, pct_chg, turn, adj_factor
        FROM daily
    """)
    con.execute("CHECKPOINT")
    row_count = con.execute("SELECT count(*) FROM daily_raw").fetchone()[0]
    log.info("完成：daily_raw 当前 %d 行", row_count)
    con.close()


if __name__ == "__main__":
    con = duckdb.connect(str(DB_PATH))
    ensure_view(con)
    con.close()
    main()
