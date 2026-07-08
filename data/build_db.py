# -*- coding: utf-8 -*-
"""
A股日线数据库 - 全量构建

数据来源：Tushare（通过 api.quicksync.cn 中转）
设计约束：
  - 不复权原始日线 + adj_factor 存同一宽表 daily_raw
  - DuckDB VIEW daily_kline 实时算前复权，下游兼容
  - 批量请求（ts_code 逗号分隔），不遍历个股
  - 每批 2 只股票（relay 单趟 6000 行上限）
  - 遵守 relay 速率限制（200次/分钟，约 0.35s 间隔）
"""
from __future__ import annotations
import logging, sys, time
from pathlib import Path

import duckdb
import pandas as pd
import tushare as ts
import tushare.pro.client as client
client.DataApi._DataApi__http_url = "http://api.quicksync.cn"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from config import DB_PATH, load_all_pool_stocks

LOG_PATH = Path(__file__).parent / "build_db.log"
START_DATE = "20080101"
END_DATE = "22220101"
BATCH_SIZE = 1
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


def batch_ts_codes(stocks, size=2):
    for i in range(0, len(stocks), size):
        batch = stocks[i:i+size]
        ts_codes = ",".join(s["full_code"] for s in batch)
        codes = [s["code"] for s in batch]
        yield ts_codes, codes


def _retry_api(fn, *args, max_retries=3, **kwargs):
    """Retry on connection errors with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 * (2 ** attempt)
            log.warning("API error (attempt %d/%d): %s, retrying in %ds...",
                        attempt + 1, max_retries, e, wait)
            time.sleep(wait)


def fetch_daily(pro, ts_codes):
    df = _retry_api(pro.daily, ts_code=ts_codes, start_date=START_DATE, end_date=END_DATE)
    if df is None or df.empty:
        return pd.DataFrame()
    df["code"] = df["ts_code"].str.replace(r"[.](SH|SZ|BJ)$", "", regex=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df.rename(columns={"trade_date": "date", "vol": "volume"}, inplace=True)
    for col in ["open", "high", "low", "close", "volume", "amount", "pct_chg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["code", "date", "open", "high", "low", "close", "volume", "amount", "pct_chg"]]


def fetch_adj_factor(pro, ts_codes):
    df = _retry_api(pro.adj_factor, ts_code=ts_codes)
    if df is None or df.empty:
        return pd.DataFrame()
    df["code"] = df["ts_code"].str.replace(r"[.](SH|SZ|BJ)$", "", regex=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce")
    return df[["code", "trade_date", "adj_factor"]].rename(columns={"trade_date": "date"})


def create_tables(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS stock_info (
            code        VARCHAR PRIMARY KEY,
            name        VARCHAR,
            market      VARCHAR,
            full_code   VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS daily_raw (
            code        VARCHAR NOT NULL,
            date        DATE    NOT NULL,
            open        DOUBLE,
            high        DOUBLE,
            low         DOUBLE,
            close       DOUBLE,
            volume      DOUBLE,
            amount      DOUBLE,
            pct_chg     DOUBLE,
            turn        DOUBLE,
            adj_factor  DOUBLE,
            PRIMARY KEY (code, date)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS daily_basic (
            code        VARCHAR NOT NULL,
            date        DATE    NOT NULL,
            total_mv    DOUBLE,
            circ_mv     DOUBLE,
            PRIMARY KEY (code, date)
        )
    """)
    log.info("表结构已创建/确认 (stock_info, daily_raw)")


def create_daily_kline_view(con):
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
    log.info("VIEW daily_kline 已创建（基于 daily_raw 实时前复权）")


def main():
    con = duckdb.connect(str(DB_PATH))
    con.execute("SET memory_limit = '2GB'")
    con.execute("SET threads = 4")

    stocks = load_all_pool_stocks()
    create_tables(con)

    # stock_info
    records = [
        (s["code"], s.get("name", ""), s.get("market", ""), s.get("full_code", ""))
        for s in stocks
    ]
    df_info = pd.DataFrame(records, columns=["code", "name", "market", "full_code"])
    con.execute("DELETE FROM stock_info")
    con.execute("INSERT INTO stock_info SELECT * FROM df_info")
    log.info("stock_info 已写入 %d 只", len(df_info))

    pro = _init_pro()
    all_daily, all_adj = [], []
    total_batches = (len(stocks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx, (ts_codes, codes) in enumerate(batch_ts_codes(stocks, BATCH_SIZE), 1):
        log.info("[批 %d/%d] %s ~ %s (%d 只)",
                 batch_idx, total_batches, codes[0], codes[-1], len(codes))

        df_d = fetch_daily(pro, ts_codes)
        if not df_d.empty:
            all_daily.append(df_d)
            log.info("  daily: %d 行", len(df_d))
        time.sleep(REQ_INTERVAL)

        df_a = fetch_adj_factor(pro, ts_codes)
        if not df_a.empty:
            all_adj.append(df_a)
            log.info("  adj_factor: %d 行", len(df_a))
        time.sleep(REQ_INTERVAL)

    log.info("合并数据 ...")
    daily = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
    adj = pd.concat(all_adj, ignore_index=True) if all_adj else pd.DataFrame()

    if daily.empty:
        log.error("未获取到任何日线数据，退出")
        con.close()
        return

    if not adj.empty:
        daily = daily.merge(adj, on=["code", "date"], how="left")
    else:
        daily["adj_factor"] = None

    daily["turn"] = None

    log.info("写入 daily_raw（%d 行）...", len(daily))
    con.execute("DELETE FROM daily_raw")
    con.execute("""
        INSERT INTO daily_raw(code, date, open, high, low, close,
                              volume, amount, pct_chg, turn, adj_factor)
        SELECT code, date, open, high, low, close,
               volume, amount, pct_chg, turn, adj_factor
        FROM daily
    """)
    con.execute("CHECKPOINT")

    create_daily_kline_view(con)

    row_count = con.execute("SELECT count(*) FROM daily_raw").fetchone()[0]
    stock_count = con.execute("SELECT count(DISTINCT code) FROM daily_raw").fetchone()[0]
    view_count = con.execute("SELECT count(*) FROM daily_kline").fetchone()[0]
    log.info("完成！daily_raw: %d 行 / %d 只股票，daily_kline(view): %d 行",
             row_count, stock_count, view_count)

    # ---- daily_basic（逐只拉取，relay 不支持批量） ----
    log.info("拉取 daily_basic (total_mv, circ_mv) ...")
    all_basic = []
    for idx, stock in enumerate(stocks, 1):
        df = fetch_daily_basic(pro, stock["full_code"])
        if not df.empty:
            all_basic.append(df)
        if idx % 50 == 0 or idx == len(stocks):
            log.info("  progress: %d/%d", idx, len(stocks))
        time.sleep(REQ_INTERVAL)

    if all_basic:
        basic = pd.concat(all_basic, ignore_index=True)
        con.execute("DELETE FROM daily_basic")
        con.execute("INSERT INTO daily_basic SELECT * FROM basic")
        log.info("daily_basic 写入 %d 行 / %d 只股票",
                 len(basic), basic["code"].nunique())
    else:
        log.warning("daily_basic 未获取到任何数据")

    # ---- 指数日线全量拉取 ----
    _ensure_index_daily_table(con)
    for ts_code, store_code in _TRACKED_INDICES:
        log.info("拉取指数 %s (%s) ...", store_code, ts_code)
        df_idx = _fetch_index(pro, ts_code, "20080101", "22220101")
        if df_idx.empty:
            log.warning("  指数 %s 无数据", ts_code)
            continue
        df_idx["code"] = store_code
        con.execute("DELETE FROM index_daily WHERE code=?", (store_code,))
        con.execute("INSERT INTO index_daily SELECT * FROM df_idx")
        n = con.execute("SELECT count(*) FROM index_daily WHERE code=?", (store_code,)).fetchone()[0]
        log.info("  index_daily 写入 %d 行", n)
        time.sleep(REQ_INTERVAL)

    con.close()


def fetch_daily_basic(pro, full_code):
    """拉取单只股票的 daily_basic（total_mv, circ_mv）。

    quicksync relay 的 daily_basic 不支持批量（ts_code 逗号分隔）
    也不支持 start_date/end_date 过滤，所以逐只全量拉取。
    """
    df = _retry_api(pro.daily_basic, ts_code=full_code)
    if df is None or df.empty:
        return pd.DataFrame()
    df["code"] = df["ts_code"].str.replace(r"[.](SH|SZ|BJ)$", "", regex=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["total_mv"] = pd.to_numeric(df["total_mv"], errors="coerce")
    df["circ_mv"] = pd.to_numeric(df["circ_mv"], errors="coerce")
    result = df[["code", "trade_date", "total_mv", "circ_mv"]].rename(columns={"trade_date": "date"})
    result = result.drop_duplicates(subset=["code", "date"])
    return result

# ---- 指数日线拉取 ----

_TRACKED_INDICES = [
    ("000985.CSI", "000985"),  # 中证全指
]


def _ensure_index_daily_table(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS index_daily (
            code    VARCHAR NOT NULL,
            date    DATE    NOT NULL,
            open    DOUBLE,
            high    DOUBLE,
            low     DOUBLE,
            close   DOUBLE,
            volume  DOUBLE,
            amount  DOUBLE,
            pct_chg DOUBLE,
            PRIMARY KEY (code, date)
        )
    """)


def _fetch_index(pro, ts_code, start, end):
    df = pro.index_daily(ts_code=ts_code, start_date=start, end_date=end)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"trade_date": "date", "vol": "volume"})
    df["date"] = pd.to_datetime(df["date"])
    cols = ["code", "date", "open", "high", "low", "close", "volume", "amount", "pct_chg"]
    return df[cols]


if __name__ == "__main__":
    main()
