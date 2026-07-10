# -*- coding: utf-8 -*-
"""
A股日线数据库 - 全量构建（按日拉版本）

数据来源：Tushare（通过 api.quicksync.cn 中转）

改进要点：
  - 按 trade_date 循环拉全市场数据，一次 API 拿到当天所有股票
  - 不再按股遍历，避免 relay 警告的"几千次遍历"
  - 等值查询（按日期）远快于范围扫描（按股票）
  - daily_basic 同时拉取市值、估值、涨跌停状态
  - 仅保留股票池中主板股票
  - 每 20 天批量写入 DuckDB
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

from config import DB_PATH

LOG_PATH = Path(__file__).parent / "build_db.log"
START_DATE = "20080101"
END_DATE   = None        # set None to pull up to today; use e.g. "20241231" for partial range
REQ_INTERVAL = 0.0       # relay response speed acts as natural rate limiter
BATCH_COMMIT = 20

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


def _get_trading_days(pro, start, end):
    """Get trading days via trade_cal; fallback to business-day probing."""
    try:
        df = _retry_api(pro.trade_cal, exchange='SSE',
                        start_date=start, end_date=end,
                        fields='cal_date,is_open')
        if df is not None and not df.empty:
            df = df[df['is_open'] == 1]
            days = sorted(df['cal_date'].tolist())
            log.info("trade_cal returned %d trading days", len(days))
            return days
    except Exception as e:
        log.warning("trade_cal failed (%s), using business days", e)

    date_range = pd.date_range(start=start, end=end, freq='B')
    return [d.strftime('%Y%m%d') for d in date_range]


def _to_code(df):
    """Strip exchange suffix from ts_code column (faster than regex)."""
    df["code"] = df["ts_code"].str[:6]
    return df


def _fetch_daily(pro, trade_date):
    df = _retry_api(pro.daily, trade_date=trade_date,
                    fields='ts_code,trade_date,open,high,low,close,vol,amount,pct_chg')
    if df is None or df.empty:
        return pd.DataFrame()
    _to_code(df)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df.rename(columns={"trade_date": "date", "vol": "volume"}, inplace=True)
    for col in ["open", "high", "low", "close", "volume", "amount", "pct_chg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["code", "date", "open", "high", "low", "close", "volume", "amount", "pct_chg"]]


def _fetch_adj_factor(pro, trade_date):
    df = _retry_api(pro.adj_factor, trade_date=trade_date,
                    fields='ts_code,trade_date,adj_factor')
    if df is None or df.empty:
        return pd.DataFrame()
    _to_code(df)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce")
    return df[["code", "trade_date", "adj_factor"]].rename(columns={"trade_date": "date"})


BASIC_FIELDS = "ts_code,trade_date,total_mv,circ_mv,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm"
BASIC_NUMERIC = ["total_mv", "circ_mv", "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio", "dv_ttm"]
BASIC_ALL_COLS = ["code", "date", "total_mv", "circ_mv", "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio", "dv_ttm"]


def _fetch_daily_basic(pro, trade_date):
    df = _retry_api(pro.daily_basic, ts_code="", trade_date=trade_date,
                    fields=BASIC_FIELDS)
    if df is None or df.empty:
        return pd.DataFrame()
    _to_code(df)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df.rename(columns={"trade_date": "date"}, inplace=True)
    for col in BASIC_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in BASIC_ALL_COLS:
        if col not in df.columns:
            df[col] = None
    cols = [c for c in df.columns if c != "ts_code"]
    return df[cols]


def create_tables(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS stock_info (
            code        VARCHAR PRIMARY KEY,
            name        VARCHAR,
            market      VARCHAR,
            full_code   VARCHAR,
            list_date   DATE
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
            pe          DOUBLE,
            pe_ttm      DOUBLE,
            pb          DOUBLE,
            ps          DOUBLE,
            ps_ttm      DOUBLE,
            dv_ratio    DOUBLE,
            dv_ttm      DOUBLE,
            PRIMARY KEY (code, date)
        )
    """)
    # Ensure new columns exist on existing tables (ALTER TABLE IF NOT EXISTS style)
    new_cols = [
        ("pe", "DOUBLE"),
        ("pe_ttm", "DOUBLE"),
        ("pb", "DOUBLE"),
        ("ps", "DOUBLE"),
        ("ps_ttm", "DOUBLE"),
        ("dv_ratio", "DOUBLE"),
        ("dv_ttm", "DOUBLE"),
    ]
    existing = con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name='daily_basic'"
    ).fetchall()
    existing_names = {r[0] for r in existing}
    for col_name, col_type in new_cols:
        if col_name not in existing_names:
            con.execute(f"ALTER TABLE daily_basic ADD COLUMN {col_name} {col_type}")
            log.info("Added column daily_basic.%s (%s)", col_name, col_type)

    log.info("Tables created / verified")



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
    log.info("VIEW daily_kline created")


def main():
    t_start = time.time()

    con = duckdb.connect(str(DB_PATH))
    con.execute("SET memory_limit = '2GB'")
    con.execute("SET threads = 4")

    pro = _init_pro()

    # Load all A-share stocks
    df_all = pro.stock_basic(exchange='', list_status='L',
                              fields='ts_code,symbol,name,area,industry,market,list_date')
    if df_all is not None and not df_all.empty:
        df_all["code"] = df_all["ts_code"].str[:6]
    else:
        log.error("stock_basic returned empty")
        return
    log.info("All A-shares: %d stocks", len(df_all))

    create_tables(con)

    # stock_info
    records = [(r["code"], r.get("name", ""), r.get("market", ""), r["ts_code"],
                pd.to_datetime(r.get("list_date", ""), format="%Y%m%d"))
               for _, r in df_all.iterrows()]
    df_info = pd.DataFrame(records, columns=["code", "name", "market", "full_code", "list_date"])
    con.execute("DELETE FROM stock_info")
    con.execute("INSERT INTO stock_info SELECT * FROM df_info")
    log.info("stock_info: %d rows", len(df_info))

    pro = _init_pro()

    # Trading days
    today_str = datetime.now().strftime("%Y%m%d")
    end_str = END_DATE if END_DATE else today_str
    log.info("Getting trading days: %s ~ %s", START_DATE, end_str)
    trading_days = _get_trading_days(pro, START_DATE, end_str)
    n_days = len(trading_days)
    log.info("Trading days to process: %d", n_days)

    # --- Main loop ---
    daily_buf, basic_buf = [], []
    processed, empty_days = 0, 0
    last_commit = 0

    for idx, td in enumerate(trading_days, 1):
        daily_items = 0
        basic_items = 0

        # 1. daily
        df_d = _fetch_daily(pro, td)

        # 2. adj_factor
        df_a = _fetch_adj_factor(pro, td)

        # 3. daily_basic
        df_b = _fetch_daily_basic(pro, td)
        time.sleep(REQ_INTERVAL)

        daily_items = len(df_d) if not df_d.empty else 0
        basic_items = len(df_b) if not df_b.empty else 0

        # Merge adj into daily
        if not df_d.empty:
            if not df_a.empty:
                df_d = df_d.merge(df_a, on=["code", "date"], how="left")
            else:
                df_d["adj_factor"] = None
            df_d["turn"] = None
            daily_buf.append(df_d)

        if not df_b.empty:
            basic_buf.append(df_b)

        if daily_items or basic_items:
            processed += 1
        else:
            empty_days += 1

        # Batch commit
        commit_now = (processed - last_commit >= BATCH_COMMIT) or (idx == n_days)
        if commit_now and (daily_buf or basic_buf):
            if daily_buf:
                df_all = pd.concat(daily_buf, ignore_index=True)
                con.execute("""
                    INSERT OR REPLACE INTO daily_raw
                        (code, date, open, high, low, close, volume, amount, pct_chg, turn, adj_factor)
                    SELECT code, date, open, high, low, close, volume, amount, pct_chg, turn, adj_factor
                    FROM df_all
                """)
                daily_buf = []
            if basic_buf:
                df_all = pd.concat(basic_buf, ignore_index=True)
                bs = "code,date,total_mv,circ_mv,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm"
                con.execute(f"INSERT OR REPLACE INTO daily_basic ({bs}) SELECT {bs} FROM df_all")
                basic_buf = []
            con.execute("CHECKPOINT")
            last_commit = processed
            elapsed = time.time() - t_start
            log.info("  committed %d days | elapsed %.0fs | ETA %s/%s",
                     BATCH_COMMIT, elapsed, idx, n_days)

        if idx % 50 == 0:
            log.info("[%d/%d] %s | daily %d basic %d", idx, n_days, td, daily_items, basic_items)

    # Final flush
    if daily_buf:
        df_all = pd.concat(daily_buf, ignore_index=True)
        con.execute("""
            INSERT OR REPLACE INTO daily_raw
                (code, date, open, high, low, close, volume, amount, pct_chg, turn, adj_factor)
            SELECT code, date, open, high, low, close, volume, amount, pct_chg, turn, adj_factor
            FROM df_all
        """)
    if basic_buf:
        df_all = pd.concat(basic_buf, ignore_index=True)
        bs = "code,date,total_mv,circ_mv,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm"
        con.execute(f"INSERT OR REPLACE INTO daily_basic ({bs}) SELECT {bs} FROM df_all")
    con.execute("CHECKPOINT")

    create_daily_kline_view(con)

    # ---- Stats ----
    raw_rows = con.execute("SELECT count(*) FROM daily_raw").fetchone()[0]
    raw_stocks = con.execute("SELECT count(DISTINCT code) FROM daily_raw").fetchone()[0]
    basic_rows = con.execute("SELECT count(*) FROM daily_basic").fetchone()[0]
    basic_stocks = con.execute("SELECT count(DISTINCT code) FROM daily_basic").fetchone()[0]

    elapsed = time.time() - t_start
    log.info("")
    log.info("=" * 50)
    log.info("BUILD COMPLETE  (%.0f seconds)", elapsed)
    log.info("  daily_raw:    %d rows / %d stocks", raw_rows, raw_stocks)
    log.info("  daily_basic:  %d rows / %d stocks", basic_rows, basic_stocks)
    log.info("  trading days: %d processed, %d empty", processed, empty_days)
    log.info("=" * 50)

    con.close()


if __name__ == "__main__":
    main()
