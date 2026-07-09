# -*- coding: utf-8 -*-
"""
A股日线数据库 - 增量更新（按日拉版本）

每天运行：查询最新交易日，一次 API 拉取全市场数据。
与 build_db.py 使用相同的按日拉模式，但只拉取最近一个交易日。
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

LOG_PATH = Path(__file__).parent / "pull_adj.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BASIC_FIELDS = "ts_code,trade_date,total_mv,circ_mv,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm"
BASIC_NUMERIC = ["total_mv", "circ_mv", "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio", "dv_ttm"]

TRACKED_INDICES = [
    ("000985.CSI", "000985"),   # 中证全指
    ("000300.SH",  "000300"),   # 沪深300
]


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


def _to_code(df):
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
    cols = [c for c in df.columns if c != "ts_code"]
    return df[cols]


def _get_new_trading_days(pro, con, today_str):
    """Find trading days after the last date in daily_raw."""
    max_d = con.execute("SELECT MAX(date) FROM daily_raw").fetchone()[0]
    if max_d is None:
        log.warning("daily_raw is empty, run build_db.py first")
        return []

    max_date = pd.Timestamp(max_d)
    start = (max_date + timedelta(days=1)).strftime("%Y%m%d")
    if start > today_str:
        log.info("Already up to date (latest: %s)", max_date.date())
        return []

    try:
        df = _retry_api(pro.trade_cal, exchange='SSE',
                        start_date=start, end_date=today_str,
                        fields='cal_date,is_open')
        if df is not None and not df.empty:
            days = sorted(df[df['is_open'] == 1]['cal_date'].tolist())
            return days
    except Exception as e:
        log.warning("trade_cal failed: %s", e)

    return []


def _ensure_table_schema(con):
    """Ensure daily_basic has all required columns."""
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
    new_cols = [
        ("pe", "DOUBLE"), ("pe_ttm", "DOUBLE"), ("pb", "DOUBLE"),
        ("ps", "DOUBLE"), ("ps_ttm", "DOUBLE"), ("dv_ratio", "DOUBLE"),
        ("dv_ttm", "DOUBLE"),
    ]
    existing = con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name='daily_basic'"
    ).fetchall()
    existing_names = {r[0] for r in existing}
    for col_name, col_type in new_cols:
        if col_name not in existing_names:
            con.execute(f"ALTER TABLE daily_basic ADD COLUMN {col_name} {col_type}")
            log.info("Added column daily_basic.%s", col_name)


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
    if "trade_date" in df.columns:
        df = df.rename(columns={"trade_date": "date", "vol": "volume"})
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d").dt.date
    return df[["date", "open", "high", "low", "close", "volume", "amount", "pct_chg"]]


def _ensure_view(con):
    try:
        con.execute("SELECT count(*) FROM daily_kline LIMIT 1")
    except Exception:
        log.info("daily_kline VIEW missing, creating ...")
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
        log.info("daily_kline VIEW created")


def main():
    con = duckdb.connect(str(DB_PATH))
    con.execute("SET memory_limit = '2GB'")
    con.execute("SET threads = 4")

    _ensure_table_schema(con)
    pro = _init_pro()

    today_str = datetime.now().strftime("%Y%m%d")

    # --- 1. Find missing trading days ---
    new_days = _get_new_trading_days(pro, con, today_str)
    if not new_days:
        log.info("No new trading days to pull.")
    else:
        log.info("New trading days: %d (%s ~ %s)", len(new_days), new_days[0],
                 new_days[-1] if len(new_days) > 1 else new_days[0])

        # --- 2. Pull each new day ---
        for idx, td in enumerate(new_days, 1):
            log.info("[%d/%d] %s", idx, len(new_days), td)

            df_d = _fetch_daily(pro, td)
            df_a = _fetch_adj_factor(pro, td)
            df_b = _fetch_daily_basic(pro, td)

            daily_items = len(df_d) if not df_d.empty else 0
            basic_items = len(df_b) if not df_b.empty else 0

            if not df_d.empty:
                if not df_a.empty:
                    df_d = df_d.merge(df_a, on=["code", "date"], how="left")
                else:
                    df_d["adj_factor"] = None
                df_d["turn"] = None
                con.execute("""
                    INSERT OR REPLACE INTO daily_raw
                        (code, date, open, high, low, close, volume, amount, pct_chg, turn, adj_factor)
                    SELECT code, date, open, high, low, close, volume, amount, pct_chg, turn, adj_factor
                    FROM df_d
                """)

            if not df_b.empty:
                bs = "code,date,total_mv,circ_mv,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm"
                con.execute(f"INSERT OR REPLACE INTO daily_basic ({bs}) SELECT {bs} FROM df_b")

            con.execute("CHECKPOINT")
            log.info("  daily: %d rows  basic: %d rows", daily_items, basic_items)

        row_count = con.execute("SELECT count(*) FROM daily_raw").fetchone()[0]
        log.info("Incremental complete: daily_raw now %d rows", row_count)

    # --- 3. Rebuild view ---
    _ensure_view(con)

    # --- 4. Index daily ---
    _ensure_index_daily_table(con)
    for ts_code, store_code in TRACKED_INDICES:
        max_d_idx = con.execute(
            "SELECT MAX(date) FROM index_daily WHERE code=?", (store_code,)
        ).fetchone()[0]
        if max_d_idx is not None:
            max_d_idx = pd.Timestamp(max_d_idx)
            if max_d_idx >= pd.Timestamp(datetime.now().date()):
                log.info("Index %s up to date (%s)", store_code, max_d_idx.date())
                continue
            start_idx = (max_d_idx + timedelta(days=1)).strftime("%Y%m%d")
        else:
            start_idx = "20080101"

        log.info("Pulling index %s: %s ~ %s", store_code, start_idx, today_str)
        df_idx = _fetch_index(pro, ts_code, start_idx, today_str)
        if df_idx.empty:
            log.info("  no new data")
            continue
        df_idx["code"] = store_code
        df_idx = df_idx[["code", "date", "open", "high", "low", "close", "volume", "amount", "pct_chg"]]
        con.execute(
            "DELETE FROM index_daily WHERE code=? AND date>=?",
            (store_code, pd.to_datetime(start_idx).date())
        )
        con.execute("INSERT INTO index_daily SELECT * FROM df_idx")
        n = con.execute("SELECT count(*) FROM index_daily WHERE code=?", (store_code,)).fetchone()[0]
        log.info("  index_daily: %d rows", n)

    con.close()
    log.info("Pull complete.")


if __name__ == "__main__":
    main()
