"""
Store four index daily data into DuckDB.
上证50: 000016.SH  沪深300: 000300.SH  中证2000: 932000.CSI  上证指数: 000001.SH
"""
import os, sys, time, logging
from pathlib import Path
from dotenv import load_dotenv

PROJ_ROOT = Path(r"C:\Users\cui\Documents\quantlab")
sys.path.insert(0, str(PROJ_ROOT))

import duckdb
import pandas as pd
import tushare as ts
import tushare.pro.client as client
client.DataApi._DataApi__http_url = "http://api.quicksync.cn"

from config import DB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

load_dotenv(PROJ_ROOT / ".env", encoding="utf-8-sig")
ts.set_token(os.getenv("TUSHARE_TOKEN"))
pro = ts.pro_api()
pro._DataApi__http_timeout = 120

INDICES = [
    ("000001.SH", "上证指数"),
    ("000016.SH", "上证50"),
    ("000300.SH", "沪深300"),
    ("932000.CSI", "中证2000"),
]

con = duckdb.connect(str(DB_PATH))

# Create table
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

for ts_code, name in INDICES:
    code = ts_code.split(".")[0]
    log.info("Fetching %s (%s) ...", name, ts_code)
    time.sleep(0.5)
    try:
        df = pro.index_daily(ts_code=ts_code, start_date="20080101", end_date="22220101")
        if df is None or df.empty:
            log.warning("  No data for %s", ts_code)
            continue

        df = df.rename(columns={
            "trade_date": "date",
            "vol": "volume",
        })
        df["code"] = code
        df["date"] = pd.to_datetime(df["date"])
        cols = ["code", "date", "open", "high", "low", "close", "volume", "amount", "pct_chg"]
        df = df[cols]

        con.execute("DELETE FROM index_daily WHERE code = ?", (code,))
        con.execute("INSERT INTO index_daily SELECT * FROM df")
        log.info("  Stored %d rows", len(df))
    except Exception as e:
        log.error("  Failed: %s", e)

# Verify
log.info("\nVerification:")
for ts_code, name in INDICES:
    code = ts_code.split(".")[0]
    r = con.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM index_daily WHERE code = ?", (code,)).fetchone()
    log.info("  %s (%s): %d rows, %s ~ %s", name, code, r[0], r[1], r[2])

con.execute("CHECKPOINT")
con.close()
log.info("\nDone.")
