"""
Build delist_info table: pull namechange for pool stocks, extract
delisting (termination) dates and ST periods.

Stores to DuckDB for use by label computation and training.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from datetime import datetime

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
import tushare as ts
import tushare.pro.client as client
client.DataApi._DataApi__http_url = "http://api.quicksync.cn"

from config import DB_PATH, get_pool_codes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def _init_pro():
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path, encoding="utf-8-sig")
    import os
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN not found in .env")
    ts.set_token(token)
    pro = ts.pro_api()
    pro._DataApi__http_timeout = 120
    return pro


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


def ensure_tables(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS namechange (
            code        VARCHAR NOT NULL,
            ts_code     VARCHAR NOT NULL,
            name        VARCHAR,
            start_date  DATE,
            end_date    DATE,
            ann_date    DATE,
            change_reason VARCHAR,
            PRIMARY KEY (code, start_date)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS delist_info (
            code            VARCHAR PRIMARY KEY,
            delist_date     DATE NOT NULL
        )
    """)


def pull_namechange_for_codes(pro, codes, con):
    """Pull namechange history for a list of 6-digit codes.
    Uses stock_info.full_code for correct exchange suffix.
    """
    code_to_ts = {}
    if codes:
        placeholders = ",".join(["?"] * len(codes))
        rows = con.execute(
            f"SELECT code, full_code FROM stock_info WHERE code IN ({placeholders})",
            codes,
        ).fetchall()
        for code, full_code in rows:
            code_to_ts[code] = full_code

    start_date = "20080101"
    end_date = datetime.now().strftime("%Y%m%d")

    all_rows = []
    failed = []
    total = len(codes)
    for i, code in enumerate(codes, 1):
        ts_code = code_to_ts.get(code)
        if ts_code is None:
            if code.startswith("6"):
                ts_code = f"{code}.SH"
            elif code.startswith(("0", "3")):
                ts_code = f"{code}.SZ"
            elif code.startswith(("4", "8")):
                ts_code = f"{code}.BJ"
            else:
                ts_code = f"{code}.SZ"

        try:
            df = _retry_api(pro.namechange, ts_code=ts_code,
                            start_date=start_date, end_date=end_date)
        except Exception as e:
            log.warning("  skip %s (%s): %s", code, ts_code, e)
            failed.append(code)
            continue

        if df is None or df.empty:
            # Retry once for empty responses (possible silent API failure)
            time.sleep(0.5)
            try:
                df = _retry_api(pro.namechange, ts_code=ts_code,
                                start_date=start_date, end_date=end_date)
            except Exception:
                pass

            if df is None or df.empty:
                if i % 100 == 0:
                    log.info("  [%d/%d] %s: no changes", i, total, code)
                continue

        df["code"] = code
        all_rows.append(df)
        if i % 100 == 0:
            log.info("  [%d/%d] %s: %d name changes", i, total, code, len(df))

    if failed:
        log.warning("Failed to pull namechange for %d codes: %s", len(failed), failed)

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)
    combined["start_date"] = pd.to_datetime(combined["start_date"], errors="coerce").dt.date
    combined["end_date"] = pd.to_datetime(combined["end_date"], errors="coerce")
    combined["ann_date"] = pd.to_datetime(combined["ann_date"], errors="coerce").dt.date
    return combined[["code", "ts_code", "name", "start_date", "end_date",
                     "ann_date", "change_reason"]]


def extract_delist_info(namechange_df):
    """Extract delisting dates from namechange data.

    Only uses change_reason == '终止上市' as the authoritative signal.
    """
    if namechange_df.empty:
        return pd.DataFrame(columns=["code", "delist_date"])

    term = namechange_df[namechange_df["change_reason"] == "终止上市"].copy()
    if term.empty:
        return pd.DataFrame(columns=["code", "delist_date"])

    term_dates = term.groupby("code")["start_date"].min().reset_index()
    term_dates.columns = ["code", "delist_date"]
    return term_dates


def main():
    con = duckdb.connect(str(DB_PATH))
    ensure_tables(con)

    pro = _init_pro()
    codes = get_pool_codes()
    log.info("Pool: %d stocks", len(codes))

    log.info("Pulling namechange for %d pool stocks ...", len(codes))
    t_start = time.time()
    df = pull_namechange_for_codes(pro, codes, con)
    elapsed = time.time() - t_start
    log.info("Total namechange rows: %d (%.0fs)", len(df), elapsed)

    if df.empty:
        log.warning("No namechange data returned.")
        con.close()
        return

    con.execute("DELETE FROM namechange")
    # Dedup in case namechange API returns duplicates for same (code, start_date)
    df = df.drop_duplicates(subset=["code", "start_date"], keep="first")
    con.execute("INSERT INTO namechange SELECT * FROM df")
    con.execute("CHECKPOINT")
    n_nc = con.execute("SELECT count(*) FROM namechange").fetchone()[0]
    log.info("namechange table: %d rows", n_nc)

    delist_df = extract_delist_info(df)
    if not delist_df.empty:
        con.execute("DELETE FROM delist_info")
        con.execute("INSERT INTO delist_info SELECT * FROM delist_df")
        con.execute("CHECKPOINT")
    n_di = con.execute("SELECT count(*) FROM delist_info").fetchone()[0]
    log.info("delist_info table: %d stocks", n_di)

    print()
    print("=" * 60)
    print("  Delisted stocks in pool:")
    if not delist_df.empty:
        for _, r in delist_df.iterrows():
            code = r["code"]
            name_row = con.execute(
                "SELECT name FROM stock_info WHERE code=?", [code]
            ).fetchone()
            name = name_row[0] if name_row else "?"
            print(f"    {code}  {name:<12s}  delist_date: {r['delist_date']}")
    else:
        print("    None found")
    print("=" * 60)

    con.close()


if __name__ == "__main__":
    main()
