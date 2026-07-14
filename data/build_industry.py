# -*- coding: utf-8 -*-
"""
A股行业分类数据 - 全量构建 + 增量更新

数据来源：Tushare（通过 api.quicksync.cn 中转）

分类体系：
  - Tushare 细分行业（stock_basic.industry，~110类）
  - 申万一级行业（SW2021 L1，31类）
  - 申万二级行业（SW2021 L2，134类）
  - 申万三级行业（SW2021 L3，346类）

用法：
    python data/build_industry.py           # 增量拉取（仅补充缺失股票）
"""
from __future__ import annotations
import logging, sys, time
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
import tushare as ts
import tushare.pro.client as client
client.DataApi._DataApi__http_url = "http://api.quicksync.cn"

from config import DB_PATH

LOG_PATH = Path(__file__).parent / "build_industry.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
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
            log.warning("Retry %d/%d: %s, waiting %ds...", attempt + 1, max_retries, e, wait)
            time.sleep(wait)


def _ensure_table(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS industry (
            code                VARCHAR PRIMARY KEY,
            industry_tushare    VARCHAR,
            sw_l1_code          VARCHAR,
            sw_l1_name          VARCHAR,
            sw_l2_code          VARCHAR,
            sw_l2_name          VARCHAR,
            sw_l3_code          VARCHAR,
            sw_l3_name          VARCHAR
        )
    """)
    # Add missing columns if table already existed with old schema
    existing = {r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name='industry'"
    ).fetchall()}
    for col, col_type in [
        ("sw_l2_code", "VARCHAR"), ("sw_l2_name", "VARCHAR"),
        ("sw_l3_code", "VARCHAR"), ("sw_l3_name", "VARCHAR"),
    ]:
        if col not in existing:
            con.execute(f"ALTER TABLE industry ADD COLUMN {col} {col_type}")
            log.info("Added column industry.%s", col)


def _pull_tushare_industry(pro):
    """Pull Tushare industry from stock_basic for all listed stocks."""
    df = _retry_api(pro.stock_basic, exchange='', list_status='L',
                    fields='ts_code,industry')
    if df is None or df.empty:
        return pd.DataFrame()

    df["code"] = df["ts_code"].str[:6]
    return df[["code", "industry"]].rename(columns={"industry": "industry_tushare"})


def _pull_sw_level(pro, level):
    """Pull SW classification for one level (L1/L2/L3).

    Returns DataFrame with columns: code, sw_{level}_code, sw_{level}_name
    """
    ldf = _retry_api(pro.index_classify, level=level, src='SW2021')
    if ldf is None or ldf.empty:
        log.error("index_classify(%s) returned empty", level)
        return pd.DataFrame()

    log.info("SW2021 %s: %d sectors", level, len(ldf))

    col_code = f"sw_{level.lower()}_code"
    col_name = f"sw_{level.lower()}_name"

    all_rows = []
    for _, row in ldf.iterrows():
        idx_code = row["index_code"]
        idx_name = row["industry_name"]

        try:
            mdf = _retry_api(pro.index_member, index_code=idx_code)
        except Exception as e:
            log.warning("  index_member %s failed: %s", idx_code, e)
            continue

        if mdf is None or mdf.empty:
            continue

        for _, m in mdf.iterrows():
            all_rows.append({
                "code": m["con_code"][:6],
                col_code: idx_code,
                col_name: idx_name,
            })

        n = len(mdf)
        if n > 0 and (len(ldf) <= 50 or len(all_rows) % 500 == 0):
            log.info("  %s (%s): %d members", idx_name, idx_code, n)
        time.sleep(0.3)

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


def main():
    t_start = time.time()

    con = duckdb.connect(str(DB_PATH))
    con.execute("SET threads = 4")
    _ensure_table(con)

    pro = _init_pro()

    # Get all stock codes from stock_info
    stock_codes = con.execute("SELECT code FROM stock_info").fetchdf()["code"].tolist()
    log.info("stock_info: %d stocks", len(stock_codes))

    # Check which stocks already have data
    existing = con.execute("SELECT code FROM industry").fetchdf()["code"].tolist()
    existing_set = set(existing)
    missing = [c for c in stock_codes if c not in existing_set]
    log.info("industry table: %d stocks, missing %d", len(existing_set), len(missing))

    # Check if table has all required columns and data
    cols = {r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name='industry'"
    ).fetchall()}
    has_all_cols = "sw_l2_code" in cols and "sw_l3_code" in cols
    has_l2_data = con.execute(
        "SELECT count(*) FROM industry WHERE sw_l2_code IS NOT NULL"
    ).fetchone()[0] > 0

    if not missing and has_all_cols and has_l2_data:
        log.info("All stocks already have industry data. Nothing to do.")
        con.close()
        return

    if not missing and (not has_all_cols or not has_l2_data):
        log.info("Table exists but L2/L3 data missing, will rebuild SW data.")

    # 1. Pull Tushare industry
    log.info("Pulling Tushare industry from stock_basic ...")
    tushare_df = _pull_tushare_industry(pro)
    log.info("  Tushare industry: %d stocks", len(tushare_df))

    # 2. Pull SW L1/L2/L3
    sw_dfs = {}
    for level in ["L1", "L2", "L3"]:
        log.info("Pulling SW %s classification ...", level)
        df = _pull_sw_level(pro, level)
        sw_dfs[level] = df
        log.info("  SW %s: %d stocks", level, len(df))

    # 3. Merge: start with stock_info, left join each level
    merged = pd.DataFrame({"code": stock_codes})
    merged = merged.merge(tushare_df, on="code", how="left")

    for level in ["L1", "L2", "L3"]:
        df = sw_dfs[level]
        if not df.empty:
            merged = merged.merge(df, on="code", how="left")

    # Ensure all columns exist
    for col in ["sw_l1_code", "sw_l1_name", "sw_l2_code", "sw_l2_name",
                "sw_l3_code", "sw_l3_name"]:
        if col not in merged.columns:
            merged[col] = None

    merged = merged[["code", "industry_tushare",
                     "sw_l1_code", "sw_l1_name",
                     "sw_l2_code", "sw_l2_name",
                     "sw_l3_code", "sw_l3_name"]]

    n_tushare = merged["industry_tushare"].notna().sum()
    n_l1 = merged["sw_l1_code"].notna().sum()
    n_l2 = merged["sw_l2_code"].notna().sum()
    n_l3 = merged["sw_l3_code"].notna().sum()
    log.info("Merged: %d stocks, Tushare %d, L1 %d, L2 %d, L3 %d",
             len(merged), n_tushare, n_l1, n_l2, n_l3)

    con.execute("INSERT OR REPLACE INTO industry SELECT * FROM merged")
    con.execute("CHECKPOINT")

    # Stats
    total = con.execute("SELECT count(*) FROM industry").fetchone()[0]
    t_tushare = con.execute("SELECT count(*) FROM industry WHERE industry_tushare IS NOT NULL").fetchone()[0]
    t_l1 = con.execute("SELECT count(*) FROM industry WHERE sw_l1_code IS NOT NULL").fetchone()[0]
    t_l2 = con.execute("SELECT count(*) FROM industry WHERE sw_l2_code IS NOT NULL").fetchone()[0]
    t_l3 = con.execute("SELECT count(*) FROM industry WHERE sw_l3_code IS NOT NULL").fetchone()[0]

    elapsed = time.time() - t_start
    log.info("")
    log.info("=" * 50)
    log.info("BUILD COMPLETE  (%.0f seconds)", elapsed)
    log.info("  industry: %d stocks total", total)
    log.info("  with Tushare industry: %d", t_tushare)
    log.info("  with SW L1: %d", t_l1)
    log.info("  with SW L2: %d", t_l2)
    log.info("  with SW L3: %d", t_l3)
    log.info("=" * 50)

    con.close()


if __name__ == "__main__":
    main()
