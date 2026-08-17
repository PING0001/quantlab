# -*- coding: utf-8 -*-
"""
Incremental factor update: compute factors for target dates only,
then write into factor_values table.

对账驱动：不再只比较 MAX(date)，而是对账 daily_kline 与 factor_values 的
日期集合--历史空洞（某天因子算到一半失败、人为删除）也会被找出并回补。

列所有权写入（factor_values 是双写入方表）：
- 新日期     -> INSERT（列子集；PK (code,date) 兜底防重）
- 已有日期   -> UPDATE ... FROM（绝不整行替换，保护 build_ai_factor 所有的
  ai_gz2000_* 列不被清空）

Usage:
    python -m factors.update
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import duckdb
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_PATH, get_pool_codes
from .compute import compute_panel
from . import integrity

log = logging.getLogger(__name__)

LOOKBACK_DAYS = 260  # trading days (~1 year, covers 250d windows + margin)


def get_latest_date(con: duckdb.DuckDBPyConnection) -> str | None:
    """Get the maximum date currently in factor_values."""
    try:
        result = con.execute("SELECT max(date) FROM factor_values").fetchone()
        if result and result[0]:
            return str(result[0])[:10]
    except Exception:
        pass
    return None


def get_missing_dates(con: duckdb.DuckDBPyConnection) -> list[str]:
    """daily_kline 有而 factor_values 缺的历史日期（升序）。"""
    rows = con.execute("""
        SELECT DISTINCT date FROM daily_kline
        EXCEPT
        SELECT DISTINCT date FROM factor_values
        ORDER BY date
    """).fetchall()
    return [str(r[0])[:10] for r in rows]


def get_lookback_start(con: duckdb.DuckDBPyConnection, from_date: str) -> str:
    """from_date 往前 LOOKBACK_DAYS 个交易日（含 from_date）窗口的最早日。

    必须基于 DISTINCT 交易日计算。原实现 `LIMIT 1 OFFSET LOOKBACK_DAYS-1`
    作用在行级 daily_kline（约 5500 行/天）上：259 行不足半日，返回的
    仍是 from_date 当天——增量因子因此只装到 1 天历史，长窗口因子
    （Return_20d/Reversal_60d 等）全 NULL，min_samples=1 类因子用短窗
    算出错误值（2026-07-13~08-17 因子污染事故根因）。
    """
    result = con.execute(
        """
        SELECT MIN(d) FROM (
            SELECT DISTINCT date AS d FROM daily_kline
            WHERE date <= ?::DATE
            ORDER BY date DESC LIMIT ?
        )
        """,
        [from_date, LOOKBACK_DAYS],
    ).fetchone()
    if result and result[0]:
        return str(result[0])[:10]
    return from_date


def write_panel(con: duckdb.DuckDBPyConnection, pdf: pd.DataFrame):
    """列所有权写入：新日期 INSERT，已有日期 UPDATE FROM。

    pdf 需含 code/date 及因子列；date 为 'YYYY-MM-DD' 字符串。
    """
    if pdf.empty:
        return 0, 0

    existing_cols = {r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name='factor_values'"
    ).fetchall()}
    factor_cols = [c for c in pdf.columns if c not in ("code", "date")]
    insert_cols = ["code", "date"] + [c for c in factor_cols if c in existing_cols]

    # 目标日期中已在 factor_values 的 -> UPDATE 回补（保留 ai_* 等他方列）
    dates = [d[:10] for d in pdf["date"].unique()]
    ph = ",".join(["?"] * len(dates))
    fv_dates = {str(r[0])[:10] for r in con.execute(
        f"SELECT DISTINCT date FROM factor_values WHERE date IN ({ph})", dates
    ).fetchall()}

    pdf = pdf.copy()
    pdf["date"] = pdf["date"].astype(str).str[:10]
    new_pdf = pdf[~pdf["date"].isin(fv_dates)]
    upd_pdf = pdf[pdf["date"].isin(fv_dates)]

    n_ins = n_upd = 0
    if not new_pdf.empty:
        sel = new_pdf[insert_cols]
        cols_str = ", ".join(insert_cols)
        con.execute(f"INSERT INTO factor_values ({cols_str}) SELECT * FROM sel", )
        n_ins = len(sel)

    if not upd_pdf.empty:
        con.execute("CREATE OR REPLACE TEMP TABLE _panel_upd AS SELECT * FROM upd_pdf")
        upd_factor_cols = [c for c in factor_cols if c in existing_cols]
        set_clause = ", ".join(f"{c} = p.{c}" for c in upd_factor_cols)
        con.execute(f"""
            UPDATE factor_values f SET {set_clause}
            FROM _panel_upd p
            WHERE f.code = p.code AND f.date = p.date
        """)
        n_upd = len(upd_pdf)

    return n_ins, n_upd


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    codes = get_pool_codes()
    log.info("Pool: %d stocks", len(codes))

    con = duckdb.connect(str(DB_PATH))
    con.execute("SET threads = 4")

    try:
        latest = get_latest_date(con)
        if not latest:
            log.warning("factor_values table is empty. Run full compute first.")
            return

        kline_max = con.execute("SELECT max(date) FROM daily_kline").fetchone()[0]
        kline_max = str(kline_max)[:10]
        if not kline_max:
            log.info("No kline data available.")
            return

        # ---- 对账：新增日期 + 历史空洞 ----
        missing = get_missing_dates(con)
        new_dates = sorted(d for d in
                           [str(r[0])[:10] for r in con.execute(
                               "SELECT DISTINCT date FROM daily_kline WHERE date > ?",
                               [latest]).fetchall()]
                           if d <= kline_max)
        target_dates = sorted(set(missing) | set(new_dates))

        if not target_dates:
            log.info("Factors are up to date (kline: %s).", kline_max)
            report = integrity.check(con)
            return

        if missing:
            log.warning("Historical holes found: %d dates (%s~%s), will backfill",
                        len(missing), missing[0], missing[-1])
        log.info("Target dates: %d (%s ~ %s)",
                 len(target_dates), target_dates[0], target_dates[-1])

        # ---- 计算：lookback 从最早目标日期起算 ----
        from_date = target_dates[0]
        lookback_start = get_lookback_start(con, from_date)
        log.info("Incremental range: lookback %s -> kline max %s", lookback_start, kline_max)

        panel = compute_panel(con, codes, start_date=lookback_start)
        if panel.is_empty():
            log.info("No new data to compute.")
            return

        panel = panel.filter(pl.col("date").is_in(target_dates))
        if panel.is_empty():
            log.info("No factor rows for target dates.")
            return

        # 去重（防御性；compute 侧 IsST 已向量化，重复根因已除）
        n_before = len(panel)
        panel = panel.unique(subset=["code", "date"], keep="last")
        if len(panel) < n_before:
            log.warning("Deduplicated panel: %d -> %d rows", n_before, len(panel))

        n_dates = panel["date"].n_unique()
        log.info("Factor rows to write: %d rows, %d dates", len(panel), n_dates)

        pdf = panel.to_pandas()
        pdf = pdf.sort_values(["date", "code"])

        n_ins, n_upd = write_panel(con, pdf)
        con.execute("CHECKPOINT")
        log.info("Written: %d inserted (new dates), %d updated (backfill)", n_ins, n_upd)

        # ---- 完整性校验（硬失败 exit 1 阻断下游；软警告仅记录）----
        report = integrity.check(con)
        if report["hard_fail"]:
            sys.exit(1)
    finally:
        con.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
