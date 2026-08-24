# -*- coding: utf-8 -*-
"""
Incremental factor update: compute factors for target dates only,
then write into factor_values table.

对账驱动：不再只比较 MAX(date)，而是对账 daily_kline 与 factor_values 的
日期集合--历史空洞（某天因子算到一半失败、人为删除）也会被找出并回补。
股票级对账：池内代码在 factor_values 缺失或历史覆盖显著偏低（相对其在
daily_kline 的应有交易日数）的纳入回补清单——池扩容后新成员的历史缺口
对日期级对账不可见（2026-07 池 614→1112，594 只新码历史从未回补）。
默认只告警，需显式 --backfill-stocks 才执行回补写入（保守，避免日常
增量意外触发大计算）；--dry-run 仅预览不计算不写库。

列所有权写入（factor_values 是双写入方表）：
- 缺失 (code,date) 行 -> INSERT（列子集；PK (code,date) 兜底防重）
- 已有行             -> UPDATE ... FROM（绝不整行替换，保护 build_ai_factor
  所有的 ai_gz2000_* 列不被清空）

Usage:
    python -m factors.update                     # 日期级增量（默认）
    python -m factors.update --dry-run           # 预览目标日期 + 股票级回补清单
    python -m factors.update --backfill-stocks   # 额外执行股票级历史回补（大计算）
"""

from __future__ import annotations

import argparse
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

# 股票级对账覆盖阈值：池代码在 factor_values 的行数 < 其 daily_kline 行数
# ×该值即纳入回补清单（当前库实测：594 只新成员覆盖 <10%，老成员 ≈100%，
# 阈值不敏感；留 5% 容差吸收个别数据缺口）
STOCK_COVERAGE_MIN = 0.95


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


def get_backfill_stocks(
    con: duckdb.DuckDBPyConnection, codes: list[str]
) -> list[tuple[str, int, int]]:
    """股票级对账：池内代码在 factor_values 覆盖不足的清单。

    返回 [(code, 应有行数, 实有行数)]（升序）。应有 = 该码在 daily_kline
    的行数（fv 与 kline 同源于 2008 起，停牌日本就无行），实有 < 应有 ×
    STOCK_COVERAGE_MIN 即纳入。日期级对账看不见这类缺口：池扩容后新码的
    历史日期在 factor_values 里已有旧池股票的行，"日期集合"判定无缺失。
    """
    rows = con.execute(
        """
        WITH pool AS (SELECT DISTINCT unnest(?::VARCHAR[]) AS code),
        k AS (SELECT code, COUNT(*) AS n FROM daily_kline GROUP BY code),
        f AS (SELECT code, COUNT(*) AS n FROM factor_values GROUP BY code)
        SELECT p.code, COALESCE(k.n, 0), COALESCE(f.n, 0)
        FROM pool p
        LEFT JOIN k ON k.code = p.code
        LEFT JOIN f ON f.code = p.code
        WHERE COALESCE(k.n, 0) > 0
          AND COALESCE(f.n, 0) < COALESCE(k.n, 0) * ?
        ORDER BY p.code
        """,
        [codes, STOCK_COVERAGE_MIN],
    ).fetchall()
    return [(str(r[0]), int(r[1]), int(r[2])) for r in rows]


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
    """列所有权写入：缺失 (code,date) 行 INSERT，已有行 UPDATE FROM。

    pdf 需含 code/date 及因子列；date 为 'YYYY-MM-DD' 字符串。

    行级（而非日期级）分流：股票级回补的新码，其历史日期在表内已存在
    （旧池股票的行），按日期分流会把这些行全部判进 UPDATE 分支而静默
    丢失（P0-B 机制之一）。
    """
    if pdf.empty:
        return 0, 0

    existing_cols = {r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name='factor_values'"
    ).fetchall()}
    factor_cols = [c for c in pdf.columns if c not in ("code", "date")]
    insert_cols = ["code", "date"] + [c for c in factor_cols if c in existing_cols]
    upd_factor_cols = [c for c in factor_cols if c in existing_cols]

    pdf = pdf.copy()
    pdf["date"] = pdf["date"].astype(str).str[:10]
    con.execute("CREATE OR REPLACE TEMP TABLE _panel_upd AS SELECT * FROM pdf")

    # 表内缺失行 -> INSERT（列子集；PK (code,date) 兜底防重）
    n_ins = con.execute("""
        SELECT COUNT(*) FROM _panel_upd p
        WHERE NOT EXISTS (
            SELECT 1 FROM factor_values f
            WHERE f.code = p.code AND f.date = p.date
        )
    """).fetchone()[0]
    if n_ins:
        cols_str = ", ".join(insert_cols)
        con.execute(f"""
            INSERT INTO factor_values ({cols_str})
            SELECT {cols_str} FROM _panel_upd p
            WHERE NOT EXISTS (
                SELECT 1 FROM factor_values f
                WHERE f.code = p.code AND f.date = p.date
            )
        """)

    # 已有行 -> UPDATE 回补（保留 ai_* 等他方列）
    if upd_factor_cols:
        set_clause = ", ".join(f"{c} = p.{c}" for c in upd_factor_cols)
        con.execute(f"""
            UPDATE factor_values f SET {set_clause}
            FROM _panel_upd p
            WHERE f.code = p.code AND f.date = p.date
        """)
    n_upd = con.execute("""
        SELECT COUNT(*) FROM _panel_upd p
        JOIN factor_values f ON f.code = p.code AND f.date = p.date
    """).fetchone()[0]

    return n_ins, n_upd


def run_stock_backfill(con: duckdb.DuckDBPyConnection, codes: list[str]):
    """股票级历史回补：复用 compute_panel 全历史重算 + 列所有权写入。"""
    log.info("Backfilling %d codes (full history via compute_panel) ...", len(codes))
    panel = compute_panel(con, codes)
    if panel.is_empty():
        log.warning("Stock backfill: empty panel, nothing written.")
        return
    panel = panel.unique(subset=["code", "date"], keep="last")
    pdf = panel.to_pandas()
    pdf = pdf.sort_values(["date", "code"])
    n_ins, n_upd = write_panel(con, pdf)
    con.execute("CHECKPOINT")
    log.info("Stock backfill written: %d inserted, %d updated", n_ins, n_upd)


def main():
    parser = argparse.ArgumentParser(
        description="Incremental factor update with date + stock level reconciliation")
    parser.add_argument("--dry-run", action="store_true",
                        help="只预览目标日期与股票级回补清单，不计算不写库")
    parser.add_argument("--backfill-stocks", action="store_true",
                        help="对股票级对账发现的覆盖不足代码执行全历史回补写入"
                             "（大计算，默认关闭，仅告警）")
    args = parser.parse_args()

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

        # ---- 对账：股票级覆盖（池扩容后新成员历史缺口，日期级对账盲区）----
        backfill = get_backfill_stocks(con, codes)
        if backfill:
            est_rows = sum(k - f for _, k, f in backfill)
            sample = ", ".join(c for c, _, _ in backfill[:5])
            log.warning(
                "Stock-level gaps: %d/%d pool codes under-covered in factor_values "
                "(<%.0f%% of daily_kline rows), ~%d rows to backfill. Sample: %s%s "
                "Run with --backfill-stocks to fix.",
                len(backfill), len(codes), STOCK_COVERAGE_MIN * 100, est_rows,
                sample, "..." if len(backfill) > 5 else "",
            )
        else:
            log.info("Stock-level coverage OK (all pool codes >= %.0f%%).",
                     STOCK_COVERAGE_MIN * 100)

        if not target_dates and not backfill:
            log.info("Factors are up to date (kline: %s).", kline_max)
            report = integrity.check(con)
            return

        if missing:
            log.warning("Historical holes found: %d dates (%s~%s), will backfill",
                        len(missing), missing[0], missing[-1])
        if target_dates:
            log.info("Target dates: %d (%s ~ %s)",
                     len(target_dates), target_dates[0], target_dates[-1])

        if args.dry_run:
            log.info("[dry-run] Would compute dates: %d; "
                     "stock backfill: %d codes, ~%d rows. No computation, no writes.",
                     len(target_dates),
                     len(backfill), sum(k - f for _, k, f in backfill))
            return

        if target_dates:
            # ---- 计算：lookback 锚定最早目标日 ----
            # 2026-08-24 审计 #7 修正：锚"最晚"目标日时，回补跨度 g>8 个交易日
            # 就截断最早目标日的 252d 长窗（完整需 g ≤ 260−252=8）；锚"最早"
            # 目标日则任意跨度完整（窗口 [t0−260, kline_max] ⊇ 全部目标日需求
            # [t0−252, tN]）。此前 F5 注释的论证恰好说反。
            from_date = target_dates[0]
            lookback_start = get_lookback_start(con, from_date)
            log.info("Incremental range: lookback %s -> kline max %s", lookback_start, kline_max)

            panel = compute_panel(con, codes, start_date=lookback_start)
            if panel.is_empty():
                log.info("No new data to compute.")
            else:
                panel = panel.filter(pl.col("date").is_in(target_dates))
                if panel.is_empty():
                    log.info("No factor rows for target dates.")
                else:
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
                    log.info("Written: %d inserted (new dates), %d updated (backfill)",
                             n_ins, n_upd)

        # ---- 股票级回补：需显式 --backfill-stocks，复用全历史计算路径 ----
        if backfill and args.backfill_stocks:
            run_stock_backfill(con, [c for c, _, _ in backfill])
        elif backfill:
            log.info("Stock backfill skipped (%d codes, ~%d rows); "
                     "pass --backfill-stocks to execute.",
                     len(backfill), sum(k - f for _, k, f in backfill))

        # ---- 完整性校验（硬失败 exit 1 阻断下游；软警告仅记录）----
        report = integrity.check(con)
        if report["hard_fail"]:
            sys.exit(1)
    finally:
        con.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
