# -*- coding: utf-8 -*-
"""数据完整性校验：硬失败/软警告分级。

workbuddy 自动化"任何一步失败即停止"，因此分级至关重要：
- 硬失败（exit 1，阻断下游 generate_lgb）：最新开市日的 factor_values 无数据
  --当日预测不可产出，继续没有意义；
- 软警告（exit 0，仅记录）：历史缺口、行数漂移、adj_factor 缺失、cyq pending
  --可见但不砍断当日产品。

日历对账依赖 trading_calendar（trade_cal 唯一真相）；日历不可用时该组检查
降级跳过并在报告中标注 degraded，绝不用业务表兜底（循环论证）。

用法：
    python -m factors.integrity     # 独立运行，exit 1 = 硬失败
    （factors.update 末尾自动调用）
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_PATH

log = logging.getLogger(__name__)

REPORT_PATH = DB_PATH.parent / "integrity_report.json"

# 行数漂移监控的表（trading 粒度）
ROW_DRIFT_TABLES = ["daily_raw", "daily_basic", "cyq_perf", "factor_values"]


def _dates_in(con, table: str) -> list[str]:
    try:
        return [str(r[0])[:10] for r in
                con.execute(f"SELECT DISTINCT date FROM {table} ORDER BY date").fetchall()]
    except Exception:
        return []


def _latest_open_day(con) -> tuple[str, bool]:
    """返回 (最新开市日, 日历是否可用)。日历缺失时降级用 daily_kline 最大日。"""
    try:
        from data import calendar as cal
        d = cal.latest(con)
        if d:
            return d, True
    except Exception:
        pass
    d = con.execute("SELECT MAX(date) FROM daily_kline").fetchone()[0]
    return (str(d)[:10], False) if d else (None, False)


def run_checks(con: duckdb.DuckDBPyConnection) -> dict:
    report = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "hard_fail": False,
        "hard_fail_reason": None,
        "soft_warnings": [],
        "degraded": False,
        "tables": {},
    }

    latest_open, cal_ok = _latest_open_day(con)
    report["latest_open_day"] = latest_open
    if not cal_ok:
        report["degraded"] = True
        report["soft_warnings"].append("trading_calendar 不可用，日历对账已跳过（降级）")

    fv_dates = set(_dates_in(con, "factor_values"))
    kline_dates = set(_dates_in(con, "daily_kline"))

    # ---- 硬失败：最新开市日 factor_values 无数据 ----
    if latest_open and latest_open not in fv_dates:
        report["hard_fail"] = True
        report["hard_fail_reason"] = (
            f"最新开市日 {latest_open} 的 factor_values 无数据（行情未拉取或因子计算失败），"
            f"当日预测不可产出"
        )

    # ---- 软警告 1：factor_values vs daily_kline 历史日期空洞 ----
    fv_holes = sorted(kline_dates - fv_dates)
    if fv_holes:
        report["soft_warnings"].append(
            f"factor_values 缺 {len(fv_holes)} 个历史交易日: {fv_holes[:5]}{'...' if len(fv_holes) > 5 else ''}"
        )
    report["tables"]["factor_values"] = {"missing_vs_kline": len(fv_holes)}

    # ---- 软警告 2：行数漂移（最新有数据日 vs 前 20 日中位数，±30%）----
    for table in ROW_DRIFT_TABLES:
        try:
            rows = con.execute(
                f"SELECT date, COUNT(*) n FROM {table} GROUP BY date ORDER BY date DESC LIMIT 21"
            ).fetchall()
        except Exception:
            continue
        if len(rows) < 2:
            continue
        latest_d, latest_n = str(rows[0][0])[:10], rows[0][1]
        hist = sorted(r[1] for r in rows[1:])
        hist.sort()
        med = hist[len(hist) // 2]
        drift = abs(latest_n - med) / med if med else 0
        info = {"latest": latest_d, "rows": latest_n, "median20": med, "drift": round(drift, 3)}
        report["tables"][table] = {**report["tables"].get(table, {}), **info}
        if drift > 0.30:
            report["soft_warnings"].append(
                f"{table} 最新日 {latest_d} 行数 {latest_n} 偏离前20日中位数 {med} 达 {drift:.0%}"
            )

    # ---- 软警告 3：最新交易日 adj_factor NULL 率 ----
    if latest_open:
        try:
            total, nulls = con.execute(
                "SELECT COUNT(*), COUNT(*) FILTER (WHERE adj_factor IS NULL) "
                "FROM daily_raw WHERE date = ?::DATE",
                [latest_open],
            ).fetchone()
            if total and nulls / total > 0.01:
                report["soft_warnings"].append(
                    f"daily_raw {latest_open} adj_factor NULL 率 {nulls}/{total} > 1%"
                )
        except Exception:
            pass

    # ---- 软警告 4：日历对账（各表自身范围内缺的开市日）----
    if cal_ok and latest_open:
        from data import calendar as cal
        for table in ["daily_raw", "daily_basic", "cyq_perf", "factor_values"]:
            tdates = set(_dates_in(con, table))
            if not tdates:
                continue
            lo, hi = min(tdates), max(tdates)
            expected = set(cal.trading_days(con, start=lo, end=hi))
            missing = sorted(expected - tdates)
            report["tables"].setdefault(table, {})["calendar_missing"] = len(missing)
            if missing:
                report["soft_warnings"].append(
                    f"{table} 在 {lo}~{hi} 范围内缺 {len(missing)} 个开市日: "
                    f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
                )

    return report


def write_report(report: dict):
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def check(con: duckdb.DuckDBPyConnection) -> dict:
    """运行校验、写报告、打日志。返回 report（调用方据 hard_fail 决定 exit code）。"""
    report = run_checks(con)
    write_report(report)
    for w in report["soft_warnings"]:
        log.warning("[integrity/soft] %s", w)
    if report["hard_fail"]:
        log.error("[integrity/HARD] %s", report["hard_fail_reason"])
    else:
        log.info("[integrity] OK (%d soft warnings) -> %s",
                 len(report["soft_warnings"]), REPORT_PATH.name)
    return report


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        report = check(con)
    finally:
        con.close()
    sys.exit(1 if report["hard_fail"] else 0)


if __name__ == "__main__":
    main()
