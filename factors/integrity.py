# -*- coding: utf-8 -*-
"""
数据完整性校验：硬失败/软警告分级。

workbuddy 自动化"任何一步失败即停止"，因此分级至关重要：
- 硬失败（exit 1，阻断下游 generate_lgb）：最新开市日的 factor_values 无数据
  --当日预测不可产出，继续没有意义；
- 软警告（exit 0，仅记录）：历史缺口、行数漂移、adj_factor 缺失、cyq pending、
  股票级覆盖缺口、因子列区间性高 NULL
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
import os
import sys
from datetime import datetime
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_PATH, POOL_NAME, ROOT, get_factor_table
from pools.membership import union_codes

log = logging.getLogger(__name__)

# 报告写本仓 data/（ROOT 锚定而非 DB_PATH.parent：dev bench 经 QUANTLAB_DB
# 共享主仓 DB 时，报告不越界写进主仓；主仓默认路径不变）
REPORT_PATH = ROOT / "data" / "integrity_report.json"

# 行数漂移监控的表（trading 粒度）
ROW_DRIFT_TABLES = ["daily_raw", "daily_basic", "cyq_perf"]

# 因子列 NULL 率监控窗口（交易日）与告警阈值。阈值取 0.30 而非 0.50：
# 实证 alpha100 在近 250 交易日 NULL 率 33%（80/250 天全空）从未触发旧
# 50% 阈值，而健康列次高仅 ~9%（筹码列，cyq 起始 2018 属正常口径差异）
NULL_WINDOW_DAYS = 250
NULL_RATE_MAX = 0.30


def _dates_in(con, table: str) -> list[str]:
    try:
        return [str(r[0])[:10] for r in
                con.execute(f"SELECT DISTINCT date FROM {table} ORDER BY date").fetchall()]
    except Exception:
        return []


def _latest_open_day(con) -> tuple[str, bool]:
    """返回 (最新开市日, 日历是否可用)。日历缺失时降级用 daily_kline 最大日。

    日历表可含官方预公布的未来日期（日历因子需要），此处必须以今天为界，
    否则会把未来交易日当"当日因子缺失"误报硬失败。"""
    try:
        from data import trading_calendar as cal
        from datetime import date as _date
        d = cal.latest(con, on_or_before=_date.today().isoformat())
        if d:
            return d, True
    except Exception:
        pass
    d = con.execute("SELECT MAX(date) FROM daily_kline").fetchone()[0]
    return (str(d)[:10], False) if d else (None, False)


def run_checks(con: duckdb.DuckDBPyConnection, pool: str | None = None) -> dict:
    fv_table = get_factor_table(pool)
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

    fv_dates = set(_dates_in(con, fv_table))
    kline_dates = set(_dates_in(con, "daily_kline"))

    # ---- 硬失败：最新开市日 factor_values 无数据 ----
    # 分时段判定（2026-08-24 审计 #12 精化）：当日行情已到位（kline 有行）
    # 而因子缺 = 因子计算问题，硬失败；行情缺失时——17:00 前属盘前正常态
    # （软警告），17:00 后属"晚间拉取真失败"，恢复硬失败以阻断下游
    if latest_open and latest_open not in fv_dates:
        import datetime as _dt
        evening = _dt.datetime.now().hour >= 17
        if latest_open in kline_dates:
            report["hard_fail"] = True
            report["hard_fail_reason"] = (
                f"最新开市日 {latest_open} 的 {fv_table} 无数据（行情已到位，"
                f"因子计算失败），当日预测不可产出"
            )
        elif evening:
            report["hard_fail"] = True
            report["hard_fail_reason"] = (
                f"最新开市日 {latest_open} 行情在 17:00 后仍缺失——晚间拉取失败，"
                f"当日预测不可产出"
            )
        else:
            report["soft_warnings"].append(
                f"最新开市日 {latest_open} 行情尚未到位（盘前运行），"
                f"因子完整性硬失败暂缓")

    # ---- 软警告 1：池因子表自身范围内 vs daily_kline 的历史日期空洞 ----
    # 锚定表内最早日期：2020 起的新池表不把 2008~2019 计为缺失（表范围外
    # 不属于该池口径）。
    fv_lo = min(fv_dates) if fv_dates else None
    fv_holes = sorted(d for d in (kline_dates - fv_dates)
                      if fv_lo is None or d >= fv_lo)
    if fv_holes:
        report["soft_warnings"].append(
            f"{fv_table} 缺 {len(fv_holes)} 个历史交易日: {fv_holes[:5]}{'...' if len(fv_holes) > 5 else ''}"
        )
    report["tables"][fv_table] = {"missing_vs_kline": len(fv_holes)}

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
        from data import trading_calendar as cal
        for table in ["daily_raw", "daily_basic", "cyq_perf", fv_table]:
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

    # ---- 软警告 5：因子值级健全性（教训：行数/日期全绿但值坏——
    #      2026-07-13~08-17 事故中 lookback 窗口损坏，长窗口因子全 NULL
    #      而当日行数 1108 一切正常）。检查最新开市日入模因子与长窗口
    #      代表因子的非空率 ----
    if latest_open:
        fv_cols = {r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = ?", [fv_table]).fetchall()}
        cols_to_check = []
        try:
            sel_path = Path(__file__).parent / f"selected_{POOL_NAME}_20d.json"
            if sel_path.exists():
                sel = json.loads(sel_path.read_text(encoding="utf-8"))
                cols_to_check += [c for c in sel.get("selected_factors", [])
                                  if c in fv_cols]
        except Exception:
            pass
        cols_to_check += [c for c in ("Return_20d", "Reversal_60d", "GZ2000_return_20d")
                          if c in fv_cols]
        for col in dict.fromkeys(cols_to_check):
            nonnull, total = con.execute(
                f"SELECT COUNT({col}), COUNT(*) FROM {fv_table} WHERE date=?::DATE",
                [latest_open],
            ).fetchone()
            if total and nonnull / total < 0.5:
                report["soft_warnings"].append(
                    f"因子值级异常: {col} 在 {latest_open} 非空率仅 "
                    f"{nonnull}/{total}（<50%，疑似 lookback 窗口损坏或特征丢失）"
                )

    # ---- 软警告 6：股票级覆盖缺口（池代码 vs factor_values 历史覆盖）----
    #      日期级空洞检查（软警告 1）看不见池扩容缺口：新成员的历史日期在
    #      表内已有旧池股票的行。量化只数与行数，供 --backfill-stocks 决策。
    try:
        from .update import STOCK_COVERAGE_MIN

        # 池时点化（2026-08-25）：快照全历史成员并集，经调用方连接查询
        pool_codes = union_codes(con=con, pool=POOL_NAME)
        rows = con.execute(
            """
            WITH pool AS (SELECT DISTINCT unnest(?::VARCHAR[]) AS code),
            k AS (SELECT code, COUNT(*) AS n FROM daily_kline GROUP BY code),
            f AS (SELECT code, COUNT(*) AS n FROM {fv_table} GROUP BY code)
            SELECT p.code, COALESCE(k.n, 0), COALESCE(f.n, 0)
            FROM pool p
            LEFT JOIN k ON k.code = p.code
            LEFT JOIN f ON f.code = p.code
            WHERE COALESCE(k.n, 0) > 0
              AND COALESCE(f.n, 0) < COALESCE(k.n, 0) * ?
            ORDER BY p.code
            """,
            [pool_codes, STOCK_COVERAGE_MIN],
        ).fetchall()
        miss_rows = sum(int(k - f) for _, k, f in rows)
        report["tables"].setdefault(fv_table, {})["stock_gaps"] = {
            "codes": len(rows), "pool": len(pool_codes), "missing_rows": miss_rows,
        }
        if rows:
            sample = ", ".join(str(c) for c, _, _ in rows[:5])
            report["soft_warnings"].append(
                f"{fv_table} 股票级缺口: {len(rows)}/{len(pool_codes)} 只池内代码"
                f"覆盖不足（<{STOCK_COVERAGE_MIN:.0%}，约缺 {miss_rows} 行），"
                f"示例 {sample}{'...' if len(rows) > 5 else ''}"
                f"（python -m factors.update --backfill-stocks 回补）"
            )
    except Exception:
        log.debug("stock coverage check skipped", exc_info=True)

    # ---- 软警告 7：因子列区间性高 NULL（alpha100 案例：1,331/4,528 天全空
    #      从未告警——运行期非确定性异常被吞成整列 NULL，值级检查只看最新日
    #      且只看入模因子，区间性坏死不可见）。近 NULL_WINDOW_DAYS 个交易日
    #      逐列 NULL 率超阈值即告警 ----
    try:
        fv_cols = {r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = ?", [fv_table]).fetchall()}
        null_cols = [c for c in sorted(fv_cols)
                     if c not in ("code", "date") and not c.startswith("ai_")]
        if null_cols:
            cutoff = con.execute(
                f"SELECT MIN(date) FROM (SELECT DISTINCT date FROM {fv_table} "
                "ORDER BY date DESC LIMIT ?)", [NULL_WINDOW_DAYS]
            ).fetchone()[0]
            total = con.execute(
                f"SELECT COUNT(*) FROM {fv_table} WHERE date >= ?", [cutoff]
            ).fetchone()[0]
            if total:
                exprs = ", ".join(f"COUNT({c})" for c in null_cols)
                nonnulls = con.execute(
                    f"SELECT {exprs} FROM {fv_table} WHERE date >= ?", [cutoff]
                ).fetchone()
                high_null = [
                    {"column": c, "null_rate": round(1 - n / total, 3)}
                    for c, n in zip(null_cols, nonnulls) if (1 - n / total) > NULL_RATE_MAX
                ]
                if high_null:
                    report["tables"].setdefault(fv_table, {})[
                        "high_null_columns"] = high_null
                    detail = ", ".join(
                        f"{h['column']}({h['null_rate']:.0%})" for h in high_null[:10])
                    report["soft_warnings"].append(
                        f"因子列近 {NULL_WINDOW_DAYS} 交易日 NULL 率超 "
                        f"{NULL_RATE_MAX:.0%}: {detail}"
                        f"{'...' if len(high_null) > 10 else ''}（疑似计算异常被吞）"
                    )
    except Exception:
        log.debug("column null-rate check skipped", exc_info=True)

    return report


def write_report(report: dict):
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def check(con: duckdb.DuckDBPyConnection, pool: str | None = None) -> dict:
    """运行校验、写报告、打日志。返回 report（调用方据 hard_fail 决定 exit code）。"""
    report = run_checks(con, pool)
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
        report = check(con, pool=POOL_NAME)
    finally:
        con.close()
    sys.exit(1 if report["hard_fail"] else 0)


if __name__ == "__main__":
    main()
