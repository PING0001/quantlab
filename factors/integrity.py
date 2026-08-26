# -*- coding: utf-8 -*-
"""
数据完整性校验：硬失败/软警告分级。

workbuddy 自动化"任何一步失败即停止"，因此分级至关重要：
- 硬失败（exit 1，阻断下游 generate_lgb）：最新开市日的池因子表无数据
  ——当日预测不可产出，继续没有意义；
- 软警告（exit 0，仅记录）：历史缺口、行数漂移、adj_factor 缺失、cyq pending、
  股票级覆盖缺口、因子列区间性高 NULL
  ——可见但不砍断当日产品。

日历对账依赖 trading_calendar（trade_cal 唯一真相）；日历不可用时该组检查
降级跳过并在报告中标注 degraded，绝不用业务表兜底（循环论证）。

2026-08-27 多池化重写：
- 因子表 SQL 全部走 factors/store（原版软警告 6 的 SQL 缺 f 前缀，
  `{fv_table}` 字面量触发 ParserException 被宽 except + log.debug 吞掉
  ——股票级覆盖检查在生产上长期静默失效，P0）。
- 逐检查异常显性化：任何检查自身抛错记入 report["check_errors"] 并
  log.warning，绝不再静默吞。
- 报告按池命名：data/integrity_report_{pool}.json（原 integrity_report.json
  无程序化消费方，安全改名）。
- 硬失败结构化字段 failed_side（"factor"/"market"）与 reason 文案共存；
  pull.py 靠 reason 字符串匹配分流的旧路径保持兼容（微盘池文案不变）。

用法：
    python -m factors.integrity     # 独立运行（默认池），exit 1 = 硬失败
    python -m factors.integrity --pool mainboard_all
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

from config import DB_PATH, ROOT
from pools.spec import get_pool, PoolSpec
from pools.membership import union_codes
from . import store

log = logging.getLogger(__name__)

# 报告写本仓 data/（ROOT 锚定而非 DB_PATH.parent：dev bench 经 QUANTLAB_DB
# 共享主仓 DB 时，报告不越界写进主仓；主仓默认路径不变）

# 行数漂移监控的表（trading 粒度）
ROW_DRIFT_TABLES = ["daily_raw", "daily_basic", "cyq_perf"]

# 因子列 NULL 率监控窗口（交易日）与告警阈值。阈值取 0.30 而非 0.50：
# 实证 alpha100 在近 250 交易日 NULL 率 33%（80/250 天全空）从未触发旧
# 50% 阈值，而健康列次高仅 ~9%（筹码列，cyq 起始 2018 属正常口径差异）
NULL_WINDOW_DAYS = 250
NULL_RATE_MAX = 0.30


def _report_path(spec: PoolSpec) -> Path:
    return ROOT / "data" / f"integrity_report_{spec.name}.json"


def _run_check(report: dict, name: str, fn) -> None:
    """逐检查异常显性化：抛错记 check_errors + log.warning，不吞不崩。"""
    try:
        fn()
    except Exception as e:  # noqa: BLE001 —— 检查自身失败必须可见
        report.setdefault("check_errors", []).append(
            {"check": name, "error": f"{type(e).__name__}: {e}"})
        log.warning("[integrity/check-error] %s: %s", name, e)


def _dates_in(con, table: str) -> list[str]:
    try:
        return [str(r[0])[:10] for r in
                con.execute(f"SELECT DISTINCT date FROM {table} ORDER BY date").fetchall()]
    except duckdb.Error:
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
    spec = get_pool(pool)
    fv_table = spec.factor_table
    report = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "pool": spec.name,
        "hard_fail": False,
        "hard_fail_reason": None,
        "failed_side": None,          # "factor" | "market" | None（结构化，与 reason 共存）
        "soft_warnings": [],
        "check_errors": [],           # 2026-08-27：检查自身异常显性化（禁吞）
        "degraded": False,
        "tables": {},
    }

    latest_open, cal_ok = _latest_open_day(con)
    report["latest_open_day"] = latest_open
    if not cal_ok:
        report["degraded"] = True
        report["soft_warnings"].append("trading_calendar 不可用，日历对账已跳过（降级）")

    fv_dates = set(store.dates(con, spec))
    kline_dates = set(_dates_in(con, "daily_kline"))

    # ---- 硬失败：最新开市日池因子表无数据 ----
    # 分时段判定（2026-08-24 审计 #12 精化）：当日行情已到位（kline 有行）
    # 而因子缺 = 因子侧问题，硬失败；行情缺失时——17:00 前属盘前正常态
    # （软警告），17:00 后属"晚间拉取真失败"（行情侧），恢复硬失败以阻断下游
    if latest_open and latest_open not in fv_dates:
        import datetime as _dt
        evening = _dt.datetime.now().hour >= 17
        if latest_open in kline_dates:
            report["hard_fail"] = True
            report["failed_side"] = "factor"
            report["hard_fail_reason"] = (
                f"最新开市日 {latest_open} 的 {fv_table} 无数据（行情已到位，"
                f"因子计算失败），当日预测不可产出"
            )
        elif evening:
            report["hard_fail"] = True
            report["failed_side"] = "market"
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
    def _row_drift():
        for table in ROW_DRIFT_TABLES:
            rows = con.execute(
                f"SELECT date, COUNT(*) n FROM {table} GROUP BY date ORDER BY date DESC LIMIT 21"
            ).fetchall()
            if len(rows) < 2:
                continue
            latest_d, latest_n = str(rows[0][0])[:10], rows[0][1]
            hist = sorted(r[1] for r in rows[1:])
            med = hist[len(hist) // 2]
            drift = abs(latest_n - med) / med if med else 0
            info = {"latest": latest_d, "rows": latest_n, "median20": med, "drift": round(drift, 3)}
            report["tables"][table] = {**report["tables"].get(table, {}), **info}
            if drift > 0.30:
                report["soft_warnings"].append(
                    f"{table} 最新日 {latest_d} 行数 {latest_n} 偏离前20日中位数 {med} 达 {drift:.0%}"
                )

    _run_check(report, "row_drift", _row_drift)

    # ---- 软警告 3：最新交易日 adj_factor NULL 率 ----
    def _adj_null():
        if not latest_open:
            return
        total, nulls = con.execute(
            "SELECT COUNT(*), COUNT(*) FILTER (WHERE adj_factor IS NULL) "
            "FROM daily_raw WHERE date = ?::DATE",
            [latest_open],
        ).fetchone()
        if total and nulls / total > 0.01:
            report["soft_warnings"].append(
                f"daily_raw {latest_open} adj_factor NULL 率 {nulls}/{total} > 1%"
            )

    _run_check(report, "adj_null", _adj_null)

    # ---- 软警告 4：日历对账（各表自身范围内缺的开市日）----
    def _calendar():
        if not (cal_ok and latest_open):
            return
        from data import trading_calendar as cal
        for table in ["daily_raw", "daily_basic", "cyq_perf", fv_table]:
            tdates = set(store.dates(con, spec) if table == fv_table
                         else _dates_in(con, table))
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

    _run_check(report, "calendar", _calendar)

    # ---- 软警告 5：因子值级健全性（教训：行数/日期全绿但值坏——
    #      2026-07-13~08-17 事故中 lookback 窗口损坏，长窗口因子全 NULL
    #      而当日行数 1108 一切正常）。检查最新开市日入模因子与长窗口
    #      代表因子的非空率 ----
    def _value_health():
        if not latest_open:
            return
        fv_cols = set(store.columns(con, spec))
        cols_to_check = []
        sel_path = Path(__file__).parent / f"selected_{spec.name}_20d.json"
        if sel_path.exists():
            sel = json.loads(sel_path.read_text(encoding="utf-8"))
            cols_to_check += [c for c in sel.get("selected_factors", [])
                              if c in fv_cols]
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

    _run_check(report, "value_health", _value_health)

    # ---- 软警告 6：股票级覆盖缺口（池代码 vs 池因子表历史覆盖）----
    #      日期级空洞检查（软警告 1）看不见池扩容缺口：新成员的历史日期在
    #      表内已有旧池股票的行。量化只数与行数，供 --backfill-stocks 决策。
    #      2026-08-27：走 store.stock_coverage（表范围限定口径），修复本检查
    #      因 SQL 缺 f 前缀长期静默失效的 P0。
    def _stock_gaps():
        pool_codes = union_codes(con=con, pool=spec.name)
        rows = store.stock_coverage(con, spec, pool_codes)
        miss_rows = sum(int(k - f) for _, k, f in rows)
        report["tables"].setdefault(fv_table, {})["stock_gaps"] = {
            "codes": len(rows), "pool": len(pool_codes), "missing_rows": miss_rows,
        }
        if rows:
            sample = ", ".join(str(c) for c, _, _ in rows[:5])
            report["soft_warnings"].append(
                f"{fv_table} 股票级缺口: {len(rows)}/{len(pool_codes)} 只池内代码"
                f"覆盖不足（<{store.STOCK_COVERAGE_MIN:.0%}，约缺 {miss_rows} 行），"
                f"示例 {sample}{'...' if len(rows) > 5 else ''}"
                f"（python -m factors.update --backfill-stocks 回补）"
            )

    _run_check(report, "stock_gaps", _stock_gaps)

    # ---- 软警告 7：因子列区间性高 NULL（alpha100 案例：1,331/4,528 天全空
    #      从未告警——运行期非确定性异常被吞成整列 NULL，值级检查只看最新日
    #      且只看入模因子，区间性坏死不可见）。近 NULL_WINDOW_DAYS 个交易日
    #      逐列 NULL 率超阈值即告警 ----
    def _null_rate():
        fv_cols = store.columns(con, spec)
        null_cols = [c for c in sorted(fv_cols)
                     if c not in ("code", "date") and not c.startswith("ai_")]
        if not null_cols:
            return
        cutoff = con.execute(
            f"SELECT MIN(date) FROM (SELECT DISTINCT date FROM {fv_table} "
            "ORDER BY date DESC LIMIT ?)", [NULL_WINDOW_DAYS]
        ).fetchone()[0]
        total = con.execute(
            f"SELECT COUNT(*) FROM {fv_table} WHERE date >= ?", [cutoff]
        ).fetchone()[0]
        if not total:
            return
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

    _run_check(report, "null_rate", _null_rate)

    return report


def write_report(report: dict, spec: PoolSpec) -> None:
    _report_path(spec).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def check(con: duckdb.DuckDBPyConnection, pool: str | None = None) -> dict:
    """运行校验、写报告、打日志。返回 report（调用方据 hard_fail 决定 exit code）。"""
    spec = get_pool(pool)
    report = run_checks(con, pool=spec.name)
    write_report(report, spec)
    for w in report["soft_warnings"]:
        log.warning("[integrity/soft] %s", w)
    if report["hard_fail"]:
        log.error("[integrity/HARD] %s", report["hard_fail_reason"])
    else:
        log.info("[integrity] OK (%d soft warnings) -> %s",
                 len(report["soft_warnings"]), _report_path(spec).name)
    return report


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    ap = argparse.ArgumentParser(description="数据完整性校验（默认池）")
    ap.add_argument("--pool", default=None,
                    help="目标池（默认 env QUANTLAB_POOL / 微盘）")
    args = ap.parse_args()

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        report = check(con, pool=args.pool)
    finally:
        con.close()
    sys.exit(1 if report["hard_fail"] else 0)


if __name__ == "__main__":
    main()
