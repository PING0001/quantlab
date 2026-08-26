# -*- coding: utf-8 -*-
"""
Factor pipeline: incremental update（日常，cron 路径）+ full rebuild（--full）。

2026-08-25 简化合并：原 factors/compute.py（全量构建）并入本文件。
2026-08-27 多池化重写：编排/IO 分离——池因子表 SQL 全部下沉 factors/store.py
（铁律：表 SQL 只在 store），本文件只保留面板计算内核（compute_panel 及其
数据装载，读的是源表 daily_kline/daily_basic/cyq_perf/...，不碰因子表）
与增量编排；池身份经 pools.spec。

增量模式（默认）：
    python -m factors.update                     # 日期级增量（默认池=env/微盘）
    python -m factors.update --pool mainboard_all
    python -m factors.update --dry-run           # 预览目标日期 + 股票级回补清单
    python -m factors.update --backfill-stocks   # 额外执行股票级历史回补（大计算）

对账驱动：不再只比较 MAX(date)，而是对账 daily_kline 与池因子表的
日期集合--历史空洞（某天因子算到一半失败、人为删除）也会被找出并回补。
股票级对账：池内代码在因子表缺失或历史覆盖显著偏低（相对其在
daily_kline 的应有交易日数）的纳入回补清单--池扩容后新成员的历史缺口
对日期级对账不可见。默认只告警，需显式 --backfill-stocks 才执行回补写入。

全量重建模式（勿轻易运行，整表 DROP 重建）：
    python -m factors.update --full              # 全历史重建
    python -m factors.update --full --from 2024-01-01 --to 2025-06-30
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_PATH
from pools.spec import get_pool

from .extra_factors import compute_non_alpha_factors, third_friday
from . import store
from . import integrity

log = logging.getLogger(__name__)


# ============================================================================
# 数据加载（DuckDB -> Polars 宽表）
# ============================================================================

def _load_ohlcv(con: duckdb.DuckDBPyConnection, codes: list[str]) -> pl.DataFrame:
    """Load daily kline data for given codes, compute VWAP."""
    placeholders = ",".join(["?"] * len(codes))
    df = con.execute(
        f"SELECT code, date, open, high, low, close, volume, amount "
        f"FROM daily_kline WHERE code IN ({placeholders}) "
        f"ORDER BY code, date",
        codes,
    ).fetchdf()
    if df.empty:
        return pl.DataFrame()
    df["date"] = df["date"].astype(str)

    result = pl.from_pandas(df)
    result = result.rename({"code": "vt_symbol", "date": "datetime"})

    # VWAP approximation
    result = result.with_columns(
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("vwap")
    )

    return result


def _load_market_cap(con: duckdb.DuckDBPyConnection, codes: list[str]) -> pl.DataFrame:
    """Load total_mv, circ_mv from daily_basic."""
    placeholders = ",".join(["?"] * len(codes))
    df = con.execute(
        f"SELECT code, date, total_mv, circ_mv "
        f"FROM daily_basic WHERE code IN ({placeholders}) "
        f"ORDER BY code, date",
        codes,
    ).fetchdf()
    if df.empty:
        return pl.DataFrame()

    df["date"] = df["date"].astype(str)

    result = pl.from_pandas(df)
    result = result.rename({"code": "vt_symbol", "date": "datetime"})

    result = result.with_columns([
        pl.col("total_mv").cast(pl.Float64),
        pl.col("circ_mv").cast(pl.Float64),
    ])

    return result


def _load_cyq(con: duckdb.DuckDBPyConnection, codes: list[str]) -> pl.DataFrame:
    """Load chip distribution raw data."""
    placeholders = ",".join(["?"] * len(codes))
    df = con.execute(
        f"SELECT code, date, winner_rate, cost_5pct, cost_15pct, cost_50pct, "
        f"cost_85pct, cost_95pct, weight_avg "
        f"FROM cyq_perf WHERE code IN ({placeholders}) "
        f"ORDER BY code, date",
        codes,
    ).fetchdf()
    if df.empty:
        return pl.DataFrame()

    df["date"] = df["date"].astype(str)
    result = pl.from_pandas(df)
    result = result.rename({"code": "vt_symbol", "date": "datetime"})
    return result


def _load_index_data(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """Load CSI (000985) / HS300 (000300) / GZ2000 (399303) market state features.

    GZ2000 列的计算公式系按列名语义重建，并已用 2026-06 历史存量值
    回归验证（全部 0.00% 偏差）--原始实现代码从未入 git，2026-07 因子
    污染事故溯源时发现缺失。注意：GZ2000 feats 必须真正 join 进返回值
    （历史上构建后被丢弃，导致入模的 GZ2000_* 因子全 NULL）。
    2026-08-27 死列清理：GZ2000 九列只留 return_5d/return_20d（其余 7 列
    零引用，连同表列一并删除；return_5d 是 gb/nn 活输入，return_20d 被
    integrity 值级检查引用）。
    """
    df = con.execute(
        "SELECT code, date, high, low, close FROM index_daily "
        "WHERE code IN ('000985', '000300', '399303') ORDER BY code, date"
    ).fetchdf()
    if df.empty:
        return pl.DataFrame()
    df["date"] = df["date"].astype(str)

    r = {}
    for code, prefix in [("000985", "CSI"), ("000300", "HS300"), ("399303", "GZ2000")]:
        part = df[df["code"] == code][["date", "high", "low", "close"]].copy()
        if part.empty:
            continue
        pl_df = pl.from_pandas(part).rename({"date": "datetime"})
        if prefix == "GZ2000":
            c = pl.col("close")
            feats = pl_df.select([
                pl.col("datetime"),
                (c / c.shift(5) - 1).alias("GZ2000_return_5d"),
                (c / c.shift(20) - 1).alias("GZ2000_return_20d"),
            ])
            r[prefix] = feats
            continue
        feats = pl_df.select([
            pl.col("datetime"),
            (pl.col("close") / pl.col("close").shift(1) - 1).alias(f"{prefix}_return_1d"),
            (pl.col("close") / pl.col("close").shift(20) - 1).alias(f"{prefix}_return_20d"),
        ])
        if prefix == "CSI":
            vol = pl_df.select([
                pl.col("datetime"),
                (pl.col("close") / pl.col("close").shift(20) - 1)
            ]).select([
                pl.col("datetime"),
                pl.col("close").rolling_std(20, min_samples=1).alias("CSI_volatility_20d")
            ])
            feats = feats.join(vol, on="datetime", how="left")
        r[prefix] = feats

    market = r.get("CSI", pl.DataFrame())
    if "HS300" in r:
        market = market.join(r["HS300"], on="datetime", how="left") if not market.is_empty() else r["HS300"]
    if "GZ2000" in r:
        market = market.join(r["GZ2000"], on="datetime", how="left") if not market.is_empty() else r["GZ2000"]
    return market


def _load_days_to_next_trading(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """DaysToNextTrading：今天到下一交易日之间的休市天数（明天开市=0；普通周五=2；
    节前最后交易日=假期长度）。trading_calendar 唯一真相源（is_open 序列相邻差-1）；
    日历末位交易日无下一日 -> 该日 NULL（诚实留空，待日历延展后自然补上）。"""
    df = con.execute(
        "SELECT date FROM trading_calendar WHERE is_open ORDER BY date"
    ).fetchdf()
    if len(df) < 2:
        return pl.DataFrame()
    dts = pd.to_datetime(df["date"]).dt.date.tolist()
    rows = [(str(dts[i]), (dts[i + 1] - dts[i]).days - 1) for i in range(len(dts) - 1)]
    return pl.DataFrame(
        {"datetime": [r[0] for r in rows], "DaysToNextTrading": [r[1] for r in rows]},
        schema={"datetime": pl.Utf8, "DaysToNextTrading": pl.Int64},
    )


def _load_days_to_delivery(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """DaysToDelivery：距下一股指期货交割日的自然日数，交割日当天=0。
    交割日 = 每月第三个周五，逢法定假日顺延至下一交易日（CFFEX 规则；
    2008-2026 共 9 个月顺延，均为春节/中秋撞期）。trading_calendar 单源；
    日历末月若交割日超出日历范围 -> 这些尾部日期无映射留 NULL。"""
    import bisect
    from datetime import timedelta

    df = con.execute(
        "SELECT date FROM trading_calendar WHERE is_open ORDER BY date"
    ).fetchdf()
    if df.empty:
        return pl.DataFrame()
    dts = pd.to_datetime(df["date"]).dt.date.tolist()
    cal = set(dts)

    deliveries = []
    for y in range(dts[0].year, dts[-1].year + 1):
        for m in range(1, 13):
            d = third_friday(y, m)
            while d not in cal and d <= dts[-1]:
                d += timedelta(days=1)
            if d in cal:
                deliveries.append(d)
    deliveries.sort()

    rows = []
    for d in dts:
        i = bisect.bisect_left(deliveries, d)
        if i < len(deliveries):
            rows.append((str(d), (deliveries[i] - d).days))
    return pl.DataFrame(
        {"datetime": [r[0] for r in rows], "DaysToDelivery": [r[1] for r in rows]},
        schema={"datetime": pl.Utf8, "DaysToDelivery": pl.Int64},
    )


def _load_stock_info(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """Load list_date for LnAge computation."""
    df = con.execute(
        "SELECT code, strftime(list_date, '%Y-%m-%d') AS list_date FROM stock_info"
    ).fetchdf()
    if df.empty:
        return pl.DataFrame()
    result = pl.from_pandas(df)
    result = result.rename({"code": "vt_symbol"})
    return result


def _compute_isst(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """Compute IsST factor from namechange table (vectorized).

    namechange 中每股一条 ST/*ST 区间记录（可能有重叠区间），与全部交易日
    做笛卡尔积后按区间过滤。

    区间终止语义（不再用 9999-12-31 兜底 NULL end_date）：
    - end_date 为 NULL 时，截断到该股下一条 namechange 记录的 start_date
      （LEAD 语义，exclusive--换名生效日即不再处于旧 ST 名称）；无下一条
      记录时才延伸到样本末端。
    - 撤销类记录（撤销ST/撤销*ST/摘星/摘帽）作为当前 ST 区间的终止信号
      （取区间内最早者），其自身不开启新 ST 区间。注意 '撤消*ST并实行ST'
      不属于撤销类（摘星后仍为 ST）。

    2026-08-24 修复（变级记录漏开区间）：'从ST变为*ST'/'从*ST变为ST' 等
    变级记录此前既不匹配 WHERE change_reason IN ('ST','*ST')（不开新
    区间），又通过 LEAD 把旧区间截断在变级日--ST 从未中断的股票在变级
    日当天被误判为非 ST（实测 2026-08-21 名带 ST 的 89 只中 12 只漏判，
    *ST萃华 戴帽 7 个月 IsST=0 进了报告榜首）。开区间判定改为"reason 含
    'ST' 且非撤销/摘星/摘帽前缀"。

    2026-08-26 修复（撤销叠加边界，全库 1 例 000615）：'撤销叠加*ST'
    语义为撤销退市风险叠加、仍为 *ST（名字印证），但 '撤销%' 前缀匹配
    把它误当终止信号 -> 该股此后 IsST=0。从终止信号中显式排除
    '撤销叠加%'（'叠加*ST' 开区间侧已被 '%ST%' 正确命中）。
    """
    df = con.execute("""
        WITH ordered AS (
            SELECT code, change_reason, start_date, end_date,
                   LEAD(start_date) OVER (
                       PARTITION BY code ORDER BY start_date, end_date NULLS LAST
                   ) AS next_start
            FROM namechange
        )
        SELECT code, start_date,
               LEAST(
                   COALESCE(end_date + 1, next_start, DATE '9999-12-31'),
                   COALESCE((
                       SELECT MIN(r.start_date) FROM namechange r
                       WHERE r.code = ordered.code
                         AND (r.change_reason IN ('撤销ST', '撤销*ST')
                              OR r.change_reason LIKE '摘星%'
                              OR r.change_reason LIKE '摘帽%')
                         AND r.start_date >= ordered.start_date
                   ), DATE '9999-12-31')
               ) AS end_excl
        FROM ordered
        WHERE change_reason LIKE '%ST%'
          AND NOT ((change_reason LIKE '撤销%'
                    OR change_reason LIKE '摘星%'
                    OR change_reason LIKE '摘帽%')
                   AND change_reason NOT LIKE '撤销叠加%')
        ORDER BY code, start_date
    """).fetchdf()
    if df.empty:
        return pl.DataFrame()

    date_range = con.execute("SELECT DISTINCT date FROM daily_kline ORDER BY date").fetchdf()
    if date_range.empty:
        return pl.DataFrame()

    nc = pl.from_pandas(df).with_columns([
        # 先 cast(Date) 再 cast(Utf8)，得到 'YYYY-MM-DD'（与 pandas
        # astype(str) 及下游 extra_df 的 datetime 格式一致；直接对
        # datetime64 cast(Utf8) 会带 ' 00:00:00.000000' 后缀导致 join 失配）
        pl.col("start_date").cast(pl.Date).cast(pl.Utf8),
        # end_excl 为排他上界：date < end_excl 等价于含端点的闭区间终点
        pl.col("end_excl").cast(pl.Date).cast(pl.Utf8),
    ])
    dates = pl.from_pandas(date_range).with_columns(
        pl.col("date").cast(pl.Date).cast(pl.Utf8)
    )

    result = (
        nc.join(dates, how="cross")
        .filter(
            (pl.col("date") >= pl.col("start_date"))
            & (pl.col("date") < pl.col("end_excl"))
        )
        .select([
            pl.col("code").alias("vt_symbol"),
            pl.col("date").alias("datetime"),
            pl.lit(1, dtype=pl.Int32).alias("IsST"),
        ])
        .unique(subset=["vt_symbol", "datetime"], keep="first")
    )
    return result


# ============================================================================
# 面板计算
# ============================================================================

def compute_panel(
    con: duckdb.DuckDBPyConnection,
    codes: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """
    Compute all factors for the given stock codes and date range.

    Returns a wide-format Polars DataFrame with columns:
    [code, date, non_alpha_factors..., IsST, ...]
    """
    t0 = time.time()

    # 1. Load data
    log.info("Loading OHLCV data ...")
    df = _load_ohlcv(con, codes)
    if df.is_empty():
        log.warning("No OHLCV data loaded.")
        return pl.DataFrame()

    log.info("Loading market cap data ...")
    mktcap = _load_market_cap(con, codes)
    if not mktcap.is_empty():
        df = df.join(mktcap, on=["datetime", "vt_symbol"], how="left")

    log.info("Loading stock info ...")
    info = _load_stock_info(con)
    if not info.is_empty():
        df = df.join(info, on="vt_symbol", how="left")

    # Filter date range
    if start_date:
        df = df.filter(pl.col("datetime") >= start_date)
    if end_date:
        df = df.filter(pl.col("datetime") <= end_date)

    # Sort for rolling-window operators
    df = df.sort(["vt_symbol", "datetime"])

    n_stocks = df["vt_symbol"].n_unique()
    n_dates = df["datetime"].n_unique()
    log.info("Data loaded: %d stocks × %d dates = %d rows", n_stocks, n_dates, len(df))

    # 2. Compute non-alpha factors
    log.info("Computing non-alpha factors ...")
    extra_df = compute_non_alpha_factors(df)

    # 7. Merge supplementary factors (chip, market state, IsST)
    log.info("Loading supplementary factors ...")

    cyq_df = _load_cyq(con, codes)
    if not cyq_df.is_empty():
        # Derive chip factors from raw cyq columns
        # WinnerRate: directly from cyq_perf
        cyq_df = cyq_df.with_columns(
            pl.col("winner_rate").alias("WinnerRate")
        )
        # CostPosition: (cost_50pct - weight_avg) / (cost_95pct - cost_5pct)
        cyq_df = cyq_df.with_columns(
            ((pl.col("cost_50pct") - pl.col("weight_avg")) /
             (pl.col("cost_95pct") - pl.col("cost_5pct") + 1e-10)).alias("CostPosition")
        )
        # ChipDispersion: (cost_85pct - cost_15pct) / cost_50pct
        cyq_df = cyq_df.with_columns(
            ((pl.col("cost_85pct") - pl.col("cost_15pct")) /
             (pl.col("cost_50pct") + 1e-10)).alias("ChipDispersion")
        )
        # ChipSkew: (cost_50pct - weight_avg) / (cost_85pct - cost_15pct)
        cyq_df = cyq_df.with_columns(
            ((pl.col("cost_50pct") - pl.col("weight_avg")) /
             (pl.col("cost_85pct") - pl.col("cost_15pct") + 1e-10)).alias("ChipSkew")
        )
        chip_cols = ["datetime", "vt_symbol", "WinnerRate", "CostPosition",
                     "ChipDispersion", "ChipSkew"]
        cyq_df = cyq_df.select(chip_cols)
        extra_df = extra_df.join(cyq_df, on=["datetime", "vt_symbol"], how="left")

    market_df = _load_index_data(con)
    if not market_df.is_empty():
        # Broadcast: add vt_symbol to market, then join
        symbols = extra_df.select("vt_symbol").unique()
        market_df = symbols.join(market_df, how="cross")
        extra_df = extra_df.join(market_df, on=["datetime", "vt_symbol"], how="left")

    gap_df = _load_days_to_next_trading(con)
    if not gap_df.is_empty():
        dates = extra_df.select("datetime").unique()
        gap_df = dates.join(gap_df, on="datetime", how="left")
        symbols = extra_df.select("vt_symbol").unique()
        gap_df = symbols.join(gap_df, how="cross")
        extra_df = extra_df.join(gap_df, on=["datetime", "vt_symbol"], how="left")

    deliv_df = _load_days_to_delivery(con)
    if not deliv_df.is_empty():
        dates = extra_df.select("datetime").unique()
        deliv_df = dates.join(deliv_df, on="datetime", how="left")
        symbols = extra_df.select("vt_symbol").unique()
        deliv_df = symbols.join(deliv_df, how="cross")
        extra_df = extra_df.join(deliv_df, on=["datetime", "vt_symbol"], how="left")

    isst_df = _compute_isst(con)
    if not isst_df.is_empty():
        extra_df = extra_df.join(isst_df, on=["datetime", "vt_symbol"], how="left")
        extra_df = extra_df.with_columns(pl.col("IsST").fill_null(0).cast(pl.Int32))
    else:
        extra_df = extra_df.with_columns(pl.lit(0).cast(pl.Int32).alias("IsST"))

    # 3. Final cleanup: clip extreme values
    # Exclude intermediate columns (any with _ prefix)
    factor_columns = [c for c in extra_df.columns
                      if c not in ("datetime", "vt_symbol")
                      and not c.startswith("_")]
    merged = extra_df
    for col in factor_columns:
        if col in merged.columns:
            merged = merged.with_columns(
                pl.when(
                    pl.col(col).is_infinite() | pl.col(col).is_nan()
                ).then(None).otherwise(pl.col(col)).alias(col)
            )

    # Drop intermediate columns
    id_cols = ["datetime", "vt_symbol"]
    merged = merged.select(id_cols + factor_columns)

    # Convert to wide format: (code, date, factors...)
    merged = merged.rename({"vt_symbol": "code", "datetime": "date"})

    elapsed = time.time() - t0
    n_factors = len(factor_columns)
    log.info("Total: %d factors, %d rows, %.1f seconds", n_factors, len(merged), elapsed)

    return merged


def run_stock_backfill(con: duckdb.DuckDBPyConnection, spec, codes: list[str]):
    """股票级历史回补：复用 compute_panel 全历史重算 + 列所有权写入。"""
    log.info("Backfilling %d codes (full history via compute_panel) ...", len(codes))
    panel = compute_panel(con, codes)
    if panel.is_empty():
        log.warning("Stock backfill: empty panel, nothing written.")
        return
    panel = panel.unique(subset=["code", "date"], keep="last")
    pdf = panel.to_pandas()
    pdf = pdf.sort_values(["date", "code"])
    n_ins, n_upd = store.upsert_panel(con, spec, pdf)
    con.execute("CHECKPOINT")
    log.info("Stock backfill written: %d inserted, %d updated", n_ins, n_upd)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Factor pipeline: incremental update (default) or full rebuild (--full)")
    parser.add_argument("--pool", default=None, choices=None,
                        help="目标池（默认 env QUANTLAB_POOL / 微盘；cron 无参=微盘契约）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只预览目标日期与股票级回补清单，不计算不写库")
    parser.add_argument("--backfill-stocks", action="store_true",
                        help="对股票级对账发现的覆盖不足代码执行全历史回补写入"
                             "（大计算，默认关闭，仅告警）")
    parser.add_argument("--full", action="store_true",
                        help="全量重建池因子表（整表 DROP 重建，保全 gb_/nn_ "
                             "等他方列；原 factors/compute.py 入口）")
    parser.add_argument("--from", dest="start_date", default=None,
                        help="全量模式起始日期（YYYY-MM-DD）")
    parser.add_argument("--to", dest="end_date", default=None,
                        help="全量模式结束日期（YYYY-MM-DD）")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    from pools.membership import latest_codes, union_codes

    spec = get_pool(args.pool)
    pool = spec.name
    log.info("Pool: %s -> factor table: %s", pool, spec.factor_table)

    if args.full:
        # ---- 全量重建：池 = 时点快照全历史成员并集 ----
        con = duckdb.connect(str(DB_PATH))
        con.execute("SET threads = 4")
        try:
            codes = union_codes(con=con, pool=pool)
            log.info("Pool: %d stocks (snapshot union, all periods)", len(codes))
            panel = compute_panel(con, codes, start_date=args.start_date,
                                  end_date=args.end_date)
            if not panel.is_empty():
                store.rebuild_table(con, spec, panel)
        finally:
            con.close()
        log.info("Done.")
        return

    codes = sorted(latest_codes(pool=pool))   # 池时点化：日更只算最新档成员
    log.info("Pool: %d stocks (latest snapshot members)", len(codes))

    con = duckdb.connect(str(DB_PATH))
    con.execute("SET threads = 4")

    try:
        latest = store.latest_date(con, spec)
        if not latest:
            log.warning("factor table %s is empty. Run with --full first.", spec.factor_table)
            return

        kline_max = con.execute("SELECT max(date) FROM daily_kline").fetchone()[0]
        kline_max = str(kline_max)[:10]
        if not kline_max:
            log.info("No kline data available.")
            return

        # ---- 对账：新增日期 + 历史空洞 ----
        missing = store.missing_dates(con, spec)
        new_dates = sorted(d for d in
                           [str(r[0])[:10] for r in con.execute(
                               "SELECT DISTINCT date FROM daily_kline WHERE date > ?",
                               [latest]).fetchall()]
                           if d <= kline_max)
        target_dates = sorted(set(missing) | set(new_dates))

        # ---- 对账：股票级覆盖（池扩容后新成员历史缺口，日期级对账盲区）----
        backfill = store.stock_coverage(con, spec, codes)
        if backfill:
            est_rows = sum(k - f for _, k, f in backfill)
            sample = ", ".join(c for c, _, _ in backfill[:5])
            log.warning(
                "Stock-level gaps: %d/%d pool codes under-covered in %s "
                "(<%.0f%% of daily_kline rows), ~%d rows to backfill. Sample: %s%s "
                "Run with --backfill-stocks to fix.",
                len(backfill), len(codes), spec.factor_table,
                store.STOCK_COVERAGE_MIN * 100, est_rows,
                sample, "..." if len(backfill) > 5 else "",
            )
        else:
            log.info("Stock-level coverage OK (all pool codes >= %.0f%%).",
                     store.STOCK_COVERAGE_MIN * 100)

        if not target_dates and not backfill:
            log.info("Factors are up to date (kline: %s).", kline_max)
            report = integrity.check(con, pool=pool)
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
            # [t0−252, tN]）。
            from_date = target_dates[0]
            lookback_start = store.get_lookback_start(con, from_date)
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

                    n_ins, n_upd = store.upsert_panel(con, spec, pdf)
                    con.execute("CHECKPOINT")
                    log.info("Written: %d inserted (new dates), %d updated (backfill)",
                             n_ins, n_upd)

        # ---- 股票级回补：需显式 --backfill-stocks，复用全历史计算路径 ----
        if backfill and args.backfill_stocks:
            run_stock_backfill(con, spec, [c for c, _, _ in backfill])
        elif backfill:
            log.info("Stock backfill skipped (%d codes, ~%d rows); "
                     "pass --backfill-stocks to execute.",
                     len(backfill), sum(k - f for _, k, f in backfill))

        # ---- 完整性校验（硬失败 exit 1 阻断下游；软警告仅记录）----
        report = integrity.check(con, pool=pool)
        if report["hard_fail"]:
            sys.exit(1)
    finally:
        con.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
