# -*- coding: utf-8 -*-
"""
Factor computation pipeline.

Loads OHLCV + supplementary data from DuckDB, computes the non-alpha factor
panel (~64 columns；alpha101 全系已于 2026-08-22 移除，经典表达式仅存于
factors/baseline_alphas.py 作评估器回归基准), stores results to the
`factor_values` table.

Usage:
    python -m factors.compute                  # full rebuild
    python -m factors.compute --from 2024-01-01  # from specific date
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_PATH, get_pool_codes

from .extra_factors import compute_non_alpha_factors

log = logging.getLogger(__name__)


# ---- Data Loading ----

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
    """Load total_mv, circ_mv from daily_basic (code without suffix)."""
    pure_codes = [c.replace(".SH", "").replace(".SZ", "") for c in codes]
    code_map = dict(zip(pure_codes, codes))

    placeholders = ",".join(["?"] * len(pure_codes))
    df = con.execute(
        f"SELECT code, date, total_mv, circ_mv "
        f"FROM daily_basic WHERE code IN ({placeholders}) "
        f"ORDER BY code, date",
        pure_codes,
    ).fetchdf()
    if df.empty:
        return pl.DataFrame()

    df["date"] = df["date"].astype(str)
    df["code"] = df["code"].map(code_map)

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

    GZ2000 九列的计算公式系按列名语义重建，并已用 2026-06 历史存量值
    回归验证（全部 0.00% 偏差）——原始实现代码从未入 git，2026-07 因子
    污染事故溯源时发现缺失。注意：GZ2000 feats 必须真正 join 进返回值
    （历史上构建后被丢弃，导致入模的 GZ2000_* 因子全 NULL）。
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
            h, l, c = pl.col("high"), pl.col("low"), pl.col("close")
            pc = c.shift(1)
            tr = pl.max_horizontal(h, pc) - pl.min_horizontal(l, pc)
            feats = pl_df.select([
                pl.col("datetime"),
                (c / c.shift(1) - 1).alias("GZ2000_return_1d"),
                (c / c.shift(5) - 1).alias("GZ2000_return_5d"),
                (c / c.shift(20) - 1).alias("GZ2000_return_20d"),
                ((c / c.shift(1) - 1).rolling_std(10)).alias("GZ2000_vol_10d"),
                ((c / c.shift(1) - 1).rolling_std(60)).alias("GZ2000_vol_60d"),
                (-(c / c.shift(60) - 1)).alias("GZ2000_reversal_60d"),
                ((c - c.rolling_min(252))
                 / (c.rolling_max(252) - c.rolling_min(252))).alias("GZ2000_pricepos_252d"),
                tr.rolling_mean(14).alias("GZ2000_atr_14d"),
                (4 * c.rolling_std(20) / c.rolling_mean(20)).alias("GZ2000_boll_width"),
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


def _load_shibor(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """Load SHIBOR daily rates (on, 1m) for macro feature."""
    df = con.execute(
        "SELECT date, shibor_on, shibor_1m FROM macro_daily ORDER BY date"
    ).fetchdf()
    if df.empty:
        return pl.DataFrame()
    df["date"] = df["date"].astype(str)
    return pl.from_pandas(df).rename({"date": "datetime"})


def _load_days_to_next_trading(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """DaysToNextTrading：今天到下一交易日之间的休市天数（明天开市=0；普通周五=2；
    节前最后交易日=假期长度）。trading_calendar 唯一真相源（is_open 序列相邻差-1）；
    日历末位交易日无下一日 → 该日 NULL（诚实留空，待日历延展后自然补上）。"""
    import pandas as pd

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
    日历末月若交割日超出日历范围 → 这些尾部日期无映射留 NULL。"""
    import bisect
    from datetime import timedelta

    import pandas as pd

    from .extra_factors import third_friday

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
    做笛卡尔积后按区间过滤。替代原先的 Python 双层循环（~4600 日 × ST 行，
    是重复行事故的根因兼性能瓶颈）。

    区间终止语义（不再用 9999-12-31 兜底 NULL end_date）：
    - end_date 为 NULL 时，截断到该股下一条 namechange 记录的 start_date
      （LEAD 语义，exclusive——换名生效日即不再处于旧 ST 名称）；无下一条
      记录时才延伸到样本末端。
    - 撤销类记录（撤销ST/撤销*ST/摘星/摘帽）作为当前 ST 区间的终止信号
      （取区间内最早者），其自身不开启新 ST 区间。注意 '撤消*ST并实行ST'
      不属于撤销类（摘星后仍为 ST）。
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
        WHERE change_reason IN ('ST', '*ST')
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


# ---- Main Pipeline ----

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
        # CostPosition: (close - cost_50pct) / (cost_95pct - cost_5pct)
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

    shibor_df = _load_shibor(con)
    if not shibor_df.is_empty():
        dates = extra_df.select("datetime").unique()
        shibor_df = dates.join(shibor_df, on="datetime", how="left")
        symbols = extra_df.select("vt_symbol").unique()
        shibor_df = symbols.join(shibor_df, how="cross")
        extra_df = extra_df.join(shibor_df, on=["datetime", "vt_symbol"], how="left")

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


# ---- Storage ----

AI_FACTOR_COLUMNS = ["ai_gz2000_20d", "ai_gz2000_median_5d"]


def store_factor_values(con: duckdb.DuckDBPyConnection, panel: pl.DataFrame):
    """Store factor panel into DuckDB factor_values table (full rebuild).

    factor_values 是列所有权分离的双写入方表：本函数拥有全部非 alpha
    因子列（alpha101 已移除），build_ai_factor.py 拥有 ai_gz2000_* 两列。
    重建时必须：
      1. 保留 ai 列结构并回填其数据（panel 不计算 ai 因子）；
      2. 重建 PRIMARY KEY (code, date)（旧实现 CREATE TABLE AS 会丢掉
         约束与 ai 列，属 schema 回归）。
    """
    if panel.is_empty():
        log.warning("Empty panel, nothing to store.")
        return

    # Deduplicate on (code, date)
    panel = panel.unique(subset=["code", "date"], keep="last")

    # 备份 ai 因子列（若旧表存在且有数据）
    old_cols = {r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name='factor_values'"
    ).fetchall()}
    has_ai = all(c in old_cols for c in AI_FACTOR_COLUMNS)
    if has_ai:
        con.execute(
            "CREATE OR REPLACE TEMP TABLE _ai_keep AS "
            "SELECT code, date, ai_gz2000_20d, ai_gz2000_median_5d "
            "FROM factor_values WHERE ai_gz2000_20d IS NOT NULL "
            "OR ai_gz2000_median_5d IS NOT NULL"
        )
        n_keep = con.execute("SELECT COUNT(*) FROM _ai_keep").fetchone()[0]
    else:
        n_keep = 0

    con.execute("DROP TABLE IF EXISTS factor_values")

    # Build table from pandas (date 列保持 VARCHAR 'YYYY-MM-DD' 约定)
    pandas_df = panel.to_pandas()
    pandas_df = pandas_df.sort_values(["date", "code"])
    con.execute("CREATE TABLE factor_values AS SELECT * FROM pandas_df")

    # 恢复 ai 列结构 + 主键
    for col in AI_FACTOR_COLUMNS:
        con.execute(f"ALTER TABLE factor_values ADD COLUMN IF NOT EXISTS {col} DOUBLE")
    con.execute("ALTER TABLE factor_values ADD PRIMARY KEY (code, date)")

    # 回填 ai 数据
    if has_ai and n_keep > 0:
        con.execute("""
            UPDATE factor_values f SET
                ai_gz2000_20d = k.ai_gz2000_20d,
                ai_gz2000_median_5d = k.ai_gz2000_median_5d
            FROM _ai_keep k
            WHERE f.code = k.code AND f.date = k.date
        """)
        n_restored = con.execute(
            "SELECT COUNT(*) FROM factor_values WHERE ai_gz2000_20d IS NOT NULL"
        ).fetchone()[0]
        log.info("ai factor columns restored: %d/%d rows", n_restored, n_keep)

    con.execute("CHECKPOINT")
    log.info("factor_values table created with %d rows, %d columns (PK on code,date)",
             len(pandas_df), len(pandas_df.columns) + len(AI_FACTOR_COLUMNS))


# ---- Main Entry ----

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="Compute factor panel")
    parser.add_argument("--from", dest="start_date", default=None,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="end_date", default=None,
                        help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    codes = get_pool_codes()
    log.info("Pool: %d stocks", len(codes))

    con = duckdb.connect(str(DB_PATH))
    con.execute("SET threads = 4")

    panel = compute_panel(con, codes, start_date=args.start_date,
                          end_date=args.end_date)

    if not panel.is_empty():
        store_factor_values(con, panel)

    con.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
