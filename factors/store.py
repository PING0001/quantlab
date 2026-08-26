# -*- coding: utf-8 -*-
"""因子表存储单点（2026-08-27 多池化重写）。

铁律：**池因子表的 SQL 只准出现在本文件**（读写/对账/列操作一律经此）。
任何模块不得再拼接 `FROM factor_values...` 字面量——非微盘 env 下那是
静默错宇宙事故（2026-08-27 审查 P1/P1'）。表名一律经 pools.spec.PoolSpec
解析，本文件不感知"哪个池是默认池"。

函数分四组：
    读      columns / dates / latest_date / missing_dates / stock_coverage /
            load_panel / load_isst
    写      rebuild_table / upsert_panel / ensure_column / update_columns /
            drop_column
    对账锚  get_lookback_start（daily_kline 侧，因子表无关但属增量管道域）
    审计    staleness_audit（Phase E）

写侧语义（从 factors.update 迁移，勿漂移）：
- factor 表是多写入方共享表：本管道（factors.update）公式因子列 +
  gb_/nn_ 构建脚本的模型因子列。列所有权：写 A 方的列不许被 B 方清空。
- upsert_panel：行级分流——缺失 (code,date) 行 INSERT（列子集，PK 兜底
  防重），已有行 UPDATE ... FROM（绝不整行替换）。
- rebuild_table：整表 DROP 重建，重建前保全旧表全部非面板列、重建后回填，
  并恢复 PRIMARY KEY (code, date)。
"""
from __future__ import annotations

import logging

import duckdb
import pandas as pd
import polars as pl

from pools.spec import PoolSpec

log = logging.getLogger(__name__)

LOOKBACK_DAYS = 260  # trading days (~1 year, covers 250d windows + margin)

# 股票级对账覆盖阈值：池代码在因子表的行数 < 其 daily_kline 行数 × 该值
# 即纳入回补清单（阈值不敏感；留 5% 容差吸收个别数据缺口）
STOCK_COVERAGE_MIN = 0.95


# ============================================================================
# 读
# ============================================================================

def columns(con: duckdb.DuckDBPyConnection, spec: PoolSpec) -> list[str]:
    """因子表全部列名（排序无关，information_schema 原序）。"""
    return [r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
        [spec.factor_table]).fetchall()]


def dates(con: duckdb.DuckDBPyConnection, spec: PoolSpec) -> list[str]:
    """因子表全部出现过的日期（升序，'YYYY-MM-DD'）。表不存在返回空。"""
    try:
        return [str(r[0])[:10] for r in con.execute(
            f"SELECT DISTINCT date FROM {spec.factor_table} ORDER BY date").fetchall()]
    except duckdb.Error:
        return []


def latest_date(con: duckdb.DuckDBPyConnection, spec: PoolSpec,
                codes: list[str] | None = None) -> str | None:
    """因子表当前最大日期（可选限定代码集）。空表/无表返回 None。"""
    where, params = "TRUE", []
    if codes is not None:
        where, params = f"code IN ({','.join(['?'] * len(codes))})", codes
    try:
        row = con.execute(
            f"SELECT max(date) FROM {spec.factor_table} WHERE {where}", params).fetchone()
        if row and row[0]:
            return str(row[0])[:10]
    except duckdb.Error:
        pass
    return None


def missing_dates(con: duckdb.DuckDBPyConnection, spec: PoolSpec) -> list[str]:
    """池因子表自身时间范围内的历史空洞（升序）。

    锚定表内最早日期：2020 起的新池表不会被 2008~2019 的 kline 日期
    误判为缺失（表范围外的日期不属于该池口径）。
    """
    rows = con.execute(f"""
        WITH scope AS (SELECT CAST(MIN(date) AS DATE) AS lo FROM {spec.factor_table})
        SELECT DISTINCT date FROM daily_kline, scope
        WHERE date >= scope.lo
        EXCEPT
        SELECT DISTINCT date FROM {spec.factor_table}
        ORDER BY date
    """).fetchall()
    return [str(r[0])[:10] for r in rows]


def stock_coverage(con: duckdb.DuckDBPyConnection, spec: PoolSpec,
                   codes: list[str]) -> list[tuple[str, int, int]]:
    """股票级对账：给定代码在池因子表覆盖不足的清单。

    返回 [(code, 应有行数, 实有行数)]（升序）。应有 = 该码在 daily_kline
    的行数（限定池表自身时间范围：2020 起的新池表不把 2008~2019 计入
    应有；停牌日本就无行），实有 < 应有 × STOCK_COVERAGE_MIN 即纳入。
    日期级对账看不见这类缺口：池扩容后新码的历史日期在因子表里已有
    旧池股票的行，"日期集合"判定无缺失。
    """
    rows = con.execute(
        f"""
        WITH pool AS (SELECT DISTINCT unnest(?::VARCHAR[]) AS code),
        k AS (SELECT code, COUNT(*) AS n FROM daily_kline
              WHERE date >= (SELECT CAST(MIN(date) AS DATE) FROM {spec.factor_table})
              GROUP BY code),
        f AS (SELECT code, COUNT(*) AS n FROM {spec.factor_table} GROUP BY code)
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


def load_panel(con: duckdb.DuckDBPyConnection, spec: PoolSpec,
               codes: list[str] | None = None,
               cols: list[str] | None = None,
               start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """读因子表面板（宽表）。

    返回 pandas DataFrame，列 = code, date + cols（cols=None 全部列）；
    date 保持 'YYYY-MM-DD' 字符串（消费方自行 pd.to_datetime）。
    行序 code, date。筛选条件全部可选；codes=None 不限代码集。
    """
    col_sql = ", ".join(["code", "date"] + list(cols)) if cols is not None else "*"
    where, params = ["TRUE"], []
    if codes is not None:
        where.append(f"code IN ({','.join(['?'] * len(codes))})")
        params += list(codes)
    if start is not None:
        where.append("date >= ?")
        params.append(str(start)[:10])
    if end is not None:
        where.append("date <= ?")
        params.append(str(end)[:10])
    return con.execute(
        f"SELECT {col_sql} FROM {spec.factor_table} "
        f"WHERE {' AND '.join(where)} ORDER BY code, date", params).fetchdf()


def load_isst(con: duckdb.DuckDBPyConnection, spec: PoolSpec,
              codes: list[str] | None = None,
              start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """读 IsST 因子列（时点 ST 判定，回测涨跌停幅度与候选过滤共用）。"""
    return load_panel(con, spec, codes=codes, cols=["IsST"], start=start, end=end)


# ============================================================================
# 写
# ============================================================================

def rebuild_table(con: duckdb.DuckDBPyConnection, spec: PoolSpec,
                  panel: pl.DataFrame) -> None:
    """全量重建因子表（--full 路径；整表 DROP 重建，保全他方列）。

    因子表系多写入方共享表：本模块调用方（factors.update）拥有公式因子列，
    gb_/nn_ 列归构建脚本所有。重建前保全旧表全部非面板列，重建后回填
    ——原实现只备份 ai 列，一次全量重建会静默清空模型因子列（2026-08-24
    拆弹）。重建时必须重建 PRIMARY KEY (code, date)（旧 CREATE TABLE AS
    会丢掉约束，属 schema 回归）。
    """
    table = spec.factor_table
    if panel.is_empty():
        log.warning("Empty panel, nothing to store.")
        return

    panel = panel.unique(subset=["code", "date"], keep="last")

    # 首次建表（新池）无旧表可保全；keep_schema 置空壳保持形状不变
    exists = con.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_schema = 'main' AND table_name = ?", [table]).fetchone()[0]
    if exists:
        con.execute(f"CREATE OR REPLACE TEMP TABLE _fv_keep AS SELECT * FROM {table}")
        keep_schema = con.execute("DESCRIBE _fv_keep").fetchdf()[["column_name", "column_type"]]
    else:
        keep_schema = pd.DataFrame(columns=["column_name", "column_type"])

    con.execute(f"DROP TABLE IF EXISTS {table}")

    # Build table from pandas (date 列保持 VARCHAR 'YYYY-MM-DD' 约定)
    pandas_df = panel.to_pandas()
    pandas_df = pandas_df.sort_values(["date", "code"])
    con.execute(f"CREATE TABLE {table} AS SELECT * FROM pandas_df")

    # 恢复非面板列结构 + 主键
    panel_cols = set(pandas_df.columns)
    kept = [(r.column_name, r.column_type) for r in keep_schema.itertuples()
            if r.column_name not in panel_cols and r.column_name not in ("code", "date")]
    for col, dtype in kept:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {dtype}")
    con.execute(f"ALTER TABLE {table} ADD PRIMARY KEY (code, date)")

    # 回填保全列数据（SET 目标列不可带表限定——DuckDB 解析器限制）
    if kept:
        sets = ", ".join(f"{c} = k.{c}" for c, _ in kept)
        con.execute(f"""
            UPDATE {table} f SET {sets}
            FROM _fv_keep k
            WHERE f.code = k.code AND f.date = k.date
        """)
        n_kept = con.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {kept[0][0]} IS NOT NULL"
        ).fetchone()[0]
        log.info("non-panel columns restored: %d 列, %d 行非空", len(kept), n_kept)

    con.execute("CHECKPOINT")
    log.info("%s table created with %d rows, %d columns (PK on code,date)",
             table, len(pandas_df), len(pandas_df.columns) + len(kept))


def upsert_panel(con: duckdb.DuckDBPyConnection, spec: PoolSpec,
                 pdf: pd.DataFrame) -> tuple[int, int]:
    """列所有权写入：缺失 (code,date) 行 INSERT，已有行 UPDATE FROM。

    pdf 需含 code/date 及因子列；date 为 'YYYY-MM-DD' 字符串。返回
    (n_inserted, n_matched_update)。

    行级（而非日期级）分流：股票级回补的新码，其历史日期在表内已存在
    （旧池股票的行），按日期分流会把这些行全部判进 UPDATE 分支而静默
    丢失（P0-B 机制之一）。
    """
    if pdf.empty:
        return 0, 0

    table = spec.factor_table
    factor_cols = [c for c in pdf.columns if c not in ("code", "date")]
    # 面板新列（如新公式因子首次增量）补齐表结构，不再静默丢弃
    for c in factor_cols:
        ensure_column(con, spec, c, _dtype_of(pdf[c]))
    insert_cols = ["code", "date"] + factor_cols

    pdf = pdf.copy()
    pdf["date"] = pdf["date"].astype(str).str[:10]
    con.execute("CREATE OR REPLACE TEMP TABLE _panel_upd AS SELECT * FROM pdf")

    # 表内缺失行 -> INSERT（列子集；PK (code,date) 兜底防重）
    n_ins = con.execute(f"""
        SELECT COUNT(*) FROM _panel_upd p
        WHERE NOT EXISTS (
            SELECT 1 FROM {table} f
            WHERE f.code = p.code AND f.date = p.date
        )
    """).fetchone()[0]
    if n_ins:
        cols_str = ", ".join(insert_cols)
        con.execute(f"""
            INSERT INTO {table} ({cols_str})
            SELECT {cols_str} FROM _panel_upd p
            WHERE NOT EXISTS (
                SELECT 1 FROM {table} f
                WHERE f.code = p.code AND f.date = p.date
            )
        """)

    n_upd = _update_from_temp(con, spec)
    return n_ins, n_upd


def _update_from_temp(con: duckdb.DuckDBPyConnection, spec: PoolSpec) -> int:
    """把临时表 _panel_upd 中与表内既有行重叠的 (code,date) 行做列回填。

    只更新双方都有的列（保留他方列）；返回匹配到表内行的行数。
    """
    table = spec.factor_table
    temp_cols = [r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = '_panel_upd'").fetchall()]
    upd_cols = [c for c in temp_cols
                if c not in ("code", "date") and c in set(columns(con, spec))]
    if not upd_cols:
        return 0
    set_clause = ", ".join(f"{c} = p.{c}" for c in upd_cols)
    con.execute(f"""
        UPDATE {table} f SET {set_clause}
        FROM _panel_upd p
        WHERE f.code = p.code AND f.date = p.date
    """)
    n = con.execute(f"""
        SELECT COUNT(*) FROM _panel_upd p
        JOIN {table} f ON f.code = p.code AND f.date = p.date
    """).fetchone()[0]
    return n


def update_columns(con: duckdb.DuckDBPyConnection, spec: PoolSpec,
                   pdf: pd.DataFrame) -> int:
    """已有行的列回填（UPDATE ... FROM，保留他方列）。

    pdf 需含 code/date + 待写列；列不在表内则先经 ensure_column 补齐再写
    （gb_/nn_ 模型因子列刷新路径）。返回匹配到表内行的行数。
    """
    if pdf.empty:
        return 0
    for c in pdf.columns:
        if c not in ("code", "date"):
            ensure_column(con, spec, c, _dtype_of(pdf[c]))
    pdf = pdf.copy()
    pdf["date"] = pdf["date"].astype(str).str[:10]
    con.execute("CREATE OR REPLACE TEMP TABLE _panel_upd AS SELECT * FROM pdf")
    return _update_from_temp(con, spec)


def _dtype_of(s: pd.Series) -> str:
    """pandas 列 -> DuckDB 列类型（ensure_column 用；够用即可）。"""
    if pd.api.types.is_integer_dtype(s):
        return "BIGINT"
    if pd.api.types.is_float_dtype(s):
        return "DOUBLE"
    return "VARCHAR"


def ensure_column(con: duckdb.DuckDBPyConnection, spec: PoolSpec,
                  col: str, dtype: str = "DOUBLE") -> bool:
    """确保列存在（不存在则 ALTER ADD）。返回是否新增。"""
    if col in set(columns(con, spec)):
        return False
    con.execute(f"ALTER TABLE {spec.factor_table} ADD COLUMN {col} {dtype}")
    return True


def drop_column(con: duckdb.DuckDBPyConnection, spec: PoolSpec, col: str) -> bool:
    """删除列（不存在返回 False，不报错）。调用方负责反向依赖审查。"""
    if col not in set(columns(con, spec)):
        return False
    con.execute(f"ALTER TABLE {spec.factor_table} DROP COLUMN {col}")
    return True


# ============================================================================
# 对账锚（daily_kline 侧）
# ============================================================================

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
