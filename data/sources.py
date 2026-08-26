# -*- coding: utf-8 -*-
"""数据源注册表（函数式，无类层次）：每个源一个 pull 函数 + 粒度声明。

粒度语义（对账与拉取策略的依据）：
- trading  : 按交易日粒度（daily/cyq/index），参与交易日历对账与滚动重拉
- calendar : 按日历日粒度（shibor），不做交易日对账（SHIBOR 非交易日也有报价）
- event    : 事件流（namechange），重叠窗口重拉，不做日期对账
- snapshot : 全量快照（stock_info），每次整体刷新

写入一律幂等（INSERT OR REPLACE），任意重跑安全。

每个 pull(con, pro, dates) 返回写入行数（snapshot 源返回 (rows, extra)）。
postcheck(con, d) 行级后验：行数漂移（目标日前 20 日中位数 ±30%）与
daily 的 adj_factor NULL 率，失败原因交由 pull.py 记入 pending_pulls。
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

from config import DB_PATH, TRACKED_INDICES
from pools.membership import union_codes

from data._ts import retry_api

log = logging.getLogger(__name__)

# ---- 字段常量（与 pull_adj.py 历史一致）----

BASIC_FIELDS = "ts_code,trade_date,total_mv,circ_mv,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm"
BASIC_NUMERIC = ["total_mv", "circ_mv", "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio", "dv_ttm"]
BASIC_COLS = ["code", "date"] + BASIC_NUMERIC

CYQ_FIELDS = ("ts_code,trade_date,his_low,his_high,"
              "cost_5pct,cost_15pct,cost_50pct,cost_85pct,cost_95pct,"
              "weight_avg,winner_rate")
CYQ_NUMERIC = ["his_low", "his_high", "cost_5pct", "cost_15pct", "cost_50pct",
               "cost_85pct", "cost_95pct", "weight_avg", "winner_rate"]
CYQ_ALL_COLS = ["code", "date"] + CYQ_NUMERIC


def _to_code(df: pd.DataFrame) -> pd.DataFrame:
    df["code"] = df["ts_code"].str[:6]
    return df


def _num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---- 表结构（幂等创建，与 build_db.py/pull_adj.py 历史一致）----

def ensure_tables(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS daily_raw (
            code VARCHAR NOT NULL, date DATE NOT NULL,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            volume DOUBLE, amount DOUBLE, pct_chg DOUBLE, turn DOUBLE,
            adj_factor DOUBLE, PRIMARY KEY (code, date))
    """)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS daily_basic (
            code VARCHAR NOT NULL, date DATE NOT NULL,
            {', '.join(f'{c} DOUBLE' for c in BASIC_NUMERIC)},
            PRIMARY KEY (code, date))
    """)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS cyq_perf (
            code VARCHAR NOT NULL, date DATE NOT NULL,
            {', '.join(f'{c} DOUBLE' for c in CYQ_NUMERIC)},
            PRIMARY KEY (code, date))
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS index_daily (
            code VARCHAR NOT NULL, date DATE NOT NULL,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            volume DOUBLE, amount DOUBLE, pct_chg DOUBLE,
            PRIMARY KEY (code, date))
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS macro_daily (
            date DATE PRIMARY KEY, shibor_on DOUBLE, shibor_1m DOUBLE)
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS namechange (
            code VARCHAR NOT NULL, ts_code VARCHAR NOT NULL, name VARCHAR,
            start_date DATE, end_date DATE, ann_date DATE, change_reason VARCHAR,
            PRIMARY KEY (code, start_date))
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS delist_info (
            code VARCHAR PRIMARY KEY, delist_date DATE NOT NULL)
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS stock_info (
            code VARCHAR PRIMARY KEY, name VARCHAR, market VARCHAR,
            full_code VARCHAR, list_date DATE,
            list_status VARCHAR, delist_date DATE)
    """)
    # 存量库补列（幂等；旧表无 list_status/delist_date）
    con.execute("ALTER TABLE stock_info ADD COLUMN IF NOT EXISTS list_status VARCHAR")
    con.execute("ALTER TABLE stock_info ADD COLUMN IF NOT EXISTS delist_date DATE")
    ensure_view(con)


def ensure_view(con):
    """daily_kline VIEW（前复权：raw × adj_factor / latest_adj，最新 adj 为基准）。

    无条件 CREATE OR REPLACE（幂等）：公式修正后存量库在下次 pull 自动重建，
    不再依赖「VIEW 缺失才创建」。
    """
    con.execute("""
        CREATE OR REPLACE VIEW daily_kline AS
        WITH latest_adj AS (
            SELECT code, MAX_BY(adj_factor, date) AS latest_adj
            FROM daily_raw
            WHERE adj_factor IS NOT NULL AND adj_factor > 0
            GROUP BY code
        )
        SELECT
            r.code, r.date,
            r.open   * (r.adj_factor / NULLIF(l.latest_adj, 0)) AS open,
            r.high   * (r.adj_factor / NULLIF(l.latest_adj, 0)) AS high,
            r.low    * (r.adj_factor / NULLIF(l.latest_adj, 0)) AS low,
            r.close  * (r.adj_factor / NULLIF(l.latest_adj, 0)) AS close,
            r.volume, r.amount, r.pct_chg, r.turn
        FROM daily_raw r
        LEFT JOIN latest_adj l ON r.code = l.code
    """)
    log.info("daily_kline VIEW ensured")


# ---- fetch（API 层，移植自 pull_adj.py）----

def _fetch_daily(pro, td: str) -> pd.DataFrame:
    df = retry_api(pro.daily, trade_date=td,
                   fields="ts_code,trade_date,open,high,low,close,vol,amount,pct_chg")
    if df is None or df.empty:
        return pd.DataFrame()
    _to_code(df)
    df["date"] = pd.to_datetime(df["trade_date"])
    df.rename(columns={"vol": "volume"}, inplace=True)
    _num(df, ["open", "high", "low", "close", "volume", "amount", "pct_chg"])
    return df[["code", "date", "open", "high", "low", "close", "volume", "amount", "pct_chg"]]


def _fetch_adj_factor(pro, td: str) -> pd.DataFrame:
    df = retry_api(pro.adj_factor, trade_date=td, fields="ts_code,trade_date,adj_factor")
    if df is None or df.empty:
        return pd.DataFrame()
    _to_code(df)
    df["date"] = pd.to_datetime(df["trade_date"])
    _num(df, ["adj_factor"])
    return df[["code", "date", "adj_factor"]]


def _fetch_daily_basic(pro, td: str) -> pd.DataFrame:
    df = retry_api(pro.daily_basic, ts_code="", trade_date=td, fields=BASIC_FIELDS)
    if df is None or df.empty:
        return pd.DataFrame()
    _to_code(df)
    df["date"] = pd.to_datetime(df["trade_date"])
    _num(df, BASIC_NUMERIC)
    for c in BASIC_COLS:
        if c not in df.columns:
            df[c] = None
    return df[BASIC_COLS]


def _fetch_cyq(pro, td: str) -> pd.DataFrame:
    # 注：官方文档 ts_code 为必填，此处按日全市场拉取依赖 quicksync 中转的
    # 宽松校验；如中转收紧，从 git 历史恢复 build_cyq.py 的逐股拉取模式。
    df = retry_api(pro.cyq_perf, ts_code="", trade_date=td, fields=CYQ_FIELDS)
    if df is None or df.empty:
        return pd.DataFrame()
    _to_code(df)
    df["date"] = pd.to_datetime(df["trade_date"])
    _num(df, CYQ_NUMERIC)
    for c in CYQ_ALL_COLS:
        if c not in df.columns:
            df[c] = None
    return df[CYQ_ALL_COLS]


def _fetch_index(pro, ts_code: str, start: str, end: str) -> pd.DataFrame:
    df = retry_api(pro.index_daily, ts_code=ts_code, start_date=start, end_date=end)
    if df is None or df.empty:
        return pd.DataFrame()
    df.rename(columns={"trade_date": "date", "vol": "volume"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    _num(df, ["open", "high", "low", "close", "volume", "amount", "pct_chg"])
    return df[["date", "open", "high", "low", "close", "volume", "amount", "pct_chg"]]


# ---- 各源 pull（写入层，幂等）----

def pull_daily(con, pro, dates: list[str]) -> int:
    """daily + adj_factor + daily_basic（3 API/天），返回写入行数（daily_raw）。"""
    total = 0
    for td in dates:
        td_compact = td.replace("-", "")
        df_d = _fetch_daily(pro, td_compact)
        if df_d.empty:
            log.info("  daily %s: empty", td)
            continue
        df_a = _fetch_adj_factor(pro, td_compact)
        if not df_a.empty:
            df_d = df_d.merge(df_a, on=["code", "date"], how="left")
        else:
            df_d["adj_factor"] = None
        df_d["turn"] = None
        con.execute("""
            INSERT OR REPLACE INTO daily_raw
                (code, date, open, high, low, close, volume, amount, pct_chg, turn, adj_factor)
            SELECT code, date, open, high, low, close, volume, amount, pct_chg, turn, adj_factor
            FROM df_d
        """)
        total += len(df_d)

        df_b = _fetch_daily_basic(pro, td_compact)
        if not df_b.empty:
            con.execute("INSERT OR REPLACE INTO daily_basic SELECT * FROM df_b")
    return total


def pull_cyq(con, pro, dates: list[str]) -> int:
    """cyq_perf 按日全市场。返回写入行数。"""
    total = 0
    for td in dates:
        df = _fetch_cyq(pro, td.replace("-", ""))
        if df.empty:
            continue
        con.execute("INSERT OR REPLACE INTO cyq_perf SELECT * FROM df")
        total += len(df)
    return total


def pull_index(con, pro, dates: list[str]) -> int:
    """TRACKED_INDICES 区间拉取（一次 API 覆盖整个 dates 区间）。"""
    if not dates:
        return 0
    start, end = dates[0].replace("-", ""), dates[-1].replace("-", "")
    total = 0
    for ts_code, store_code in TRACKED_INDICES:
        df = _fetch_index(pro, ts_code, start, end)
        if df.empty:
            log.info("  index %s: empty", store_code)
            continue
        df["code"] = store_code
        df = df[["code", "date", "open", "high", "low", "close", "volume", "amount", "pct_chg"]]
        con.execute("DELETE FROM index_daily WHERE code=? AND date>=?::DATE", [store_code, dates[0]])
        con.execute("INSERT INTO index_daily SELECT * FROM df")
        total += len(df)
    return total


def pull_shibor(con, pro, dates: list[str]) -> int:
    """SHIBOR（calendar 粒度，dates 为日历日）。

    修复原实现的 dropna() 过激问题：on/1m 任一非空即保留该行。
    """
    if not dates:
        return 0
    start, end = dates[0].replace("-", ""), dates[-1].replace("-", "")
    df = retry_api(pro.shibor, start_date=start, end_date=end)
    if df is None or df.empty:
        return 0
    df["date"] = pd.to_datetime(df["date"])
    out = df[["date", "on", "1m"]].copy()
    out.columns = ["date", "shibor_on", "shibor_1m"]
    out = out.dropna(subset=["shibor_on", "shibor_1m"], how="all")
    if out.empty:
        return 0
    con.execute("INSERT OR REPLACE INTO macro_daily SELECT * FROM out")
    return len(out)


def _pool_union_codes(con) -> set[str]:
    """池代码并集（时点快照全历史成员）--namechange 过滤与行业触发的范围。

    必须传调用方已持有的连接（pull 进程持有写连接，membership 若自开
    只读连接会撞 DuckDB 单写者文件锁）。pool_snapshots 缺表时此处大声
    失败（fail-fast）：池快照是全系统宇宙定义，静默空集会让 namechange
    停更、ST/退市事件断流（2026-08-25 json 池删除事故的教训）。
    """
    return set(union_codes(con=con))


def dedup_namechange(df: pd.DataFrame) -> pd.DataFrame:
    """namechange 去重：先全键去重（只去完全重复行）；同 (code, start_date)
    不同值时保留 ann_date 最新一条（更晚公告≈更正/覆盖）并告警。

    不再 drop_duplicates(keep="first") 静默丢同键不同值行；同时避免
    INSERT OR REPLACE 同一语句内两次写同一主键触发 DuckDB 冲突错误。
    """
    df = df.drop_duplicates()
    dup = df.duplicated(subset=["code", "start_date"], keep=False)
    if dup.any():
        log.warning("namechange: %d rows share (code, start_date) with different "
                    "values; keeping latest ann_date per key", int(dup.sum()))
        df = (df.sort_values("ann_date", na_position="first")
                .drop_duplicates(subset=["code", "start_date"], keep="last"))
    return df


def _sync_delist_info(con):
    """重派生 delist_info（与 namechange 窗口拉取解耦，每次固定执行）。

    stock_basic(list_status='D') 的 delist_date 为主源（最后写入、覆盖补充值，
    为真实摘牌日）；namechange '终止上市' 仅补主源缺失的池内 code（其
    MIN(start_date) 可能早于摘牌日，只作补充）。不再按池并集 DELETE：
    delist_info 以全市场退市档案为准，行集合不随池成员变动。
    """
    pool_codes = _pool_union_codes(con)
    if pool_codes:
        ph = ",".join(["?"] * len(pool_codes))
        con.execute(f"""
            INSERT OR REPLACE INTO delist_info
            SELECT code, MIN(start_date) FROM namechange
            WHERE change_reason = '终止上市' AND code IN ({ph})
            GROUP BY code
        """, sorted(pool_codes))
    con.execute("""
        INSERT OR REPLACE INTO delist_info
        SELECT code, delist_date FROM stock_info
        WHERE list_status = 'D' AND delist_date IS NOT NULL
    """)


def pull_namechange(con, pro, dates: list[str]) -> int:
    """namechange 事件流（忽略 dates）：按公告日（ann_date）锚定近窗口重拉。

    修复原实现的全市场污染：写入限定所有池的并集。
    窗口锚定 ann_date（API 的 start_date/end_date 过滤的是公告日；退市/ST
    公告常远晚于生效日，原按生效日 start_date 锚定 +7d 回看会让公告滞后
    >7d 的事件永久落在窗口外），窗口放宽到 90d 兜底公告滞后与拉取失败。
    """
    n = 0
    raw = con.execute("SELECT MAX(ann_date) FROM namechange").fetchone()[0]
    if raw is not None:
        start_date = (pd.Timestamp(raw) - pd.Timedelta(days=90)).strftime("%Y%m%d")
    else:
        start_date = "20080101"
    end_date = datetime.now().strftime("%Y%m%d")

    if start_date <= end_date:
        df = retry_api(pro.namechange, start_date=start_date, end_date=end_date)
        if df is not None and not df.empty:
            pool_codes = _pool_union_codes(con)
            df["code"] = df["ts_code"].str[:6]
            df = df[df["code"].isin(pool_codes)]
            if not df.empty:
                df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce").dt.date
                df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
                df["ann_date"] = pd.to_datetime(df["ann_date"], errors="coerce").dt.date
                cols = ["code", "ts_code", "name", "start_date", "end_date", "ann_date", "change_reason"]
                df = dedup_namechange(df[cols])
                con.execute("INSERT OR REPLACE INTO namechange SELECT * FROM df")
                n = len(df)

    _sync_delist_info(con)
    return n


def pull_stock_info(con, pro, dates: list[str]) -> tuple[int, list[str]]:
    """stock_info 快照（忽略 dates）。返回 (写入行数, 新增 code 列表)。

    原系统无此源：新上市股票永远进不了 stock_info，LnAge/industry 缺失。
    现合并拉取 list_status='L'（当前上市）+ 'D'（退市）：表为只增不删的
    累积快照，只拉 L 会让历史退市股永久缺席（池构建幸存者偏差的根因），
    故每次以 D 列表回填退市档案（含 list_status/delist_date）。
    """
    fields = "ts_code,symbol,name,area,industry,market,list_date,list_status,delist_date"
    frames = []
    for status in ("L", "D"):
        df = retry_api(pro.stock_basic, exchange="", list_status=status, fields=fields)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return 0, []
    df = pd.concat(frames, ignore_index=True)
    # 同一 ts_code 不应同时出现在 L/D 两个列表；万一出现，保留 L（当前上市）
    # 避免同一主键在一次 INSERT OR REPLACE 中写入两次导致整批失败。
    df = df.drop_duplicates(subset=["ts_code"], keep="first")
    existing = {r[0] for r in con.execute("SELECT code FROM stock_info").fetchall()}
    df["code"] = df["ts_code"].str[:6]
    new_codes = sorted(set(df["code"]) - existing)

    out = pd.DataFrame({
        "code": df["code"],
        "name": df.get("name", ""),
        "market": df.get("market", ""),
        "full_code": df["ts_code"],
        "list_date": pd.to_datetime(df["list_date"], format="%Y%m%d", errors="coerce"),
        "list_status": df.get("list_status", ""),
        "delist_date": pd.to_datetime(df.get("delist_date"), format="%Y%m%d", errors="coerce"),
    })
    con.execute("""
        INSERT OR REPLACE INTO stock_info
            (code, name, market, full_code, list_date, list_status, delist_date)
        SELECT code, name, market, full_code, list_date, list_status, delist_date
        FROM out
    """)
    return len(out), new_codes


# ---- 行级后验 ----

def _median_rows_before(con, table: str, date_str: str, n: int = 20):
    row = con.execute(
        f"""SELECT median(n) FROM (
                SELECT COUNT(*) n FROM {table}
                WHERE date < ?::DATE GROUP BY date
                ORDER BY date DESC LIMIT {n})""",
        [date_str],
    ).fetchone()
    return row[0] if row else None


def postcheck(con, source: str, table: str, date_str: str) -> str | None:
    """行级后验：返回失败原因（None=通过）。

    - 行数骤降：当日行数 < 此前 20 个有数据日中位数的 70%（数据丢失方向）。
      只检测骤降不检测暴增：覆盖范围扩大（如 cyq 从池内 ~1200 行/日改为
      全市场 ~5500 行/日）是改善而非异常。
    - daily 源附加：adj_factor NULL 率 >1%
    """
    row = con.execute(
        f"SELECT COUNT(*) FROM {table} WHERE date = ?::DATE", [date_str]
    ).fetchone()
    n = row[0] if row else 0
    if n == 0:
        return "0 rows"
    med = _median_rows_before(con, table, date_str)
    if med and n < med * 0.70:
        return f"row drop: {n} vs median {med}"
    if med and n > med * 1.30:
        log.info("  postcheck[%s %s]: rows %d > median %d (coverage widened, ok)",
                 table, date_str, n, med)
    if source == "daily":
        nulls = con.execute(
            "SELECT COUNT(*) FROM daily_raw WHERE date=?::DATE AND adj_factor IS NULL",
            [date_str],
        ).fetchone()[0]
        if nulls / n > 0.01:
            return f"adj_factor NULL {nulls}/{n}"
    return None


SOURCES = {
    "daily":       {"grain": "trading",  "tables": ["daily_raw", "daily_basic"], "pull": pull_daily},
    "cyq":         {"grain": "trading",  "tables": ["cyq_perf"],                "pull": pull_cyq},
    "index":       {"grain": "trading",  "tables": ["index_daily"],             "pull": pull_index},
    "shibor":      {"grain": "calendar", "tables": ["macro_daily"],             "pull": pull_shibor},
    "namechange":  {"grain": "event",    "tables": ["namechange"],              "pull": pull_namechange},
    "stock_info":  {"grain": "snapshot", "tables": ["stock_info"],              "pull": pull_stock_info},
}
