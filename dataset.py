# -*- coding: utf-8 -*-
"""训练装配单点（2026-08-27 多池化重写）。

收编此前五个文件各自复制的装载+过滤链（run_lgb / factors.select_factors /
factors.mining / _leak_check / factors.baseline_check）：
  - 装载：因子宽表面板（factors/store 单点 SQL）、标签 K 线、退市表、
    行业编码、IsST/次日开盘封板掩码
  - 纯函数：training_panel_index（面板 ∩ 标签非 NaN → date>=train_start →
    当期档成员——run_lgb.train_model 与 _leak_check C3 复刻共用同一实现，
    消手动同步）；label_far_cross（标签远引用越界掩码，train_exclude 组装
    与 C1e 复算共用）
  - 训练协议常量：TRAIN_START/TEST_START/TEST_END/WARMUP_DAYS/CALIB_TAIL_DAYS/
    MIN_TRAIN（原 run_lgb 模块常量迁入，_leak_check 从此 import 自本文件）

池身份：一律显式传 PoolSpec（无默认池——调用方 CLI 层解析，防静默错宇宙）。
数据加载范围 = union_codes(since=spec.data_since)（池时点化 2026-08-24 语义：
首档覆盖训练起点 2020-01 的完整成员）。

等价性契约：load_factors/load_kline 的行集与列序与旧 run_lgb 实现逐字节
一致（SELECT * + 同谓词 + set_index.sort_index），重训 parquet 可对拍。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MODEL_CONFIGS
from pools.spec import PoolSpec
from pools.membership import union_codes, member_mask
from strategies.labels import compute_median_open, compute_nextopen_limit_mask
from factors import store

# ---- 训练协议常量（原 run_lgb 模块常量迁入；_leak_check 单源 import）----
TRAIN_START = pd.Timestamp("2020-01-01")
TEST_START = pd.Timestamp("2025-06-01")
TEST_END = pd.Timestamp("2026-06-01")
WARMUP_DAYS = 90
# 输出校准：训练窗内留出尾段（交易日数）估计 out-of-sample 收缩斜率
CALIB_TAIL_DAYS = 60
MIN_TRAIN = 252


# ============================================================================
# 装载
# ============================================================================

def load_factors(con: duckdb.DuckDBPyConnection, spec: PoolSpec) -> pd.DataFrame:
    """因子宽表面板：union_codes(since=data_since) 成员的全列行集，
    (date, code) MultiIndex 日期升序。"""
    codes = union_codes(since=spec.data_since, con=con, pool=spec.name)
    df = store.load_panel(con, spec, codes=codes)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["date", "code"]).sort_index()


def load_kline(con: duckdb.DuckDBPyConnection, spec: PoolSpec,
               start: str | None = None) -> pd.DataFrame:
    """标签 K 线（open/close，后复权 VIEW）。"""
    codes = union_codes(since=spec.data_since, con=con, pool=spec.name)
    ph = ",".join(["?"] * len(codes))
    sql = (f"SELECT code, date, open, close FROM daily_kline "
           f"WHERE code IN ({ph})")
    params = list(codes)
    if start is not None:
        sql += " AND date >= ?"
        params.append(str(start)[:10])
    sql += " ORDER BY code, date"
    return con.execute(sql, params).fetchdf()


def load_delist_info(con: duckdb.DuckDBPyConnection) -> dict[str, pd.Timestamp]:
    try:
        df = con.execute("SELECT code, delist_date FROM delist_info").fetchdf()
        if df.empty:
            return {}
        return {r["code"]: pd.Timestamp(r["delist_date"]) for _, r in df.iterrows()}
    except Exception:
        return {}


def load_industry_sw_l3(con: duckdb.DuckDBPyConnection) -> tuple[pd.Series, dict[str, int]]:
    """Load SW L3 codes for all stocks, encode into deterministic integers."""
    df = con.execute(
        "SELECT code, sw_l3_code FROM industry WHERE sw_l3_code IS NOT NULL"
    ).fetchdf()
    if df.empty:
        return pd.Series(dtype=int), {}

    categories = sorted(df["sw_l3_code"].astype(str).unique())
    mapping = {code: i for i, code in enumerate(categories)}
    codes = df["sw_l3_code"].map(mapping).fillna(-1).astype(int)
    return pd.Series(codes.values, index=df["code"], name="sw_l3"), mapping


def load_isst_series(con: duckdb.DuckDBPyConnection, spec: PoolSpec) -> pd.Series:
    """IsST 时点判定（bool Series，(date, code) 索引）。"""
    codes = union_codes(since=spec.data_since, con=con, pool=spec.name)
    df = store.load_isst(con, spec, codes=codes)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["date", "code"])["IsST"].astype(bool)


@dataclass
class TrainingData:
    """主窗口训练的完整装配束（run_lgb / mining / _leak_check 共用形状）。"""
    spec: PoolSpec
    factors: pd.DataFrame                       # (date,code) 宽表（含 IsST 列）
    kline: pd.DataFrame                         # open/close 平铺（标签用）
    delist_info: dict[str, pd.Timestamp]
    industry_sw_l3: pd.Series
    sw_l3_mapping: dict[str, int]
    st_series: pd.Series | None
    limit_mask: pd.Series


def assemble(con: duckdb.DuckDBPyConnection, spec: PoolSpec) -> TrainingData:
    """装配主窗口训练全量数据（原 run_lgb.main 的装载段单点化）。"""
    factors = load_factors(con, spec)
    kline = load_kline(con, spec)
    delist_info = load_delist_info(con)
    industry_sw_l3, sw_l3_mapping = load_industry_sw_l3(con)
    st_series = (factors["IsST"].astype(bool)
                 if "IsST" in factors.columns else None)
    limit_mask = compute_nextopen_limit_mask(kline, st_series=st_series)
    return TrainingData(spec=spec, factors=factors, kline=kline,
                        delist_info=delist_info, industry_sw_l3=industry_sw_l3,
                        sw_l3_mapping=sw_l3_mapping, st_series=st_series,
                        limit_mask=limit_mask)


# ============================================================================
# 纯函数（过滤链/掩码——训练与泄漏断言共用同一实现）
# ============================================================================

def training_panel_index(fv_index: pd.MultiIndex, label: pd.Series,
                         train_start: pd.Timestamp,
                         con=None, spec: PoolSpec = None) -> pd.MultiIndex:
    """训练面板行过滤链：面板 ∩ 标签非 NaN → date >= train_start → 当期档成员。

    run_lgb.train_model 的 X 行集与 _leak_check C3 的复刻共用本实现
    （2026-08-24 语义：ST/退市/封板/远引用越界只从训练剔除，预测行集
    保持全量面板——故链条到此为止）。member_mask 需要池身份：显式调用
    传 spec，或已在持有连接的进程内传 con。
    """
    common = fv_index.intersection(label.index)
    lab = label.reindex(common)
    idx = common[lab.notna().to_numpy()]

    idx_date = idx.get_level_values("date")
    idx = idx[np.asarray(idx_date >= train_start)]

    pool = spec.name if spec is not None else None
    mm = member_mask(idx.get_level_values("date"), idx.get_level_values("code"),
                     con=con, pool=pool)
    return idx[mm]


def label_far_cross(index: pd.MultiIndex, e0: int,
                    test_start: pd.Timestamp) -> pd.Series:
    """标签远引用越界掩码：标签按个股自身交易行前移（在 index 的行序上
    近似 kline 行序，逐股同源——调用方需保证 index 已排序），而
    label_buffer 按池轴回退——停牌股的 T+e0 可落到测试窗内。

    train_exclude 组装（run_lgb）与 C1e 复算（_leak_check）共用本实现。
    返回 bool Series（index 对齐）。
    """
    date_s = pd.Series(index.get_level_values("date"), index=index)
    far_max = pd.concat(
        [date_s.groupby(level="code", sort=False).shift(-lag) for lag in range(1, e0 + 1)],
        axis=1).max(axis=1)
    return (far_max >= test_start).fillna(False)


def compute_model_label(kline: pd.DataFrame, model: str) -> pd.Series:
    """模型标签（compute_median_open），与训练目标同一构造。"""
    cfg = MODEL_CONFIGS[model]
    s, e = cfg["label_window"]
    return compute_median_open(kline, start_day=s, end_day=e,
                               baseline=cfg["baseline"])
