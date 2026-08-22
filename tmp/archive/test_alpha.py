# -*- coding: utf-8 -*-
"""
自动化因子测试：任意 DSL 表达式清单 -> IC 预测力 + 相关度报告。

复用现有组件：utility.calculate_by_expression（表达式执行）、
labels.compute_forward_returns（前向收益，退市感知）、select_factors 的
逐日 rank IC 口径（ST/退市排除，训练窗口一致）。

产出三类指标：
  1. 预测力：逐日截面 Spearman IC -> mean / ICIR / t 值 / 正率 / 分年度均值
  2. 新因子间相关：逐日截面相关按日平均（同 select_factors）
  3. 与现有入模因子的最大 |corr| 及其来源（变体检测器）

报告同时写 JSON（data/alpha_test_report.json），可直接回填给 LLM 反馈循环。

用法：
    python -m factors.test_alpha --demo                 # 内置演示因子
    python -m factors.test_alpha exprs.json             # [{"name":..., "expr":...},...]
    python -m factors.test_alpha exprs.json --start 2018-01-01 --end 2024-12-31
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_PATH, POOL_NAME, get_pool_codes
from factors.utility import calculate_by_expression
from strategies.labels import compute_forward_returns

TRAIN_START = pd.Timestamp("2015-01-01")
TEST_START = pd.Timestamp("2025-06-01")
HORIZONS = [5, 10, 20, 30]
PRIMARY_HORIZON = 20
MIN_STOCKS_PER_DATE = 30

REPORT_PATH = DB_PATH.parent / "alpha_test_report.json"

DEMO_EXPRS = [
    {"name": "MO01_rev5", "expr": "-(close/ts_delay(close,5)-1)"},
    {"name": "VO04_max20", "expr": "ts_max(ret,20)"},
    {"name": "LQ01_amihud20", "expr": "ts_mean(abs(ret)/amount,20)"},
    {"name": "CH01_dWinnerRate5", "expr": "winner_rate - ts_delay(winner_rate,5)"},
    {"name": "LM18_closePos20", "expr": "ts_mean((close-open)/(high-low),20)"},
]


# ---- 数据准备（仿 compute_panel 的加载口径，独立轻量版）----

def load_base_data(con, codes) -> pl.DataFrame:
    ph = ",".join(["?"] * len(codes))
    k = con.execute(f"""
        SELECT code, date, open, high, low, close, volume, amount
        FROM daily_kline WHERE code IN ({ph}) ORDER BY code, date
    """, codes).fetchdf()
    if k.empty:
        raise RuntimeError("no kline data for pool")
    k["date"] = k["date"].astype(str)
    df = pl.from_pandas(k).rename({"code": "vt_symbol", "date": "datetime"})

    mb = con.execute(f"""
        SELECT code, date, total_mv, circ_mv FROM daily_basic
        WHERE code IN ({ph}) ORDER BY code, date
    """, codes).fetchdf()
    if not mb.empty:
        mb["date"] = mb["date"].astype(str)
        df = df.join(
            pl.from_pandas(mb).rename({"code": "vt_symbol", "date": "datetime"}),
            on=["vt_symbol", "datetime"], how="left")

    cyq = con.execute(f"""
        SELECT code, date, winner_rate, weight_avg, his_low, his_high,
               cost_5pct, cost_15pct, cost_50pct, cost_85pct, cost_95pct
        FROM cyq_perf WHERE code IN ({ph}) ORDER BY code, date
    """, codes).fetchdf()
    if not cyq.empty:
        cyq["date"] = cyq["date"].astype(str)
        df = df.join(
            pl.from_pandas(cyq).rename({"code": "vt_symbol", "date": "datetime"}),
            on=["vt_symbol", "datetime"], how="left")

    # 派生列（与 compute.py / AGENTS.md 口径一致）
    df = df.with_columns(
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("vwap"))
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("vt_symbol") - 1).alias("ret"))
    df = df.with_columns(
        (pl.col("volume") * pl.col("close")
         / pl.col("circ_mv").replace(0, None)).alias("turnover"))
    df = df.with_columns(pl.col("total_mv").alias("cap"))
    df = df.sort(["vt_symbol", "datetime"])
    return df


def load_isst(con, codes) -> pd.Series:
    ph = ",".join(["?"] * len(codes))
    try:
        st = con.execute(f"""
            SELECT date, code, IsST FROM factor_values WHERE code IN ({ph})
        """, codes).fetchdf()
        st["date"] = pd.to_datetime(st["date"])
        st = st.set_index(["date", "code"])["IsST"].astype(bool)
        return st
    except Exception:
        return pd.Series(dtype=bool)


def load_existing_factors(con, codes, window: tuple) -> pd.DataFrame:
    """现有入模因子（selected_{pool}.json），用于变体检测。"""
    sel_path = Path(__file__).parent / f"selected_{POOL_NAME}.json"
    if not sel_path.exists():
        return pd.DataFrame()
    selected = json.loads(sel_path.read_text(encoding="utf-8")).get("selected_factors", [])
    if not selected:
        return pd.DataFrame()
    ph = ",".join(["?"] * len(codes))
    cols = ", ".join(f'"{c}"' for c in selected)
    df = con.execute(f"""
        SELECT date, code, {cols} FROM factor_values
        WHERE code IN ({ph}) AND date >= ? AND date < ?
    """, [*codes, str(window[0].date()), str(window[1].date())]).fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["date", "code"]).sort_index()


# ---- 表达式执行 ----

def eval_expressions(base_df: pl.DataFrame, exprs: list[dict]):
    """逐表达式执行，返回 (name -> pd.Series indexed by (date,code), errors)。"""
    codes = base_df["vt_symbol"].unique().to_list()
    idx = pd.MultiIndex.from_arrays([
        pd.to_datetime(base_df["datetime"].to_list()),
        base_df["vt_symbol"].to_list(),
    ], names=["date", "code"])
    out, errors = {}, {}
    for item in exprs:
        name, expr = item["name"], item["expr"]
        try:
            res = calculate_by_expression(base_df, expr)
            out[name] = pd.Series(res["data"].to_numpy(), index=idx, name=name)
        except Exception as e:
            errors[name] = f"{type(e).__name__}: {e}"
    return out, errors


# ---- IC / 相关（向量化，口径同 select_factors）----

def _rank_ic(f_vals, l_vals):
    valid = ~np.isnan(f_vals) & ~np.isnan(l_vals)
    if valid.sum() < MIN_STOCKS_PER_DATE:
        return np.nan
    f_r, l_r = rankdata(f_vals[valid]), rankdata(l_vals[valid])
    f_c, l_c = f_r - f_r.mean(), l_r - l_r.mean()
    denom = np.sqrt(np.dot(f_c, f_c) * np.dot(l_c, l_c))
    return np.dot(f_c, l_c) / denom if denom else np.nan


def daily_ic(factor: pd.Series, labels: pd.DataFrame, horizons) -> pd.DataFrame:
    df = factor.to_frame("f").join(labels)
    df = df.dropna(subset=["f"])
    dates = df.index.get_level_values("date")
    records = []
    for date, g in df.groupby(level="date", sort=True):
        f_vals = g["f"].to_numpy(dtype=np.float64, copy=True)
        f_vals[~np.isfinite(f_vals)] = np.nan
        for h in horizons:
            ic = _rank_ic(f_vals, g[h].to_numpy(dtype=np.float64))
            if not np.isnan(ic):
                records.append({"date": date, "horizon": h, "ic": ic})
    return pd.DataFrame(records)


def yearly_ic_summary(ic_df: pd.DataFrame, horizon) -> pd.Series:
    sub = ic_df[ic_df["horizon"] == horizon]
    return sub.groupby(sub["date"].dt.year)["ic"].mean()


def mean_cross_corr(factors_df: pd.DataFrame) -> pd.DataFrame:
    """逐日截面相关（Pearson on values）按日平均，同 select_factors 口径。"""
    dates = factors_df.index.get_level_values("date")
    corr_sum, n = None, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for date, g in factors_df.groupby(level="date", sort=True):
            if len(g) < MIN_STOCKS_PER_DATE:
                continue
            arr = np.clip(g.to_numpy(dtype=np.float64), -1e15, 1e15)
            m = pd.DataFrame(arr, columns=factors_df.columns).corr(
                min_periods=MIN_STOCKS_PER_DATE).fillna(0.0).values
            corr_sum = m if corr_sum is None else corr_sum + m
            n += 1
    if not n:
        return pd.DataFrame()
    out = pd.DataFrame(corr_sum / n, index=factors_df.columns,
                       columns=factors_df.columns).copy()
    vals = out.to_numpy(copy=True)
    np.fill_diagonal(vals, 1.0)
    out = pd.DataFrame(vals, index=out.index, columns=out.columns)
    return out


# ---- 报告 ----

def main():
    ap = argparse.ArgumentParser(description="Batch test factor expressions: IC + correlation")
    ap.add_argument("expr_file", nargs="?", help="JSON file: [{name, expr}, ...]")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--start", default=str(TRAIN_START.date()))
    ap.add_argument("--end", default=str(TEST_START.date()))
    args = ap.parse_args()

    if args.demo:
        exprs = DEMO_EXPRS
    elif args.expr_file:
        raw = json.loads(Path(args.expr_file).read_text(encoding="utf-8"))
        exprs = raw if isinstance(raw, list) else raw.get("factors", [])
    else:
        ap.error("need expr_file or --demo")

    window = (pd.Timestamp(args.start), pd.Timestamp(args.end))
    codes = get_pool_codes()
    print(f"Pool: {len(codes)} stocks | window {window[0].date()}~{window[1].date()} | "
          f"{len(exprs)} expressions")

    con = duckdb.connect(str(DB_PATH), read_only=True)
    t0 = time.time()
    base = load_base_data(con, codes)
    base = base.filter(
        (pl.col("datetime") >= str(window[0].date()))
        & (pl.col("datetime") < str(window[1].date()))
    )
    kline = con.execute(
        f"SELECT code, date, open, close FROM daily_kline WHERE code IN "
        f"({','.join(['?'] * len(codes))}) AND date >= ? AND date < ? ORDER BY code, date",
        [*codes, window[0], window[1]]).fetchdf()
    kline["date"] = pd.to_datetime(kline["date"])
    kline = kline.sort_values(["code", "date"])
    delist_rows = con.execute("SELECT code, delist_date FROM delist_info").fetchall()
    delist = {c: pd.Timestamp(d) for c, d in delist_rows}
    isst = load_isst(con, codes)
    existing = load_existing_factors(con, codes, window)
    con.close()
    print(f"Data loaded: {len(base)} rows, {base['datetime'].n_unique()} dates "
          f"({time.time()-t0:.1f}s)")

    # 表达式 -> 因子值
    t1 = time.time()
    factor_map, errors = eval_expressions(base, exprs)
    if errors:
        print("\n表达式执行失败:")
        for n, e in errors.items():
            print(f"  {n}: {e}")
    if not factor_map:
        print("ERROR: all expressions failed")
        return
    factors_df = pd.DataFrame(factor_map).sort_index()

    # 标签 + 排除（ST / 退市 / NaN），同 select_factors
    labels = pd.DataFrame({
        h: compute_forward_returns(kline, horizon=h, delist_info=delist)
        for h in HORIZONS})
    common = factors_df.index.intersection(labels.index)
    factors_df, labels = factors_df.loc[common], labels.loc[common]
    exclude = ~labels.notna().all(axis=1)
    if len(isst):
        exclude |= isst.reindex(labels.index).fillna(False).astype(bool)
    exclude |= (labels == -1.0).any(axis=1)
    factors_df = factors_df.loc[~exclude]
    labels = labels.loc[~exclude]
    print(f"Evaluated: {len(factors_df)} rows, "
          f"{factors_df.index.get_level_values('date').nunique()} dates "
          f"({time.time()-t1:.1f}s)")

    # ---- IC ----
    ic_results = {}
    yearly = {}
    for name in factors_df.columns:
        ic_df = daily_ic(factors_df[name], labels, HORIZONS)
        primary = ic_df[ic_df["horizon"] == PRIMARY_HORIZON]["ic"]
        if len(primary) < 50:
            ic_results[name] = {"error": f"insufficient IC dates ({len(primary)})"}
            continue
        ic_std = primary.std()
        ic_results[name] = {
            "ic_mean": float(primary.mean()),
            "ic_std": float(ic_std),
            "icir": float(primary.mean() / ic_std) if ic_std > 0 else 0.0,
            "t_stat": float(primary.mean() / ic_std * np.sqrt(len(primary)))
            if ic_std > 0 else 0.0,
            "hit_rate": float((primary > 0).mean()),
            "n_dates": int(len(primary)),
        }
        yearly[name] = yearly_ic_summary(ic_df, PRIMARY_HORIZON).round(4).to_dict()

    # ---- 相关：新因子间 ----
    corr_new = mean_cross_corr(factors_df)

    # ---- 相关：与现有入模因子 ----
    vs_existing = {}
    if not existing.empty:
        common2 = factors_df.index.intersection(existing.index)
        if len(common2):
            merged = factors_df.loc[common2].join(existing.loc[common2], how="inner")
            corr_all = mean_cross_corr(merged)
            new_cols, old_cols = list(factors_df.columns), list(existing.columns)
            sub = corr_all.loc[new_cols, old_cols].abs()
            for name in new_cols:
                if not sub.loc[name].empty and sub.loc[name].max() > 0:
                    vs_existing[name] = {
                        "max_abs_corr": float(sub.loc[name].max()),
                        "most_similar_to": str(sub.loc[name].idxmax()),
                    }

    # ---- 终端报告 ----
    print(f"\n{'='*88}")
    print(f"  IC 报告（主 horizon = {PRIMARY_HORIZON}d，窗口 {window[0].date()}~{window[1].date()}）")
    print(f"{'='*88}")
    print(f"{'因子':24s} {'IC均值':>8s} {'ICIR':>7s} {'t值':>7s} {'正率':>6s} {'vs现有池最大相关':>14s}")
    for name, r in sorted(ic_results.items(),
                          key=lambda kv: -abs(kv[1].get("ic_mean", 0) or 0)):
        if "error" in r:
            print(f"{name:24s} ERROR: {r['error']}")
            continue
        ve = vs_existing.get(name, {})
        corr_tag = (f"{ve['max_abs_corr']:.3f} ({ve['most_similar_to'][:18]})"
                    if ve else "-")
        print(f"{name:24s} {r['ic_mean']:>+8.4f} {r['icir']:>7.2f} "
              f"{r['t_stat']:>7.1f} {r['hit_rate']:>6.1%} {corr_tag:>20s}")

    if not corr_new.empty and len(corr_new) > 1:
        print(f"\n新因子间相关矩阵（逐日平均）:")
        print(corr_new.round(3).to_string())

    if yearly:
        print(f"\n分年度 IC（主 horizon）:")
        ydf = pd.DataFrame(yearly)
        print(ydf.to_string())

    # ---- JSON 报告（供 LLM 反馈循环回填）----
    report = {
        "window": [str(window[0].date()), str(window[1].date())],
        "pool": POOL_NAME,
        "primary_horizon": PRIMARY_HORIZON,
        "results": ic_results,
        "yearly_ic": yearly,
        "errors": errors,
        "corr_among_new": corr_new.round(4).to_dict() if not corr_new.empty else {},
        "corr_vs_existing": vs_existing,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\nJSON 报告: {REPORT_PATH}")


if __name__ == "__main__":
    main()
