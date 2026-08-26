# -*- coding: utf-8 -*-
"""
评估器回归门禁：8 个基准因子（原生 Polars 重实现，2026-08-25 并入原
factors/baseline_alphas.py）的固定窗口 rank IC 汇总，与冻结参考值
（factors/baseline_reference_{pool}.json，按池独立冻结）比对。

用途：评估口径（select_factors 的 IC 计算 / labels 前向收益）变更后，
跑本工具确认"已知因子"的 IC 没有非预期漂移。手动工具，不进每日流水线。

    python -m factors.baseline_check           # 比对模式，超限 exit 1
    python -m factors.baseline_check --init    # 冻结/刷新参考值

口径与 select_factors 主线一致：固定训练窗（2018-01-01 ~ 2025-06-01）、
逐日截面 Spearman rank IC、剔除 IsST=1 与退市填充(-1.0)与 NaN 标签行、
截面最少 30 只。标签为 close 锚 20d 前向收益（经典口径）。

repaint 注意：daily_kline 为前复权 VIEW，新除权事件会使历史 IC 缓慢
漂移--容差带（ic_mean ±0.01 / icir ±0.10）用于吸收；无评估器变更却
持续超限时，用 --init 重新冻结并记录原因。
（2026-08-25 宇宙切换：池代码从旧 json 并集改为时点快照并集，参考值已
随之重冻结，见 baseline_reference.json 的 frozen_at。）

公式出处：Kakushadze (2016)《101 Formulaic Alphas》（document/alpha101.md）
及已删除的 vnpy 移植版 alpha101.py，表达式原文存档于 BASELINE_EXPRESSIONS。
实现与原 DSL 算子语义逐一对齐（2026-08-22 全池交叉验证 |Pearson|>0.999）：
  ts_std(x, w)   -> rolling_std(w, min_samples=1, ddof=0)
  ts_corr(a,b,w) -> pl.rolling_corr(w, min_samples=1)；inf -> null
  ts_rank(x, w)  -> 过去 w-1 个值中严格小于当前值的占比（分母为非空
                    shifted 计数；为 0 时置 0）；inf/nan -> null
  ts_argmax(x,w) -> 窗口内最大值位置（1-based，np.argmax 首现）
  cs_rank(x)     -> 同日截面 rank()/count() 百分位（average 法，null 不参与）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_PATH
from pools.spec import get_pool, PoolSpec
from factors.select_factors import MIN_STOCKS_PER_DATE, _rank_ic_np
from strategies.labels import compute_forward_returns
from pools.membership import union_codes
from factors import store

TRAIN_START = "2018-01-01"
TRAIN_END = "2025-06-01"
HORIZON = 20
LOAD_FROM = "2017-09-01"   # 滚动窗预热（最长回看 ~30 交易日）
LOAD_TO = "2025-08-31"     # 标签需 T+20 收盘
IC_MEAN_TOL = 0.01
ICIR_TOL = 0.10

def _ref_path(spec: PoolSpec) -> Path:
    """冻结参考值按池命名（原 baseline_reference.json 为微盘池冻结值 git mv 而来）。"""
    return Path(__file__).resolve().parent / f"baseline_reference_{spec.name}.json"

BASELINE_FACTORS = [
    "alpha1_v0", "alpha18_v0", "alpha50_v0", "alpha60_v0",
    "alpha6", "alpha40", "alpha42", "alpha101",
]

# 原表达式存档（自 alpha101.py 抄录，DSL 已删除）
BASELINE_EXPRESSIONS = {
    "alpha1_v0": "cs_rank(ts_argmax(pow1(ts_corr(close, volume, 5), 2.0), 5) - 0.5)",
    "alpha18_v0": "-1 * ((ts_std(abs(close - open), 5) + (close - open)) + ts_corr(close, open, 10))",
    "alpha50_v0": "cs_rank(-1 * ts_corr(ts_rank(close, 10), ts_rank(volume, 10), 10))",
    "alpha60_v0": "-1 * (ts_rank(ts_std(close, 20), 10) - ts_rank(ts_std(close, 5), 10))",
    "alpha6": "(-1) * ts_corr(open, volume, 10)",
    "alpha40": "((-1) * cs_rank(ts_std(high, 10))) * ts_corr(high, volume, 10)",
    "alpha42": "cs_rank((vwap - close)) / cs_rank((vwap + close))",
    "alpha101": "((close - open) / ((high - low) + 0.001))",
}


# ============================================================================
# 基准因子实现（输入约定：宽表 [datetime, vt_symbol, open, high, low, close,
# volume, vwap]，内部自排 (vt_symbol, datetime)）
# ============================================================================

def _ts_std(col: str, window: int) -> pl.Expr:
    return pl.col(col).rolling_std(window, min_samples=1, ddof=0).over("vt_symbol")


def _ts_rank(col: str, window: int) -> pl.Expr:
    """ts_ops.ts_rank 等价：count(shift_i < cur, i=1..w-1) / count(非空 shift)。"""
    lt = pl.lit(0, dtype=pl.Int32)
    cnt = pl.lit(0, dtype=pl.Int32)
    for i in range(1, window):
        shifted = pl.col(col).shift(i)
        lt = lt + (shifted < pl.col(col)).cast(pl.Int32)
        cnt = cnt + shifted.is_not_null().cast(pl.Int32)
    rank_expr = lt / pl.when(cnt > 0).then(cnt).otherwise(1)
    rank_expr = pl.when(rank_expr.is_infinite() | rank_expr.is_nan()) \
        .then(None).otherwise(rank_expr)
    return rank_expr.over("vt_symbol")


def _ts_argmax(col: str, window: int) -> pl.Expr:
    """ts_ops.ts_argmax 等价：窗口内最大值 1-based 位次（首现）。"""
    return pl.col(col).rolling_map(
        lambda s: int(np.argmax(s.to_numpy())) + 1, window
    ).over("vt_symbol")


def _cs_rank(col: str) -> pl.Expr:
    """cs_ops.cs_rank 等价：同日截面百分位 (0,1]。"""
    return pl.col(col).rank().over("datetime") / pl.col(col).count().over("datetime")


def _rolling_corr(df: pl.DataFrame, a: str, b: str, window: int, out: str) -> pl.DataFrame:
    """ts_ops.ts_corr 等价（需物化输入列后按名计算）。"""
    df = df.with_columns(
        pl.rolling_corr(a, b, window_size=window, min_samples=1)
        .over("vt_symbol").alias(out)
    )
    return df.with_columns(
        pl.when(pl.col(out).is_infinite()).then(None)
        .otherwise(pl.col(out)).alias(out)
    )


def _clean(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    return df.with_columns([
        pl.when(pl.col(c).is_infinite() | pl.col(c).is_nan())
        .then(None).otherwise(pl.col(c)).alias(c)
        for c in cols
    ])


def compute_baseline_factors(ohlcv: pl.DataFrame) -> pl.DataFrame:
    """在 OHLCV 宽表上追加 8 个基准因子列，返回原列 + 基准列。"""
    df = ohlcv.sort(["vt_symbol", "datetime"])

    # alpha1_v0
    df = _rolling_corr(df, "close", "volume", 5, "_a1_c")
    df = df.with_columns((pl.col("_a1_c") ** 2.0).alias("_a1_x"))
    df = df.with_columns(_ts_argmax("_a1_x", 5).alias("_a1_arg"))
    df = df.with_columns((pl.col("_a1_arg") - 0.5).alias("_a1_arg2"))
    df = df.with_columns(_cs_rank("_a1_arg2").alias("alpha1_v0"))

    # alpha18_v0
    df = df.with_columns((pl.col("close") - pl.col("open")).abs().alias("_a18_do"))
    df = df.with_columns(_ts_std("_a18_do", 5).alias("_a18_std"))
    df = _rolling_corr(df, "close", "open", 10, "_a18_c")
    df = df.with_columns(
        (-1 * ((pl.col("_a18_std") + (pl.col("close") - pl.col("open")))
               + pl.col("_a18_c"))).alias("alpha18_v0")
    )

    # alpha50_v0
    df = df.with_columns(_ts_rank("close", 10).alias("_a50_rc"))
    df = df.with_columns(_ts_rank("volume", 10).alias("_a50_rv"))
    df = _rolling_corr(df, "_a50_rc", "_a50_rv", 10, "_a50_c")
    df = df.with_columns((-1 * pl.col("_a50_c")).alias("_a50_nc"))
    df = df.with_columns(_cs_rank("_a50_nc").alias("alpha50_v0"))

    # alpha60_v0
    df = df.with_columns(_ts_std("close", 20).alias("_a60_s20"))
    df = df.with_columns(_ts_std("close", 5).alias("_a60_s5"))
    df = df.with_columns(_ts_rank("_a60_s20", 10).alias("_a60_r20"))
    df = df.with_columns(_ts_rank("_a60_s5", 10).alias("_a60_r5"))
    df = df.with_columns(
        (-1 * (pl.col("_a60_r20") - pl.col("_a60_r5"))).alias("alpha60_v0")
    )

    # alpha6
    df = _rolling_corr(df, "open", "volume", 10, "_a6_c")
    df = df.with_columns((-1 * pl.col("_a6_c")).alias("alpha6"))

    # alpha40
    df = df.with_columns(_ts_std("high", 10).alias("_a40_s"))
    df = df.with_columns(_cs_rank("_a40_s").alias("_a40_r"))
    df = _rolling_corr(df, "high", "volume", 10, "_a40_c")
    df = df.with_columns(
        (((-1) * pl.col("_a40_r")) * pl.col("_a40_c")).alias("alpha40")
    )

    # alpha42
    df = df.with_columns((pl.col("vwap") - pl.col("close")).alias("_a42_d"))
    df = df.with_columns((pl.col("vwap") + pl.col("close")).alias("_a42_s"))
    df = df.with_columns(_cs_rank("_a42_d").alias("_a42_rd"))
    df = df.with_columns(_cs_rank("_a42_s").alias("_a42_rs"))
    df = df.with_columns((pl.col("_a42_rd") / pl.col("_a42_rs")).alias("alpha42"))

    # alpha101
    df = df.with_columns(
        ((pl.col("close") - pl.col("open"))
         / ((pl.col("high") - pl.col("low")) + 0.001)).alias("alpha101")
    )

    df = _clean(df, BASELINE_FACTORS)
    drop_cols = [c for c in df.columns if c.startswith(("_a1_", "_a18_", "_a50_",
                                                         "_a60_", "_a6_", "_a40_",
                                                         "_a42_"))]
    return df.drop(drop_cols)


# ============================================================================
# 门禁主体
# ============================================================================

def load_kline_pd(con, codes):
    ph = ",".join(["?"] * len(codes))
    df = con.execute(
        f"SELECT code, date, close FROM daily_kline "
        f"WHERE code IN ({ph}) AND date >= '{LOAD_FROM}' AND date <= '{LOAD_TO}' "
        f"ORDER BY code, date", codes
    ).fetchdf()
    return df


def load_delist_info(con) -> dict:
    try:
        df = con.execute("SELECT code, delist_date FROM delist_info").fetchdf()
        return {r["code"]: pd.Timestamp(r["delist_date"]) for _, r in df.iterrows()}
    except Exception:
        return {}


def load_isst(con, spec, codes) -> pd.Series:
    df = store.load_isst(con, spec, codes=codes)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["date", "code"])["IsST"]


def compute_metrics(spec: PoolSpec) -> dict:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    # 池时点化（2026-08-25）：时点快照全历史成员并集（原 json 池并集已删）
    codes = union_codes(con=con, pool=spec.name)

    # 行情（polars 面板，供因子计算）
    ph = ",".join(["?"] * len(codes))
    kdf = con.execute(
        f"SELECT code, date, open, high, low, close, volume FROM daily_kline "
        f"WHERE code IN ({ph}) AND date >= '{LOAD_FROM}' AND date <= '{LOAD_TO}' "
        f"ORDER BY code, date", codes
    ).fetchdf()
    kdf["date"] = kdf["date"].astype(str)
    panel = pl.from_pandas(kdf).rename({"code": "vt_symbol", "date": "datetime"})
    panel = panel.with_columns(
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("vwap")
    )
    res = compute_baseline_factors(panel)

    # 标签（close 锚 20d，退市感知）
    kline_pd = load_kline_pd(con, codes)
    data_max_date = str(kline_pd["date"].max())
    delist_info = load_delist_info(con)
    fwd = compute_forward_returns(kline_pd, horizon=HORIZON, delist_info=delist_info)

    # IsST（池因子表现值，走 store 单点）
    isst = load_isst(con, spec, codes)
    con.close()

    # 对齐到 (date, code) 索引
    fdf = res.rename({"vt_symbol": "code", "datetime": "date"}).to_pandas()
    fdf["date"] = pd.to_datetime(fdf["date"])
    fdf = fdf.set_index(["date", "code"]).sort_index()
    fdf = fdf.loc[~fdf.index.duplicated(keep="last")]

    common = fdf.index.intersection(fwd.index)
    fdf, fwd = fdf.loc[common], fwd.loc[common]

    st = isst.reindex(common).fillna(0).astype(bool)
    exclude = st | ~fwd.notna() | (fwd == -1.0)
    fdf, fwd = fdf.loc[~exclude], fwd.loc[~exclude]

    fdf = fdf.loc[(fdf.index.get_level_values("date") >= pd.Timestamp(TRAIN_START))
                  & (fdf.index.get_level_values("date") < pd.Timestamp(TRAIN_END))]

    dates = fdf.index.get_level_values("date").unique().sort_values()
    F = fdf[BASELINE_FACTORS].values.astype(np.float64)
    F[~np.isfinite(F)] = np.nan
    date_arr = fdf.index.get_level_values("date").values
    uniq, start_idx, counts = np.unique(date_arr, return_index=True, return_counts=True)
    label_arr = fwd.loc[fdf.index].values

    ics = {f: [] for f in BASELINE_FACTORS}
    f_idx = {f: i for i, f in enumerate(BASELINE_FACTORS)}
    for g in range(len(uniq)):
        s, c = start_idx[g], counts[g]
        if c < MIN_STOCKS_PER_DATE:
            continue
        l_vals = label_arr[s:s + c]
        f_block = F[s:s + c]
        for f in BASELINE_FACTORS:
            ic = _rank_ic_np(f_block[:, f_idx[f]], l_vals)
            if not np.isnan(ic):
                ics[f].append(ic)

    metrics = {}
    for f in BASELINE_FACTORS:
        arr = np.array(ics[f])
        mean, std = float(arr.mean()), float(arr.std())
        metrics[f] = {
            "ic_mean": mean,
            "ic_std": std,
            "icir": mean / std if std > 0 else float("nan"),
            "pos_rate": float((arr > 0).mean()),
            "n_dates": int(len(arr)),
        }
    metrics["_meta"] = {
        "pool": spec.name,
        "universe": "pool_snapshots union (2026-08-25)",
        "train_window": [TRAIN_START, TRAIN_END],
        "horizon": HORIZON,
        "data_max_date": data_max_date,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluator regression gate")
    parser.add_argument("--init", action="store_true",
                        help="freeze/refresh the reference values")
    parser.add_argument("--pool", default=None,
                        help="目标池（默认 env QUANTLAB_POOL / 微盘；每池独立参考值文件）")
    args = parser.parse_args()
    spec = get_pool(args.pool)
    ref_path = _ref_path(spec)

    t0 = time.time()
    metrics = compute_metrics(spec)

    if args.init:
        ref = {
            "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tolerances": {"ic_mean_abs": IC_MEAN_TOL, "icir_abs": ICIR_TOL},
            **metrics,
        }
        ref_path.write_text(json.dumps(ref, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"Reference frozen -> {ref_path}")
        for f in BASELINE_FACTORS:
            m = metrics[f]
            print(f"  {f:<12} ic_mean={m['ic_mean']:+.4f}  icir={m['icir']:+.3f}"
                  f"  pos={m['pos_rate']:.2f}  n={m['n_dates']}")
        return

    if not ref_path.exists():
        print(f"ERROR: {ref_path} not found. Run with --init first.")
        sys.exit(2)

    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    print(f"reference frozen at {ref.get('frozen_at')}, "
          f"data_max_date={ref.get('_meta', {}).get('data_max_date')}")
    print(f"current  data_max_date={metrics['_meta']['data_max_date']}")
    print(f"\n{'factor':<12} {'ic_mean':>16} {'icir':>18} {'n':>6}")
    ok = True
    for f in BASELINE_FACTORS:
        m, r = metrics[f], ref.get(f)
        if r is None:
            print(f"{f:<12}  MISSING IN REFERENCE")
            ok = False
            continue
        d_mean = m["ic_mean"] - r["ic_mean"]
        d_icir = m["icir"] - r["icir"]
        flag = ""
        if abs(d_mean) > IC_MEAN_TOL or abs(d_icir) > ICIR_TOL:
            flag = "  <-- BREACH"
            ok = False
        print(f"{f:<12} {m['ic_mean']:+.4f} ({d_mean:+.4f})"
              f" {m['icir']:+.3f} ({d_icir:+.3f}) {m['n_dates']:>6}{flag}")

    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}  ({time.time() - t0:.1f}s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
