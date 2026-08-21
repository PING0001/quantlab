# -*- coding: utf-8 -*-
"""
Rolling fold CV driver（2026-08-21 用户裁定：连续 7 折半年窗，训练起点锁 2020-01）。

每折独立全链，防两类泄漏：
- 标签前视：run_lgb 的 label_buffer 机制（复用，无新代码）
- 筛选泄漏：每折重跑 select_factors，窗口 [2020, 折 test_start)，训练强制读折专属清单

每折产出双执行语义回测：limit（收盘限价 v7 语义）/ market（开盘市价 v7mo 语义）。
汇总报告：每折×每语义指标 + 跨折平均 + 最差折，写 data/fold_cv_report.json。

全程 DB 只读，与夜间流水线无锁冲突。

Usage:
    python fold_cv.py --dry-run                 # 打印每折计划
    python fold_cv.py                           # 7 折全链 × 双语义
    python fold_cv.py --folds F1,F2 --exec market
    python fold_cv.py --skip-train              # 复用折产物只重跑回测
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (POOL_NAME, FOLDS, FOLD_TRAIN_START, get_fold,
                    get_backtest_dir, get_lgb_predictions_path,
                    get_lgb_predictions_meta_path)

PY = sys.executable
ROOT = Path(__file__).resolve().parent
RF = 0.025  # 与 backtest/run_lgb.py 同口径


def run(cmd: list[str], log_path: Path) -> None:
    t0 = time.time()
    with open(log_path, "a") as f:
        f.write(f"\n{'=' * 70}\n$ {' '.join(cmd)}\n")
        f.flush()
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"command failed (exit {r.returncode}): {' '.join(cmd)} — see {log_path}")
    print(f"    done in {time.time() - t0:.0f}s")


def leak_checks(fid: str) -> None:
    """折产物泄漏断言：筛选窗口终点=折 test_start；训练截止<test_start；预测范围⊆折窗。"""
    test_start, test_end = get_fold(fid)
    ts, te = pd.Timestamp(test_start), pd.Timestamp(test_end)
    for m in ("20d", "6d"):
        sj = ROOT / "factors" / "folds" / fid / f"selected_{POOL_NAME}_{m}.json"
        d = json.loads(sj.read_text())
        assert d["train_end"] == test_start, \
            f"{fid}/{m}: selected train_end={d['train_end']} != test_start={test_start}"
        meta = json.loads(get_lgb_predictions_meta_path(m, fold=fid).read_text())
        assert pd.Timestamp(meta["train_end"]) < ts, \
            f"{fid}/{m}: meta train_end={meta['train_end']} >= test_start"
        assert meta["test_start"] == test_start and meta["test_end"] == test_end, \
            f"{fid}/{m}: meta test window {meta['test_start']}~{meta['test_end']} != fold def"
        pred = pd.read_parquet(get_lgb_predictions_path(m, fold=fid))
        dts = pred.index.get_level_values("date")
        assert dts.min() >= ts and dts.max() <= te, \
            f"{fid}/{m}: predictions [{dts.min()}~{dts.max()}] outside fold window"


def fold_metrics(fid: str, exec_label: str) -> dict:
    """从折回测 CSV 计算：总收益/夏普/回撤/仓位/往返/单笔 t 值/胜率 + 基准。"""
    bt = get_backtest_dir(fold=fid)
    e = pd.read_csv(bt / f"equity_lgb_combined_daily_{exec_label}_rebalance.csv",
                    index_col=0, parse_dates=True)["Equity"]
    r = e.pct_change().dropna()
    sharpe = float((r.mean() - RF / 252) / r.std() * np.sqrt(242)) if r.std() > 0 else 0.0
    mdd = float(((e / e.cummax()) - 1).min())
    out = {"total_return": float(e.iloc[-1] / e.iloc[0] - 1), "sharpe": sharpe,
           "max_dd": mdd, "n_days": int(len(r))}

    tpath = bt / f"trades_lgb_combined_daily_{exec_label}_rebalance.csv"
    if tpath.exists():
        t = pd.read_csv(tpath, parse_dates=["date"], dtype={"code": str}).sort_values("date")
        trips, books = [], {}
        for row in t.itertuples():
            q = books.setdefault(row.code, [])
            if row.action == "BUY":
                q.append([row.date, row.price, row.shares])
            else:
                rem = row.shares
                while rem > 0 and q:
                    bd, bp, bs = q[0]
                    take = min(rem, bs)
                    trips.append((row.date - bd).days, row.price / bp - 1)
                    q[0][2] -= take
                    rem -= take
                    if q[0][2] == 0:
                        q.pop(0)
        rets = pd.Series([x[1] for x in trips])
        out.update({
            "n_trips": len(trips),
            "hold_days_median": float(pd.Series([x[0] for x in trips]).median()),
            "trip_mean": float(rets.mean()),
            "trip_t": float(rets.mean() / (rets.std() / np.sqrt(len(rets)))) if len(rets) > 2 else float("nan"),
            "win_rate": float((rets > 0).mean()),
        })
    bench = bt / "benchmark_combined.csv"
    if bench.exists():
        b = pd.read_csv(bench, index_col=0, parse_dates=True)["equity"]
        out["benchmark_return"] = float(b.iloc[-1] / b.iloc[0] - 1)
    return out


def main():
    parser = argparse.ArgumentParser(description="Rolling fold CV driver")
    parser.add_argument("--folds", default=",".join(FOLDS),
                        help="逗号分隔折号（默认全部 7 折）")
    parser.add_argument("--exec", choices=["both", "limit", "market"], default="both")
    parser.add_argument("--skip-train", action="store_true",
                        help="复用已有折筛选/模型/预测，只重跑回测与汇总")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fids = [x.strip() for x in args.folds.split(",") if x.strip()]
    for fid in fids:
        assert fid in FOLDS, f"unknown fold {fid}"
    execs = ["limit", "market"] if args.exec == "both" else [args.exec]
    log_path = ROOT / "data" / "fold_cv_run.log"

    print(f"Fold CV: {fids} | exec={execs} | train_start={FOLD_TRAIN_START}（扩张窗口）")
    for fid in fids:
        ts, te = get_fold(fid)
        n_years = (pd.Timestamp(ts) - pd.Timestamp(FOLD_TRAIN_START)).days / 365.25
        print(f"  {fid}: test {ts}~{te}（训练 {FOLD_TRAIN_START}~{ts} 前，≈{n_years:.1f} 年）")

    if args.dry_run:
        print("\n[dry-run] 每折将依次执行：")
        print("  1) python -m factors.select_factors --model 20d/6d --fold F*")
        print("  2) python run_lgb.py --model all --fold F*")
        print(f"  3) python -m backtest.run_lgb --fold F* --exec {'/'.join(execs)}")
        print(f"  产物：factors/folds/F*/、models/{POOL_NAME}/folds/F*/、data/folds/F*/、"
              f"backtest/{POOL_NAME}/folds/F*/")
        return

    t_all = time.time()
    for fid in fids:
        print(f"\n===== {fid} =====")
        if not args.skip_train:
            for m in ("20d", "6d"):
                run([PY, "-m", "factors.select_factors", "--model", m, "--fold", fid], log_path)
            run([PY, "run_lgb.py", "--model", "all", "--fold", fid], log_path)
        leak_checks(fid)
        print(f"    leak checks passed")
        for ex in execs:
            run([PY, "-m", "backtest.run_lgb", "--fold", fid, "--exec", ex], log_path)

    # ---- summary ----
    report = {}
    for ex in execs:
        rows = {}
        for fid in fids:
            rows[fid] = fold_metrics(fid, ex)
        df = pd.DataFrame(rows).T
        report[ex] = {
            "per_fold": rows,
            "mean_total_return": float(df["total_return"].mean()),
            "worst_total_return": float(df["total_return"].min()),
            "mean_sharpe": float(df["sharpe"].mean()),
            "worst_sharpe": float(df["sharpe"].min()),
            "mean_benchmark_return": float(df.get("benchmark_return", pd.Series(dtype=float)).mean())
            if "benchmark_return" in df else None,
        }

    out_path = ROOT / "data" / "fold_cv_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    for ex in execs:
        print(f"\n{'=' * 78}\n  {ex.upper()} 执行语义 — 各折汇总\n{'=' * 78}")
        df = pd.DataFrame({fid: report[ex]["per_fold"][fid] for fid in fids}).T
        with pd.option_context("display.float_format", "{:.3f}".format):
            print(df.to_string())
        agg = report[ex]
        print(f"\n  跨折: 平均收益 {agg['mean_total_return']:+.1%} | 最差折 {agg['worst_total_return']:+.1%}"
              f" | 平均夏普 {agg['mean_sharpe']:.2f} | 最差夏普 {agg['worst_sharpe']:.2f}"
              + (f" | 平均基准 {agg['mean_benchmark_return']:+.1%}"
                 if agg["mean_benchmark_return"] is not None else ""))
    print(f"\n报告: {out_path} | 总耗时 {(time.time() - t_all) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
