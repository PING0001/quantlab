# -*- coding: utf-8 -*-
"""
滚动 7 折 × 搭便车清洗版折验证（2026-08-23 用户裁定"精简版"）。

复现 v8 清单诞生配方中无主观判断的那一段——机械规则：贡献分析里
gain 份额为 0 且 permutation ΔIC 为 0（四舍五入到 1e-4）的因子视为
搭便车，从折清单剔除——在每折内部重演：

  1) 贡献分析（factor_contribution --fold，读折胖清单冻结模型）
  2) 机械清洗折清单（写回 factors/folds/{fid}/，保留 train_end 等字段，
     附 cleaned 溯源；运行前整树备份到 tmp/fat_fold_lists_backup/）
  3) 重训（run_lgb --fold，含 gap1d 复训幂等）+ fold_cv.leak_checks 断言
  4) market 语义回测（覆盖 folds/{fid}/ 的 market 产物，胖清单版原始
     回测 CSV 已随备份保存）
  5) 汇总逐折收益/夏普 vs 基准，对照胖清单版报告（data/fold_cv_report.json）

诚实边界（两处，报告 JSON 亦标注）：
- 清洗规则在折自己的测试窗上度量（与主窗口 v8 当时做法一致）——这是
  "配方复现"，不是纯 OOS 程序；
- 不做 6d/open2d 极限瘦身（主窗口 4/3 清单含人工裁定的尺寸实验）——
  本折验证的对象是"搭便车清洗后的方法论"，非精确 34/4/3 配方。

Usage:
    python fold_cv_slim.py --folds F1     # 单折冒烟
    python fold_cv_slim.py                # 7 折全量
    python fold_cv_slim.py --report-only  # 只汇总盘上折产物
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import POOL_NAME, FOLDS
from fold_cv import leak_checks, fold_metrics, run

PY = sys.executable
ROOT = Path(__file__).resolve().parent
CLEAN_MODELS = ("20d", "6d", "open2d")


def backup_fat_artifacts() -> None:
    """首次运行前备份胖清单版折清单与折回测 CSV（可重复运行，幂等）。"""
    stamp = time.strftime("%Y%m%d")
    for src, name in ((ROOT / "factors" / "folds", "fat_fold_lists_backup"),
                      (ROOT / "backtest" / POOL_NAME / "folds", "fat_fold_bt_backup")):
        dst = ROOT / "tmp" / f"{name}_{stamp}"
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)
            print(f"  备份 {src.name} -> {dst}")


def clean_fold_lists(fid: str) -> dict:
    """机械清洗折清单：gain==0 且 ic_drop==0 者剔除，写回原 json。"""
    rep = json.loads(
        (ROOT / "data" / "folds" / fid / "factor_contribution_report.json").read_text())
    dropped = {}
    for m in CLEAN_MODELS:
        jp = ROOT / "factors" / "folds" / fid / f"selected_{POOL_NAME}_{m}.json"
        d = json.loads(jp.read_text())
        fat = list(d["selected_factors"])
        rows = {r["factor"]: r for r in rep[m]["rows"]}
        drop = [f for f in fat
                if f in rows and rows[f]["gain"] == 0.0 and rows[f]["ic_drop"] == 0.0]
        slim = [f for f in fat if f not in set(drop)]
        if len(slim) < 5:
            print(f"  WARNING {fid}/{m}: 清洗后仅剩 {len(slim)} 因子，保留原清单不动")
            continue
        d["selected_factors"] = slim
        d["cleaned"] = {
            "source": "fold_cv_slim", "fold": fid,
            "rule": "gain==0 and ic_drop==0 (rounded 1e-4)",
            "dropped": drop, "n_before": len(fat), "n_after": len(slim),
        }
        jp.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        dropped[m] = {"n_before": len(fat), "n_after": len(slim), "dropped": drop}
        print(f"  {fid}/{m}: {len(fat)} -> {len(slim)}（剔除 {len(drop)}: "
              f"{', '.join(drop) if drop else '无'}）")
    return dropped


def main():
    parser = argparse.ArgumentParser(description="搭便车清洗版折验证驱动")
    parser.add_argument("--folds", default=",".join(FOLDS), help="逗号分隔折号（默认全部）")
    parser.add_argument("--report-only", action="store_true", help="只汇总，不跑折")
    args = parser.parse_args()

    fids = [x.strip() for x in args.folds.split(",") if x.strip()]
    for fid in fids:
        assert fid in FOLDS, f"unknown fold {fid}"

    clean_info: dict[str, dict] = {}
    if args.report_only:
        # 汇总模式：清洗记录从各折清单的 cleaned 字段读取
        for fid in fids:
            ci = {}
            for m in CLEAN_MODELS:
                jp = ROOT / "factors" / "folds" / fid / f"selected_{POOL_NAME}_{m}.json"
                c = json.loads(jp.read_text()).get("cleaned")
                if c:
                    ci[m] = {"n_before": c["n_before"], "n_after": c["n_after"],
                             "dropped": c["dropped"]}
            clean_info[fid] = ci
    else:
        backup_fat_artifacts()
        log_path = ROOT / "data" / "fold_cv_slim_run.log"
        t_all = time.time()
        for fid in fids:
            print(f"\n===== {fid} =====")
            run([PY, "-m", "factors.factor_contribution", "--fold", fid], log_path)
            clean_info[fid] = clean_fold_lists(fid)
            run([PY, "run_lgb.py", "--model", "all", "--fold", fid], log_path)
            leak_checks(fid)
            print("    leak checks passed")
            run([PY, "-m", "backtest.run_lgb", "--fold", fid, "--exec", "market"], log_path)
        print(f"\n全部折完成，总耗时 {(time.time() - t_all) / 60:.1f} 分钟")

    # ---- 汇总 ----
    rows = {fid: fold_metrics(fid, "market") for fid in fids}
    df = pd.DataFrame(rows).T
    report = {
        "method": "搭便车清洗版折验证（gain==0 且 perm ΔIC==0 剔除；无尺寸实验瘦身）",
        "caveats": [
            "清洗规则在折自己的测试窗上度量（与主窗口 v8 配方一致，非纯 OOS）",
            "6d/open2d 未做极限瘦身，验证对象是清洗版方法论而非精确 34/4/3 配方",
        ],
        "clean_info": clean_info,
        "per_fold": rows,
        "mean_total_return": float(df["total_return"].mean()),
        "worst_total_return": float(df["total_return"].min()),
        "mean_sharpe": float(df["sharpe"].mean()),
        "worst_sharpe": float(df["sharpe"].min()),
        "mean_benchmark_return": float(df["benchmark_return"].mean()),
    }
    out_path = ROOT / "data" / "fold_cv_slim_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    print(f"\n{'=' * 78}\n  SLIM（搭便车清洗版）market — 各折汇总\n{'=' * 78}")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(df.to_string())
    print(f"\n  跨折: 平均收益 {report['mean_total_return']:+.1%}"
          f" | 最差折 {report['worst_total_return']:+.1%}"
          f" | 平均夏普 {report['mean_sharpe']:.2f}"
          f" | 平均基准 {report['mean_benchmark_return']:+.1%}"
          f" | 超额为正折数 "
          f"{sum(1 for r in rows.values() if r['total_return'] > r.get('benchmark_return', -9))}/{len(fids)}")
    print(f"\n报告: {out_path}")


if __name__ == "__main__":
    main()
