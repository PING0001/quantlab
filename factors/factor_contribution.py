# -*- coding: utf-8 -*-
"""dev 工具：因子边际贡献粗筛（方案 B1，2026-08-23 审查计划）。

对当前三个入模模型（冻结的 joblib，不重训）：
- gain importance：booster_.feature_importance('gain') 归一化份额
- permutation importance：测试窗逐因子在【日内截面内】打乱，看模型测试 rank IC 掉多少
  （日内打乱保留日期结构，度量的正是"截面信息"的贡献）

判读：gain 份额低 + |ΔIC| 小 = 搭便车者（B2 精测候选）；
      单因子 IC 高但 ΔIC≈0 = 对模型无增量；IC 低但 ΔIC 大 = 交互型因子。

Usage: python factors/factor_contribution.py
输出: data/factor_contribution_report.json + 控制台表
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DB_PATH, POOL_NAME
from factors.select_factors import _rank_ic_np
from factors.select_factors import load_factors as _sf_load_factors
from run_lgb import load_kline, load_industry_sw_l3
from strategies.lgb import LGBStrategy
from strategies.labels import compute_median_open
from config import DB_PATH, POOL_NAME, FOLDS, get_fold, get_lgb_model_path

MODELS = ["20d", "6d", "open2d"]
TEST_START = pd.Timestamp("2025-06-01")
TEST_END = pd.Timestamp("2026-06-01")
RNG = np.random.default_rng(42)


def daily_ic(pred: pd.Series, y: pd.Series) -> float:
    df = pd.DataFrame({"p": pred, "y": y}).dropna()
    ics = []
    for dt, ch in df.groupby(level="date", sort=True):
        ic = _rank_ic_np(ch["p"].values, ch["y"].values)
        if not np.isnan(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else np.nan


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="因子贡献分析（gain + 日内截面 permutation ΔIC）")
    parser.add_argument("--fold", choices=sorted(FOLDS), default=None,
                        help="折模式：读折模型与折测试窗，报告写 data/folds/{fid}/")
    args = parser.parse_args()

    if args.fold:
        ts, te = (pd.Timestamp(x) for x in get_fold(args.fold))
    else:
        ts, te = TEST_START, TEST_END
    tag = f" | fold={args.fold}" if args.fold else ""

    con = duckdb.connect(str(DB_PATH), read_only=True)
    factors_raw = _sf_load_factors(con)
    kline = load_kline(con)
    sw_l3, _ = load_industry_sw_l3(con)
    con.close()

    y_all = {}
    for m, (s_, e_) in {"20d": (16, 20), "6d": (4, 6), "open2d": (2, 2)}.items():
        y_all[m] = compute_median_open(kline, start_day=s_, end_day=e_, baseline="next_open")

    report = {}
    for m in MODELS:
        model_path = get_lgb_model_path(m, fold=args.fold)
        strategy = LGBStrategy.load(model_path)
        fnames = list(strategy.factor_names)
        cols = [f for f in fnames if f != "sw_l3"]
        X = factors_raw[cols].copy()
        if "sw_l3" in fnames:
            idx_codes = X.index.get_level_values("code")
            X["sw_l3"] = idx_codes.map(sw_l3).fillna(-1).astype(int)

        mask = (X.index.get_level_values("date") >= ts) & \
               (X.index.get_level_values("date") <= te)
        Xt = X.loc[mask]
        yt = y_all[m].reindex(Xt.index)

        base_pred = strategy.predict(Xt)
        pcol = [c for c in base_pred.columns if c.startswith("pred_")][0]
        base_ic = daily_ic(base_pred[pcol], yt)
        print(f"\n===== {m}{tag}: {len(fnames)} factors, base test IC {base_ic:.4f} =====")

        # gain importance
        booster = strategy._models[list(strategy._models)[0]].booster_
        gain = np.array(booster.feature_importance(importance_type="gain"), dtype=float)
        gain_share = gain / gain.sum() if gain.sum() > 0 else gain
        gain_map = dict(zip(booster.feature_name(), gain_share))

        rows = []
        perm_cols = [f for f in fnames if f != "sw_l3"]
        dates = Xt.index.get_level_values("date")
        for f in perm_cols:
            Xp = Xt.copy()
            col = Xp[f].copy()
            # 日内截面内打乱（保留日期结构）
            shuffled = col.groupby(dates).transform(
                lambda s: pd.Series(RNG.permutation(s.values), index=s.index))
            Xp[f] = shuffled.values
            pred = strategy.predict(Xp)
            ic = daily_ic(pred[pcol], yt)
            rows.append({
                "factor": f,
                "gain": round(float(gain_map.get(f, 0.0)), 4),
                "ic_perm": round(ic, 4),
                "ic_drop": round(base_ic - ic, 4),
            })
        rep = pd.DataFrame(rows)
        rep["abs_drop"] = rep["ic_drop"].abs()
        rep = rep.sort_values("abs_drop", ascending=False)
        print(rep.head(15).to_string(index=False))
        print("--- 贡献最低 12（B2 嫌疑）---")
        print(rep.tail(12).to_string(index=False))
        report[m] = {"base_ic": round(base_ic, 4),
                     "rows": rep.drop(columns=["abs_drop"]).to_dict(orient="records")}

    if args.fold:
        out = Path(__file__).resolve().parents[1] / "data" / "folds" / args.fold / \
            "factor_contribution_report.json"
    else:
        out = Path(__file__).resolve().parents[1] / "data" / "factor_contribution_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"generated": str(date.today()), "fold": args.fold, **report},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告: {out}")


if __name__ == "__main__":
    main()
