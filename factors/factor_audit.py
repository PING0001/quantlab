# -*- coding: utf-8 -*-
"""dev 工具：已有因子画像审计（方案 A，2026-08-23 用户批准的审查计划）。

对筛选池全部因子产出健康报告：
- IC 四标签（6d/20d/open2d/gap1d，挖掘窗 2020~2025-06，与训练对齐）
- ICIR、逐年 IC、近期衰减（2024H2~2025H1 vs 2020~2023）
- 全池 max 相关 + 前 3 对手
- 口径标注（静态表：遗留已知项）
- 按健康度排序

Usage: python factors/factor_audit.py [--pool | --selected]
输出: data/factor_audit_report.json + 控制台表格
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DB_PATH, get_pool_codes, SELECTED_FACTORS, POOL_NAME
from factors.select_factors import _rank_ic_np, MIN_STOCKS_PER_DATE
from factors.registry import lineage, MODEL_FACTORS
from strategies.labels import compute_median_open

MINE_START = pd.Timestamp("2020-01-01")
MINE_END = pd.Timestamp("2025-06-01")
# gap1d 标签是 open[T+1]/close[T]−1（close 锚）；其余 next_open 锚
LABELS = {"6d": (4, 6), "20d": (16, 20), "open2d": (2, 2)}
BASELINE = "next_open"

# 口径标注（静态）：已知遗留项，2026-08-22/23 审计结论
CAVEATS = {
    "sw_l3": "行业快照无时点性（当前分类应用于全历史）",
    "LimitUpCnt_20d": "涨停为 |ret|>=9.5% 近似口径（qfq 收益率）",
    "LimitUpStreakMax_60d": "同 LimitUpCnt 近似口径",
    "WinnerRate": "筹码 2018 起覆盖，此前 NULL",
    "CostPosition": "筹码 2018 起覆盖",
    "ChipDispersion": "筹码 2018 起覆盖",
    "ChipSkew": "筹码 2018 起覆盖",
    "DaysToDelivery": "日历因子：同日同值，横截面无独立 IC（未入模）",
    "DaysToNextTrading": "日历因子：同日同值（未入模）",
    "Turnover_3d": "依赖 daily_basic circ_mv 按日 join",
    "Turnover_3d_ratio": "依赖 daily_basic circ_mv 按日 join",
    "IsST": "namechange 区间语义（2026-08-20 修复版）",
    "StockIndexCorr_20d": "池等权收益用当前池成员构造（成员资格含幸存者成分）",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", action="store_true",
                    help="只审当前 selected_*.json 入模清单（缺省审全池）")
    args = ap.parse_args()

    codes = get_pool_codes()
    ph = ",".join(["?"] * len(codes))
    con = duckdb.connect(str(DB_PATH), read_only=True)
    k = con.execute(
        f"SELECT code, date, open, close FROM daily_kline WHERE code IN ({ph}) "
        f"ORDER BY code, date", codes).fetchdf()
    fv = con.execute(
        f"SELECT * FROM factor_values WHERE code IN ({ph})", codes).fetchdf()
    con.close()
    k["date"] = pd.to_datetime(k["date"])
    fv["date"] = pd.to_datetime(fv["date"])

    if args.selected:
        factor_cols = []
        for m in ("6d", "20d", "open2d"):
            p = Path(__file__).resolve().parent / f"selected_{POOL_NAME}_{m}.json"
            factor_cols += json.loads(p.read_text())["selected_factors"]
        factor_cols = sorted(set(factor_cols) & set(fv.columns))
    else:
        factor_cols = [c for c in SELECTED_FACTORS if c in fv.columns]
        # 模型因子（mf_ 前缀）自动纳入审计——它们与公式因子同位
        factor_cols += sorted(c for c in fv.columns
                              if c.startswith("mf_") and c not in factor_cols)
    print(f"审计 {len(factor_cols)} 个因子 | 窗口 {MINE_START.date()}~{MINE_END.date()}")

    # ---- 标签 ----
    labels = {}
    for name, (s_, e_) in LABELS.items():
        labels[name] = compute_median_open(k, start_day=s_, end_day=e_, baseline=BASELINE)
    # gap1d：close 锚的隔夜跳空（compute_median_open 的 (1,1)+close 组合）
    labels["gap1d"] = compute_median_open(k, start_day=1, end_day=1, baseline="close")

    fv = fv.set_index(["code", "date"]).sort_index()
    is_st = fv["IsST"].astype(bool) if "IsST" in fv.columns else None

    # ---- 逐日横截面 IC（四标签）----
    print("computing daily ICs ...")
    df = fv[factor_cols].copy()
    if is_st is not None:
        df = df[~is_st.reindex(df.index, fill_value=False)]

    ic_records = []  # (factor, label, date, ic)
    for lab_name, lab in labels.items():
        j = df.join(lab.rename("__y__"), how="inner")
        j = j[(j.index.get_level_values("date") >= MINE_START) &
              (j.index.get_level_values("date") < MINE_END)]
        for dt, ch in j.groupby(level="date", sort=True):
            y = ch["__y__"].values
            for c in factor_cols:
                ic = _rank_ic_np(ch[c].values, y)
                if not np.isnan(ic):
                    ic_records.append((c, lab_name, dt, ic))
    ic_df = pd.DataFrame(ic_records, columns=["factor", "label", "date", "ic"])

    # ---- 全池相关（逐日 Spearman 平均，成对 NaN 处理）----
    print("computing full-pool correlations ...")
    n_dates = 0
    corr_acc = {(a, b): [] for i, a in enumerate(factor_cols)
                for b in factor_cols[i + 1:]}
    sub = df[(df.index.get_level_values("date") >= MINE_START) &
             (df.index.get_level_values("date") < MINE_END)]
    for dt, ch in sub.groupby(level="date", sort=True):
        if len(ch) < MIN_STOCKS_PER_DATE:
            continue
        n_dates += 1
        corr = ch.rank().corr(min_periods=80)
        for i, a in enumerate(factor_cols):
            for b in factor_cols[i + 1:]:
                v = corr.loc[a, b]
                if pd.notna(v):
                    corr_acc[(a, b)].append(v)
    pair_mean = {k2: float(np.mean(v)) for k2, v in corr_acc.items() if v}
    maxcorr = {}
    for i, a in enumerate(factor_cols):
        partners = []
        for b in factor_cols:
            if b == a:
                continue
            key = (a, b) if (a, b) in pair_mean else (b, a)
            if key in pair_mean:
                partners.append((b, pair_mean[key]))
        partners.sort(key=lambda kv: -abs(kv[1]))
        maxcorr[a] = partners[:3]

    # ---- 汇总画像 ----
    rows = []
    for c in factor_cols:
        row = {"factor": c, "layer": lineage(c), "caveat": CAVEATS.get(c, "")}
        for lab_name in list(LABELS) + ["gap1d"]:
            s = ic_df[(ic_df["factor"] == c) & (ic_df["label"] == lab_name)]
            if s.empty:
                row[f"ic_{lab_name}"] = np.nan
                continue
            g = s.set_index("date")["ic"]
            row[f"ic_{lab_name}"] = round(g.mean(), 4)
            if lab_name in ("6d", "20d"):
                yearly = g.groupby(g.index.year).mean()
                early = yearly.loc[2020:2023].mean()
                recent = g[(g.index >= pd.Timestamp("2024-07-01"))].mean()
                row[f"icir_{lab_name}"] = round(
                    g.mean() / g.std() * np.sqrt(len(g)), 1) if g.std() > 0 else np.nan
                row[f"decay_{lab_name}"] = round(
                    recent / early, 2) if early and abs(early) > 1e-4 else np.nan
                row[f"yearly_{lab_name}"] = {int(y): round(v, 4)
                                             for y, v in yearly.items()}
        mc = maxcorr.get(c, [])
        row["max_corr"] = round(mc[0][1], 3) if mc else np.nan
        row["corr_vs"] = mc[0][0] if mc else ""
        row["corr_top3"] = "; ".join(f"{b}={v:+.2f}" for b, v in mc)
        # 健康度（粗排序键）：|IC_20d| 与 |IC_6d| 主贡献，衰减惩罚，冗余惩罚
        ic20 = abs(row.get("ic_20d") or 0)
        ic6 = abs(row.get("ic_6d") or 0)
        dec = row.get("decay_20d")
        dec_pen = (1 - min(abs(dec), 2)) if dec and np.isfinite(dec) and abs(dec) < 2 else 0.3
        red_pen = 1 - min(abs(row.get("max_corr") or 0), 1)
        row["_health"] = round((ic20 * 2 + ic6) * dec_pen * red_pen, 5)
        rows.append(row)

    rep = pd.DataFrame(rows).sort_values("_health", ascending=False)
    rep_out = rep.drop(columns=["_health"])
    out = Path(__file__).resolve().parents[1] / "data" / "factor_audit_report.json"
    out.write_text(json.dumps(
        {"generated": str(date.today()), "window": f"{MINE_START.date()}~{MINE_END.date()}",
         "n_dates_ic": int(n_dates), "factors": rep_out.to_dict(orient="records")},
        ensure_ascii=False, indent=2), encoding="utf-8")

    show_cols = ["factor", "layer", "ic_20d", "ic_6d", "ic_open2d", "ic_gap1d",
                 "icir_20d", "decay_20d", "max_corr", "corr_vs", "caveat"]
    # 控制台截断 layer（模型因子全名单很长），JSON 报告保留全文
    show = rep_out[show_cols].copy()
    show["layer"] = show["layer"].where(
        show["layer"].str.len() <= 40, show["layer"].str.slice(0, 39) + "…")
    print(f"\n按健康度排序（前 25 / 共 {len(rep)}）：")
    print(show.head(25).to_string(index=False))
    print(f"\n最弱 15 个：")
    print(show.tail(15).to_string(index=False))
    print(f"\n报告: {out}")


if __name__ == "__main__":
    main()
