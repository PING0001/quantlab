# -*- coding: utf-8 -*-
"""挖矿三件套合一（2026-08-25：原 factor_audit.py / factor_contribution.py /
test_new_factors.py 三文件合并；行为不变，入口改为子命令）。

    python -m factors.mining audit [--selected]          # A: 因子画像审计
    python -m factors.mining contribution [--fold F*]    # B1: gain + permutation 贡献
    python -m factors.mining batch [--start --end]       # C: 候选因子批测

纪律（AGENTS.md）：
- 加因子看 train-test 泛化缺口；冗余判定对全池取 max
  （<0.75 增量 / >0.95 冗余，用户标准）；强因子替换弱因子优先于堆加
- audit 的画像：四标签 IC/ICIR/衰减/全池 max 相关/口径标注
- contribution 判读：gain 份额低 + |ΔIC| 小 = 搭便车者；单因子 IC 高但
  ΔIC≈0 = 对模型无增量；IC 低但 ΔIC 大 = 交互型因子
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

from config import DB_PATH, SELECTED_FACTORS, FOLDS, get_fold
from pools.spec import get_pool
from factors.select_factors import _rank_ic_np, MIN_STOCKS_PER_DATE
from dataset import load_factors, load_kline, load_industry_sw_l3
from strategies.lgb import LGBStrategy
from strategies.labels import compute_median_open
from pools.membership import union_codes
from factors import store


def _layer(col: str) -> str:
    """因子分层标注（原 factors/registry.py lineage 的轻量替代：
    mf_ 注册表机制已删，gb_/nn_ 构建脚本自持列所有权）。"""
    if col.startswith(("gb_", "nn_", "mf_")):
        return f"模型因子（{col.split('_')[0]} 族，构建脚本所有）"
    return "公式因子"


# ============================================================================
# A: 因子画像审计（原 factor_audit.py）
# ============================================================================

AUDIT_MINE_START = pd.Timestamp("2020-01-01")
AUDIT_MINE_END = pd.Timestamp("2025-06-01")
# gap1d 标签是 open[T+1]/close[T]−1（close 锚）；其余 next_open 锚
AUDIT_LABELS = {"6d": (4, 6), "20d": (16, 20), "open2d": (2, 2)}
AUDIT_BASELINE = "next_open"

# 口径标注（静态）：已知遗留项，2026-08-22/23 审计结论
AUDIT_CAVEATS = {
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


def run_audit(args) -> None:
    """对筛选池全部因子产出健康报告：IC 四标签/ICIR/逐年/衰减/全池 max 相关/口径。"""
    spec = get_pool(args.pool)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    codes = union_codes(con=con, pool=spec.name)
    ph = ",".join(["?"] * len(codes))
    k = con.execute(
        f"SELECT code, date, open, close FROM daily_kline WHERE code IN ({ph}) "
        f"ORDER BY code, date", codes).fetchdf()
    fv = store.load_panel(con, spec, codes=codes)
    con.close()
    k["date"] = pd.to_datetime(k["date"])
    fv["date"] = pd.to_datetime(fv["date"])

    if args.selected:
        factor_cols = []
        for m in ("6d", "20d", "open2d"):
            p = Path(__file__).resolve().parent / f"selected_{spec.name}_{m}.json"
            factor_cols += json.loads(p.read_text())["selected_factors"]
        factor_cols = sorted(set(factor_cols) & set(fv.columns))
    else:
        factor_cols = [c for c in SELECTED_FACTORS if c in fv.columns]
        # 模型因子（gb_/nn_ 前缀）自动纳入审计--它们与公式因子同位
        factor_cols += sorted(c for c in fv.columns
                              if c.startswith(("gb_", "nn_")) and c not in factor_cols)
    print(f"审计 {len(factor_cols)} 个因子 | 窗口 {AUDIT_MINE_START.date()}~{AUDIT_MINE_END.date()}")

    # ---- 标签 ----
    labels = {}
    for name, (s_, e_) in AUDIT_LABELS.items():
        labels[name] = compute_median_open(k, start_day=s_, end_day=e_, baseline=AUDIT_BASELINE)
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
        j = j[(j.index.get_level_values("date") >= AUDIT_MINE_START) &
              (j.index.get_level_values("date") < AUDIT_MINE_END)]
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
    sub = df[(df.index.get_level_values("date") >= AUDIT_MINE_START) &
             (df.index.get_level_values("date") < AUDIT_MINE_END)]
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
        row = {"factor": c, "layer": _layer(c), "caveat": AUDIT_CAVEATS.get(c, "")}
        for lab_name in list(AUDIT_LABELS) + ["gap1d"]:
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
    out = (Path(__file__).resolve().parents[1] / "data"
           / f"factor_audit_report_{spec.name}.json")
    out.write_text(json.dumps(
        {"generated": str(date.today()),
         "window": f"{AUDIT_MINE_START.date()}~{AUDIT_MINE_END.date()}",
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


# ============================================================================
# B1: 因子边际贡献（原 factor_contribution.py）
# ============================================================================

CONTRIB_MODELS = ["20d", "6d", "open2d"]
CONTRIB_TEST_START = pd.Timestamp("2025-06-01")
CONTRIB_TEST_END = pd.Timestamp("2026-06-01")
CONTRIB_RNG = np.random.default_rng(42)


def _daily_ic(pred: pd.Series, y: pd.Series) -> float:
    df = pd.DataFrame({"p": pred, "y": y}).dropna()
    ics = []
    for dt, ch in df.groupby(level="date", sort=True):
        ic = _rank_ic_np(ch["p"].values, ch["y"].values)
        if not np.isnan(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else np.nan


def run_contribution(args) -> None:
    """对当前三个入模模型（冻结的 joblib，不重训）：gain + 日内截面 permutation ΔIC。"""
    if args.fold:
        ts, te = (pd.Timestamp(x) for x in get_fold(args.fold))
    else:
        ts, te = CONTRIB_TEST_START, CONTRIB_TEST_END
    tag = f" | fold={args.fold}" if args.fold else ""

    spec = get_pool(args.pool)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    factors_raw = load_factors(con, spec)
    kline = load_kline(con, spec)
    sw_l3, _ = load_industry_sw_l3(con)
    con.close()

    y_all = {}
    for m, (s_, e_) in {"20d": (16, 20), "6d": (4, 6), "open2d": (2, 2)}.items():
        y_all[m] = compute_median_open(kline, start_day=s_, end_day=e_, baseline="next_open")

    report = {}
    for m in CONTRIB_MODELS:
        model_path = spec.lgb_model_path(m, fold=args.fold)
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
        base_ic = _daily_ic(base_pred[pcol], yt)
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
                lambda s: pd.Series(CONTRIB_RNG.permutation(s.values), index=s.index))
            Xp[f] = shuffled.values
            pred = strategy.predict(Xp)
            ic = _daily_ic(pred[pcol], yt)
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
        out = (Path(__file__).resolve().parents[1] / "data" / "folds" / args.fold /
               f"factor_contribution_report_{spec.name}.json")
    else:
        out = (Path(__file__).resolve().parents[1] / "data"
               / f"factor_contribution_report_{spec.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"generated": str(date.today()), "fold": args.fold, **report},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告: {out}")


# ============================================================================
# C: 候选因子批测（原 test_new_factors.py）
# ============================================================================

BATCH_LABELS = {
    "6d": (4, 6),
    "20d": (16, 20),
}
BATCH_BASELINE = "next_open"


def build_candidates(k: pd.DataFrame) -> pd.DataFrame:
    """k: index (code,date) 已按 code,date 排序，含 open/high/low/close/volume/amount/prev_close/circ_mv。"""
    g = k.groupby("code", sort=False)
    ret = k["close"] / k["prev_close"] - 1.0
    gap = k["open"] / k["prev_close"] - 1.0
    intr = k["close"] / k["open"] - 1.0
    out = pd.DataFrame(index=k.index)
    out["LimitUpCnt_20d"] = g["ret_flag_up"].transform(lambda s: s.rolling(20).sum())
    out["LimitDownCnt_20d"] = g["ret_flag_dn"].transform(lambda s: s.rolling(20).sum())
    out["UpLimitDist"] = (1.1 * k["prev_close"] - k["close"]) / k["close"]
    out["PostHighDrawdown_10d"] = g.apply(
        lambda d: (d["close"] / d["high"].rolling(5).max() - 1.0).rolling(10).min()
    ).reset_index(level=0, drop=True).reindex(k.index)
    out["OvernightMean_20d"] = g.apply(
        lambda d: (d["open"] / d["prev_close"] - 1.0).rolling(20).mean()
    ).reset_index(level=0, drop=True).reindex(k.index)
    out["OvernightMean_5d"] = out["OvernightMean_20d"].copy()  # 占位，下方重算
    out["OvernightMean_5d"] = g.apply(
        lambda d: (d["open"] / d["prev_close"] - 1.0).rolling(5).mean()
    ).reset_index(level=0, drop=True).reindex(k.index)
    gap_std = g.apply(lambda d: (d["open"] / d["prev_close"] - 1.0).rolling(20).std()
                      ).reset_index(level=0, drop=True).reindex(k.index)
    intr_std = g.apply(lambda d: (d["close"] / d["open"] - 1.0).rolling(20).std()
                       ).reset_index(level=0, drop=True).reindex(k.index)
    out["ONIDVolRatio_20d"] = gap_std / intr_std.replace(0, np.nan)
    out["OvernightSkew_60d"] = g.apply(
        lambda d: (d["open"] / d["prev_close"] - 1.0).rolling(60).skew()
    ).reset_index(level=0, drop=True).reindex(k.index)
    out["IntradaySkew_60d"] = g.apply(
        lambda d: (d["close"] / d["open"] - 1.0).rolling(60).skew()
    ).reset_index(level=0, drop=True).reindex(k.index)
    ext = (k["close"] / k["open"] - 1.0).abs().ge(0.05).astype(float)
    out["ExtremeIntradayCnt_20d"] = ext.groupby(k.index.get_level_values("code"), sort=False).transform(
        lambda s: s.rolling(20).mean())
    neg = ret.where(ret < 0)
    pos = ret.where(ret > 0)
    gcode = k.index.get_level_values("code")
    out["DownsideVol_20d"] = neg.groupby(gcode, sort=False).transform(lambda s: s.rolling(20).std())
    up_std = pos.groupby(gcode, sort=False).transform(lambda s: s.rolling(20).std())
    out["UpDownVolRatio_20d"] = up_std / out["DownsideVol_20d"].replace(0, np.nan)
    out["MIN_5d"] = ret.groupby(gcode, sort=False).transform(lambda s: s.rolling(5).min())
    out["VolPriceCorr_20d"] = g.apply(
        lambda d: d["volume"].rolling(20).corr(d["close"].pct_change())
    ).reset_index(level=0, drop=True).reindex(k.index)
    out["MvChg_20d"] = g["circ_mv"].transform(lambda s: s.pct_change(20))
    out["LogClose"] = np.log(k["close"])

    # ---- 批次二（2026-08-22，因子优先战略：IC 为唯一迭代指标）----
    up60 = k["ret_flag_up"]
    out["LimitUpCnt_60d"] = up60.groupby(gcode, sort=False).transform(lambda s: s.rolling(60).sum())
    # 最长连续涨停 streak（60d 内）
    def _max_streak(s):
        streak = s.groupby((s != s.shift()).cumsum()).cumsum()
        return streak.rolling(60, min_periods=1).max()
    out["LimitUpStreakMax_60d"] = up60.groupby(gcode, sort=False).transform(_max_streak)
    neg20 = ret.where(ret < 0)
    pos20 = ret.where(ret > 0)
    out["DownsideVol_20d"] = neg20.groupby(gcode, sort=False).transform(
        lambda s: s.rolling(20, min_periods=5).std())
    dv = out["DownsideVol_20d"]
    uv = pos20.groupby(gcode, sort=False).transform(lambda s: s.rolling(20, min_periods=5).std())
    out["UpDownVolRatio_20d"] = uv / dv.replace(0, np.nan)
    # 池等权日收益（横截面均值），用于残差动量与个股-池协同
    pool_ret = ret.groupby(k.index.get_level_values("date")).transform("mean")
    idio = ret - pool_ret
    out["IdioMomentum_20d"] = idio.groupby(gcode, sort=False).transform(lambda s: s.rolling(20).sum())
    k2 = k.assign(_ret=ret, _pool=pool_ret, _g=gap, _i=intr)
    out["StockIndexCorr_20d"] = k2.groupby("code", sort=False).apply(
        lambda d: d["_ret"].rolling(20, min_periods=15).corr(d["_pool"])
    ).reset_index(level=0, drop=True).reindex(k.index)
    # 筹码变化（WinnerRateChg_20d / ChipSkewChg_20d）在 run_batch() 里 join 现有列后计算
    turn = k["amount"] / k["circ_mv"] / 10
    out["TurnoverSkew_20d"] = turn.groupby(gcode, sort=False).transform(
        lambda s: s.rolling(20, min_periods=10).skew())
    out["AmountConc_20d"] = (k["amount"] ** 2).groupby(gcode, sort=False).transform(
        lambda s: s.rolling(20).sum()) / k["amount"].groupby(gcode, sort=False).transform(
        lambda s: s.rolling(20).sum()) ** 2
    a5 = k["amount"].groupby(gcode, sort=False).transform(lambda s: s.rolling(5).mean())
    a60 = k["amount"].groupby(gcode, sort=False).transform(lambda s: s.rolling(60).mean())
    out["AmountShrink_5_60"] = a5 / a60
    vol_ma20 = k["volume"].groupby(gcode, sort=False).transform(lambda s: s.rolling(20).mean())
    spike = (k["volume"] > 2 * vol_ma20).astype(float)
    no_prior = spike.groupby(gcode, sort=False).transform(lambda s: s.rolling(5).sum()) <= 1
    out["FirstVolumeSpike_5d"] = (spike * no_prior.astype(float)).where(spike > 0)
    out["OvernightAR_60d"] = k2.groupby("code", sort=False).apply(
        lambda d: d["_g"].rolling(60, min_periods=30).corr(d["_g"].shift(1))
    ).reset_index(level=0, drop=True).reindex(k.index)
    out["ONIDCorr_60d"] = k2.groupby("code", sort=False).apply(
        lambda d: d["_g"].rolling(60, min_periods=30).corr(d["_i"])
    ).reset_index(level=0, drop=True).reindex(k.index)
    out["OpenPos_mean_20d"] = k.assign(_op=(k["open"] - k["low"]) / (k["high"] - k["low"]).replace(0, np.nan)).groupby(
        "code", sort=False)["_op"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    sg = gap.groupby(gcode, sort=False).transform(lambda s: s.abs().rolling(20).sum())
    si = intr.groupby(gcode, sort=False).transform(lambda s: s.abs().rolling(20).sum())
    out["OvernightShare_20d"] = sg / (sg + si)
    return out


def run_batch(args) -> None:
    """候选批测：IC + 与全池现有因子的逐日横截面 Spearman max 相关。"""
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    spec = get_pool(args.pool)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    codes = union_codes(con=con, pool=spec.name)
    ph = ",".join(["?"] * len(codes))
    k = con.execute(
        f"""SELECT k.code, k.date, k.open, k.high, k.low, k.close, k.volume, k.amount, b.circ_mv
            FROM daily_kline k LEFT JOIN daily_basic b
              ON k.code = replace(replace(b.code,'.SZ',''),'.SH','') AND k.date = b.date
            WHERE k.code IN ({ph}) AND k.date >= ? AND k.date <= ?
            ORDER BY k.code, k.date""",
        [*codes, str(start.date()), str(end.date())],
    ).fetchdf()
    existing = store.load_panel(con, spec, codes=codes)
    con.close()

    k["date"] = pd.to_datetime(k["date"])
    k = k.set_index(["code", "date"]).sort_index()
    k["prev_close"] = k.groupby("code", sort=False)["close"].shift(1)
    k = k.dropna(subset=["prev_close", "open", "close"])
    ret = k["close"] / k["prev_close"] - 1.0
    k["ret_flag_up"] = ret.ge(0.095).astype(float)
    k["ret_flag_dn"] = ret.le(-0.095).astype(float)

    print(f"kline rows: {len(k)}, stocks: {k.index.get_level_values('code').nunique()}")

    cand = build_candidates(k)
    cand = cand.replace([np.inf, -np.inf], np.nan)

    existing["date"] = pd.to_datetime(existing["date"])
    existing = existing.set_index(["code", "date"]).sort_index()
    is_st = existing["IsST"].astype(bool)

    # 筹码因子的时序变化（批次二：依赖 factor_values 现有列，join 后按 code 求 20 行差分）
    for src, name in (("WinnerRate", "WinnerRateChg_20d"), ("ChipSkew", "ChipSkewChg_20d")):
        if src in existing.columns:
            s = k[[]].join(existing[[src]], how="left")[src]
            cand[name] = s.groupby(level="code", sort=False).diff(20)
    cand = cand.replace([np.inf, -np.inf], np.nan)

    # 标签（与训练口径一致：next_open 锚，中位开）
    for name, (s_, e_) in BATCH_LABELS.items():
        lab = compute_median_open(k.reset_index(), start_day=s_, end_day=e_, baseline=BATCH_BASELINE)
        # compute_median_open 返回 MultiIndex (date, code) Series
        df = cand.join(lab.rename("__label__"))
        df = df.join(is_st)
        df = df[~df["IsST"].astype(bool)] if "IsST" in df else df
        df = df.drop(columns=["IsST"], errors="ignore").dropna(subset=["__label__"])
        print(f"\n===== label {name} (T+{s_}..T+{e_}, {BATCH_BASELINE}) =====")
        dates = df.index.get_level_values("date")
        cand_cols = [c for c in cand.columns]
        # 逐日 IC
        ics = {c: [] for c in cand_cols}
        for d, chunk in df.groupby(level="date", sort=True):
            L = chunk["__label__"].values
            for c in cand_cols:
                ic = _rank_ic_np(chunk[c].values, L)
                if not np.isnan(ic):
                    ics[c].append((d, ic))
        rows = []
        for c in cand_cols:
            s = pd.Series(dict(ics[c]))
            if len(s) == 0:
                continue
            yearly = s.groupby(s.index.year).mean().round(4).to_dict()
            rows.append({
                "factor": c, "mean_ic": round(s.mean(), 4),
                "icir": round(s.mean() / s.std() * np.sqrt(len(s)), 2) if s.std() > 0 else np.nan,
                "t": round(s.mean() / s.std() * np.sqrt(len(s)), 2) if s.std() > 0 else np.nan,
                "n_days": len(s),
                "yearly": yearly,
            })
        rep = pd.DataFrame(rows).sort_values("mean_ic", key=lambda x: x.abs(), ascending=False)
        print(rep.to_string(index=False))

        # 与现有因子的相关性（逐日横截面 Spearman 平均）。
        # 2026-08-22 用户方法论裁定：冗余判定必须对【全部】筛选池因子取
        # max 相关（<0.75 增量 / >0.95 冗余），不允许只对手挑子集
        corr_cols = [c for c in SELECTED_FACTORS if c in existing.columns and c not in cand_cols]
        print(f"\n--- 相关性（vs 全池 {len(corr_cols)} 个现有因子，逐日横截面 Spearman 平均，报 max）---")
        acc = {c: {ec: [] for ec in corr_cols} for c in cand_cols}
        merged_all = df[cand_cols].join(existing[corr_cols])
        for d, chunk in merged_all.groupby(level="date", sort=True):
            if len(chunk) < 100:
                continue
            corr = chunk.rank().corr(min_periods=80).loc[cand_cols, corr_cols]
            for c in cand_cols:
                for ec, v in corr.loc[c].dropna().items():
                    acc[c][ec].append(v)
        for c in cand_cols:
            s = pd.Series({ec: np.mean(v) for ec, v in acc[c].items() if v})
            if s.empty:
                print(f"  {c}: 无有效对照")
                continue
            top3 = sorted(s.items(), key=lambda kv: -abs(kv[1]))[:3]
            verdict = "增量" if abs(top3[0][1]) < 0.75 else ("冗余" if abs(top3[0][1]) > 0.95 else "边界")
            print(f"  {c:22s} max={top3[0][1]:+.3f} ({top3[0][0]}) [{verdict}]  "
                  f"2nd={top3[1][0]}={top3[1][1]:+.2f}" if len(top3) > 1 else "")


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="挖矿三件套：audit（画像）/ contribution（贡献）/ batch（批测）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_audit = sub.add_parser("audit", help="已有因子画像审计（四标签 IC/衰减/全池相关）")
    ap_audit.add_argument("--selected", action="store_true",
                          help="只审当前 selected_*.json 入模清单（缺省审全池）")

    ap_contrib = sub.add_parser("contribution",
                                help="因子边际贡献（gain + 日内截面 permutation ΔIC）")
    ap_contrib.add_argument("--fold", choices=sorted(FOLDS), default=None,
                            help="折模式：读折模型与折测试窗，报告写 data/folds/{fid}/")

    ap_batch = sub.add_parser("batch", help="候选因子批测（IC + 全池 max 相关）")
    ap_batch.add_argument("--start", default="2020-01-01")
    ap_batch.add_argument("--end", default="2025-06-01")

    ap.add_argument("--pool", default=None,
                    help="目标池（默认 env QUANTLAB_POOL / 微盘）")

    args = ap.parse_args()
    if args.cmd == "audit":
        run_audit(args)
    elif args.cmd == "contribution":
        run_contribution(args)
    elif args.cmd == "batch":
        run_batch(args)


if __name__ == "__main__":
    main()
