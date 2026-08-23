"""dev 工具：批量测试候选因子（原生 pandas 实现）的 IC 与相关性。

用法: python factors/test_new_factors.py [--start 2020-01-01] [--end 2025-06-01]

候选来自 document/llm_factor_mining/ 300 假设库（2026-08-22 首批 16 个，
现有数据可实现）。涨跌停为近似口径（|ret|>=0.095，前复权收益率），
与 strategies/labels.py 的 limit 检测同族近似。

口径对齐 factors/select_factors.py：窗口 [TRAIN_START, TEST_START)，
ST 剔除，逐日横截面 Spearman IC；相关性=逐日横截面 Spearman 对现有
factor_values 列的平均。
"""
import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DB_PATH, get_pool_codes
from factors.select_factors import _rank_ic_np, MIN_STOCKS_PER_DATE
from strategies.labels import compute_median_open

LABELS = {
    "6d": (4, 6),
    "20d": (16, 20),
}
BASELINE = "next_open"


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
    hi5 = g["high"].transform(lambda s: s.rolling(5).max())
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
    # 筹码变化（WinnerRateChg_20d / ChipSkewChg_20d）在 main() 里 join 现有列后计算
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2025-06-01")
    args = ap.parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)

    codes = get_pool_codes()
    ph = ",".join(["?"] * len(codes))
    con = duckdb.connect(str(DB_PATH), read_only=True)
    k = con.execute(
        f"""SELECT k.code, k.date, k.open, k.high, k.low, k.close, k.volume, k.amount, b.circ_mv
            FROM daily_kline k LEFT JOIN daily_basic b
              ON k.code = replace(replace(b.code,'.SZ',''),'.SH','') AND k.date = b.date
            WHERE k.code IN ({ph}) AND k.date >= ? AND k.date <= ?
            ORDER BY k.code, k.date""",
        [*codes, str(start.date()), str(end.date())],
    ).fetchdf()
    existing = con.execute(
        f"SELECT * FROM factor_values WHERE code IN ({ph})", codes
    ).fetchdf()
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
    for name, (s_, e_) in LABELS.items():
        lab = compute_median_open(k.reset_index(), start_day=s_, end_day=e_, baseline=BASELINE)
        # compute_median_open 返回 MultiIndex (date, code) Series
        df = cand.join(lab.rename("__label__"))
        df = df.join(is_st)
        df = df[~df["IsST"].astype(bool)] if "IsST" in df else df
        df = df.drop(columns=["IsST"], errors="ignore").dropna(subset=["__label__"])
        print(f"\n===== label {name} (T+{s_}..T+{e_}, {BASELINE}) =====")
        rows = []
        dates = df.index.get_level_values("date")
        uniq = dates.unique()
        cand_cols = [c for c in cand.columns]
        by_date = {d: None for d in []}
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
        # max 相关（<0.75 增量 / >0.95 冗余），不允许只对手挑子集——
        # UpLimitDist−0.87 vs Intraday_return、LogClose 0.93 vs SMA 均系
        # 全池检查才暴露
        from config import SELECTED_FACTORS as _POOL
        cand_cols = [c for c in cand.columns]
        corr_cols = [c for c in _POOL if c in existing.columns and c not in cand_cols]
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
        n_dates_corr = len([1 for _ in merged_all.groupby(level="date") if len(_) >= 100])
        for c in cand_cols:
            s = pd.Series({ec: np.mean(v) for ec, v in acc[c].items() if v})
            if s.empty:
                print(f"  {c}: 无有效对照")
                continue
            top3 = sorted(s.items(), key=lambda kv: -abs(kv[1]))[:3]
            verdict = "增量" if abs(top3[0][1]) < 0.75 else ("冗余" if abs(top3[0][1]) > 0.95 else "边界")
            print(f"  {c:22s} max={top3[0][1]:+.3f} ({top3[0][0]}) [{verdict}]  "
                  f"2nd={top3[1][0]}={top3[1][1]:+.2f}" if len(top3) > 1 else "")


if __name__ == "__main__":
    main()
