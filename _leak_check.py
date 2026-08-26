# -*- coding: utf-8 -*-
"""
_leak_check.py — 双回归 bench 泄漏断言（Task 9，2026-08-23 重写）。

旧版是三分类时代的一次性污染量化脚本；本版对当前双回归体系（v8：
open2d / 6d / 20d 三模型 + gap1d 独立实验模型）的盘上主窗口产物做系统性断言：

  C1 训练掩码终点   重算 buffered_train_end，断言 meta.train_end 一致；
                    末个纳入训练的日期（train_end 为排他上界）其标签最远
                    引用（T+label_window 末日）严格早于测试窗首日；
  C2 输出校准尾段   重算 run_lgb 校准块的 calib_dates（< train_end 的留出
                    尾段），断言全部 < train_end 且标签不触及测试期；
                    meta 含 calib_slope / calib_median；
  C3 训练样本排除   重算 IsST / 退市 / 次日开盘封板三掩码，复刻 train_model
                    的过滤链得到期望预测行集，断言 parquet 行集与之逐行一致
                    （三类污染观测 = 0）、行数与 meta.n_pred 一致、
                    日期范围 ⊆ [test_start, test_end]；
  C4 标签函数方向   compute_median_open 源码只含 shift(-d)（向后），附合成
                    面板功能对拍 + "窗口外未来价改动不影响标签"不变性。

训练协议常量（TRAIN_START/TEST_*/WARMUP/CALIB_TAIL_DAYS/label_buffer）全部
从 run_lgb / MODEL_CONFIGS 单源 import，不在本文件重复定义。

全部通过 exit 0；任一失败 exit 1 并打印明细。本工具是审计件，允许非零
退出——生产防断流是 forecast_display/generate_lgb.py 三级降级的职责。
折产物（factors/folds/）由 fold_cv.py 内置 leak_checks 覆盖，此处只查主窗口。

Usage:
    python _leak_check.py
"""
from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (DB_PATH, POOL_NAME, MODEL_CONFIGS,
                    get_lgb_predictions_path, get_lgb_predictions_meta_path)
from strategies.lgb import buffered_train_end
from strategies.labels import compute_median_open, compute_nextopen_limit_mask
from pools.membership import union_codes, member_mask

# 训练入口单源：常量与 delist 加载直接复用，防两处漂移
import run_lgb as trainer

_RESULTS: list[tuple[str, bool, str]] = []


def _check(name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(ok), detail))
    tag = "PASS" if ok else "FAIL"
    line = f"    [{tag}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)


# ---------------------------------------------------------------------------
# C4: 标签函数方向（不依赖 DB / 盘上产物，先跑）
# ---------------------------------------------------------------------------
def check_c4_label_direction() -> None:
    print("\n[C4] 标签函数 compute_median_open 只向后 shift ...")
    src = inspect.getsource(compute_median_open)
    args = [a.strip() for a in re.findall(r"\.shift\(([^)]*)\)", src)]
    _check("源码 shift 方向全为负", bool(args) and all(a.startswith("-") for a in args),
           f"{len(args)} 处 shift 调用: {args}")

    # 合成面板对拍：10 天 × 2 股（单股 groupby-apply 会塌缩索引，标签函数
    # 必须多股票帧调用——项目已知坑），000001 的 open/close 逐日可手算
    days = pd.date_range("2026-01-05", periods=10, freq="D")
    df = pd.concat([
        pd.DataFrame({"date": days, "code": "000001",
                      "open": [10., 11., 12., 13., 14., 15., 16., 17., 18., 19.],
                      "close": [20., 21., 22., 23., 24., 25., 26., 27., 28., 29.]}),
        pd.DataFrame({"date": days, "code": "000002",
                      "open": [30., 31., 32., 33., 34., 35., 36., 37., 38., 39.],
                      "close": [40., 41., 42., 43., 44., 45., 46., 47., 48., 49.]}),
    ], ignore_index=True)
    lab_no = compute_median_open(df, start_day=2, end_day=4, baseline="next_open")
    a_no = lab_no.xs("000001", level="code")
    exp_no = 13.0 / 11.0 - 1.0   # T=0: median(open[T+2..T+4]) / open[T+1] − 1
    _check("next_open 锚公式对拍 (T=0)",
           abs(float(a_no.iloc[0]) - exp_no) < 1e-12,
           f"label={float(a_no.iloc[0]):+.6f}, expect={exp_no:+.6f}")

    lab_cl = compute_median_open(df, start_day=2, end_day=4, baseline="close")
    a_cl = lab_cl.xs("000001", level="code")
    exp_cl = 13.0 / 20.0 - 1.0   # close 锚: median / close[T] − 1
    _check("close 锚公式对拍 (T=0)",
           abs(float(a_cl.iloc[0]) - exp_cl) < 1e-12,
           f"label={float(a_cl.iloc[0]):+.6f}, expect={exp_cl:+.6f}")

    # 不变性：改动 T+5 之后的开盘价（idx 8/9），前 4 日（窗口最远 T+4）
    # 标签必须不变，且后段至少一个变化（防测试空转）
    df2 = df.copy()
    df2.loc[[8, 9], "open"] = [990., 999.]   # 行 0-9 属 000001
    lab2 = compute_median_open(df2, start_day=2, end_day=4, baseline="next_open")
    a2 = lab2.xs("000001", level="code")
    head_same = np.allclose(a_no.iloc[:4].to_numpy(), a2.iloc[:4].to_numpy(),
                            equal_nan=True)
    tail_diff = not np.allclose(a_no.iloc[4:].to_numpy(), a2.iloc[4:].to_numpy(),
                                equal_nan=True)
    _check("窗口外未来价不变性", head_same and tail_diff,
           "T≤3 标签不变，T≥4 至少一处变化")


# ---------------------------------------------------------------------------
# C1-C3: 逐模型盘上产物断言
# ---------------------------------------------------------------------------
def replicate_model_panel(
    m: str,
    fv_idx: pd.MultiIndex,
    label: pd.Series,
    st_series: pd.Series,
    limit_mask: pd.Series,
    delist_series: pd.Series,
) -> pd.MultiIndex:
    """复刻 run_lgb.train_model 的 X 行过滤链，返回最终 (date, code) 索引。

    2026-08-24 语义变更后：链条 = factor 面板 ∩ 标签非 NaN → date >=
    TRAIN_START（**到此为止**）。ST/退市/封板/远引用越界只从训练集剔除
    （train_exclude），预测 parquet 保持全量面板行集。
    """
    common = fv_idx.intersection(label.index)
    lab = label.reindex(common)
    keep = lab.notna().to_numpy()
    idx = common[keep]

    idx_date = idx.get_level_values("date")
    idx = idx[np.asarray(idx_date >= trainer.TRAIN_START)]

    # 池时点化：与 run_lgb.train_model 同步的成员资格过滤
    mm = member_mask(idx.get_level_values("date"), idx.get_level_values("code"))
    idx = idx[mm]
    return idx


def check_model(
    m: str,
    meta: dict,
    pred: pd.DataFrame,
    axis_dates: list[pd.Timestamp],        # 该模型 X 的日期轴（复刻，含 warmup 前缀剔除后）
    all_dates: list[pd.Timestamp],         # 全历史交易日（位置参考系）
    expected_pred_idx: pd.MultiIndex,      # C3 复刻的期望预测行集
    far_cross_n: int,                      # C1e 复算的标签远引用越界行数
) -> None:
    cfg = MODEL_CONFIGS[m]
    s0, e0 = cfg["label_window"]
    buffer = cfg["label_buffer"]
    ts, te = trainer.TEST_START, trainer.TEST_END
    pos = {d: i for i, d in enumerate(all_dates)}
    first_test = min(d for d in all_dates if d >= ts)

    print(f"\n== Model {m}: label T+{s0}..T+{e0} open median, "
          f"baseline={cfg['baseline']}, buffer={buffer} ==")

    # ---- C1 训练掩码终点 ----
    print("  [C1] 训练掩码终点 ...")
    expected_end = buffered_train_end(axis_dates, ts, buffer)
    _check("meta.train_start 与复刻轴一致",
           meta["train_start"] == str(axis_dates[0].date()),
           f"meta={meta['train_start']}, 复刻={axis_dates[0].date()}")
    _check("meta.train_end == buffered_train_end",
           meta["train_end"] == str(expected_end.date()),
           f"meta={meta['train_end']}, 复刻={expected_end.date()} (buffer={buffer})")
    # train_end 是排他上界：末个纳入训练的日期是其前一交易日
    last_included = max(d for d in axis_dates if d < expected_end)
    reach = pos[last_included] + e0
    _check("末训练日标签最远引用 < 测试窗首日",
           reach < pos[first_test],
           f"last_included={last_included.date()} (T+{e0} → "
           f"{all_dates[min(reach, len(all_dates) - 1)].date()}) < "
           f"first_test={first_test.date()}")
    _check("label_buffer ≥ 标签窗末日（buffer 语义充分）",
           buffer >= e0, f"buffer={buffer}, label_window_end={e0}")

    # C1e（2026-08-24 新增）：停牌股标签远引用越界的排除机制交叉核对——
    # 标签按个股交易行前移、buffer 按池轴回退，停牌股 T+e0 可越界（实测
    # 2025-04-29 有 25 只）；断言复算越界数与训练时记录一致，防规则被
    # 静默移除
    meta_cnt = (meta.get("train_exclude_counts") or {}).get("label_far_cross")
    _check("标签远引用越界排除数与 meta 一致",
           meta_cnt is not None and int(meta_cnt) == far_cross_n,
           f"复算={far_cross_n}, meta={meta_cnt}")
    _check("meta 测试窗 = 训练入口常量",
           meta["test_start"] == str(ts.date()) and meta["test_end"] == str(te.date()),
           f"meta={meta['test_start']}~{meta['test_end']}")

    # ---- C2 输出校准尾段 ----
    print("  [C2] 输出校准尾段 ...")
    calib_dates = [d for d in axis_dates if d < expected_end][-trainer.CALIB_TAIL_DAYS:]
    _check("校准尾段日期全部 < train_end",
           bool(calib_dates) and max(calib_dates) < expected_end,
           f"tail={len(calib_dates)}d [{calib_dates[0].date()}~{calib_dates[-1].date()}]"
           f" < {expected_end.date()}；拟合集 < {calib_dates[0].date()}")
    reach_cal = pos[calib_dates[-1]] + e0
    _check("校准尾段标签最远引用 < 测试窗首日",
           reach_cal < pos[first_test],
           f"尾段末日 T+{e0} → {all_dates[reach_cal].date()} < {first_test.date()}")
    res = meta.get("results", {})
    _check("meta 含 calib_slope / calib_median",
           "calib_slope" in res and "calib_median" in res,
           f"slope={res.get('calib_slope')}, median={res.get('calib_median')}")

    # ---- C3 训练样本排除 ----
    print("  [C3] 训练样本排除（ST/退市/封板）...")
    col = f"pred_{cfg['horizon']}"
    _check(f"预测列 {col} 存在", col in pred.columns,
           f"columns={list(pred.columns)}")
    dts = pred.index.get_level_values("date")
    _check("预测日期范围 ⊆ 测试窗",
           dts.min() >= ts and dts.max() <= te,
           f"[{dts.min().date()} ~ {dts.max().date()}] ⊆ "
           f"[{ts.date()} ~ {te.date()}]")

    got = set(map(tuple, pred.index))
    want = set(map(tuple, expected_pred_idx))
    _check("预测行集与复刻过滤链逐行一致",
           got == want,
           f"parquet={len(got)}, 期望={len(want)}, "
           f"多出={len(got - want)}, 缺失={len(want - got)}")
    _check("行数与 meta.n_pred 一致",
           int(res.get("n_pred", -1)) == len(pred),
           f"meta.n_pred={res.get('n_pred')}, parquet={len(pred)}")


def main() -> int:
    print("=" * 72)
    print("双回归 bench 泄漏断言 — 主窗口盘上产物（4 模型）")
    print("=" * 72)

    check_c4_label_direction()

    pool = union_codes(since=trainer.HISTORY_SINCE)
    ph = ",".join(["?"] * len(pool))
    con = duckdb.connect(str(DB_PATH), read_only=True)

    # 全历史交易日（位置参考系）
    fv_dates_raw = pd.to_datetime(con.execute(
        f"SELECT DISTINCT date FROM factor_values WHERE code IN ({ph})",
        pool).fetchdf()["date"])
    all_dates = sorted(fv_dates_raw)

    # factor 面板行键（train_model 的 X 就从这里来）
    fv_df = con.execute(
        f"SELECT code, date FROM factor_values WHERE code IN ({ph})",
        pool).fetchdf()
    fv_df["date"] = pd.to_datetime(fv_df["date"])
    fv_idx = pd.MultiIndex.from_frame(fv_df[["date", "code"]])
    del fv_df

    # kline 只需覆盖 TRAIN_START 之后的标签窗（标签只向后引用）
    kline = con.execute(
        f"SELECT code, date, open, close FROM daily_kline "
        f"WHERE code IN ({ph}) AND date >= ? ORDER BY code, date",
        [*pool, str(trainer.TRAIN_START.date())]).fetchdf()
    kline["date"] = pd.to_datetime(kline["date"])

    st_df = con.execute(
        f"SELECT code, date, IsST FROM factor_values WHERE code IN ({ph})",
        pool).fetchdf()
    st_df["date"] = pd.to_datetime(st_df["date"])
    st_series = st_df.set_index(["date", "code"])["IsST"].astype(bool)

    delist_info = trainer.load_delist_info(con)
    con.close()

    delist_series = pd.Series(delist_info)
    limit_mask = compute_nextopen_limit_mask(kline, st_series=st_series)

    ts, te = trainer.TEST_START, trainer.TEST_END
    for m in sorted(MODEL_CONFIGS):
        cfg = MODEL_CONFIGS[m]

        meta_path = get_lgb_predictions_meta_path(m)
        pred_path = get_lgb_predictions_path(m)
        if not (meta_path.exists() and pred_path.exists()):
            _check(f"{m}: 预测产物存在", False,
                   f"缺 {meta_path.name} / {pred_path.name}（先跑 python run_lgb.py）")
            continue
        meta = json.loads(meta_path.read_text())
        pred = pd.read_parquet(pred_path)

        label = compute_median_open(kline, start_day=cfg["label_window"][0],
                                    end_day=cfg["label_window"][1],
                                    baseline=cfg["baseline"])
        x_idx = replicate_model_panel(m, fv_idx, label, st_series, limit_mask,
                                      delist_series)
        x_idx = x_idx.sort_values()   # DB 原序非日期序，shift 类计算必须先排
        axis_dates = sorted(x_idx.get_level_values("date").unique())[trainer.WARMUP_DAYS:]

        idx_date = x_idx.get_level_values("date")
        in_tw = np.asarray((idx_date >= ts) & (idx_date <= te))
        expected_pred_idx = x_idx[in_tw]

        # C1e 用：复算标签远引用越界行数（与 run_lgb 的 far_cross 同式，
        # 只计训练侧 date < test_start——测试窗行 far_cross 恒真但无训练意义）
        first_test = min(d for d in all_dates if d >= ts)
        date_s = pd.Series(idx_date, index=x_idx)
        e0 = cfg["label_window"][1]
        far_max = pd.concat(
            [date_s.groupby(level="code", sort=False).shift(-lag) for lag in range(1, e0 + 1)],
            axis=1).max(axis=1)
        far_cross_n = int(((far_max >= first_test) & (date_s < ts)).sum())

        check_model(m, meta, pred, axis_dates, all_dates, expected_pred_idx,
                    far_cross_n)

    # ---- summary ----
    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    n_fail = len(_RESULTS) - n_pass
    print("\n" + "=" * 72)
    print(f"共 {len(_RESULTS)} 项断言：{n_pass} 通过，{n_fail} 失败")
    if n_fail:
        print("失败项：")
        for name, ok, detail in _RESULTS:
            if not ok:
                print(f"  - {name} — {detail}")
    print("=" * 72)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
