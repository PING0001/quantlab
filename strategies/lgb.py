# -*- coding: utf-8 -*-
"""
LightGBM 回归策略 + walk-forward 框架 + IC 评估 + 融合分。

2026-08-25 简化合并：原 base.py / lgb.py / combine.py / evaluation.py 四文件
合一；删除三分类 classifier 分支、peak loss、dart、滚动 walk-forward 分支
（均无调用方）与旧双模型 combine_scores。行为对 v8 回归路径保持逐行等价。

融合分 v8（2026-08-22 用户裁定 0.4/0.35/0.25；2026-08-28 用户裁定改 0.3/0.4/0.3，
权威值在 config.W2D/W6D/W20D，此处默认值仅 Fallback）：
    score = 0.3 * pred_2d + 0.4 * pred_6d + 0.3 * pred_20d
三模型统一 next_open 锚；某侧缺失时按可用权重重归一。
"""
from __future__ import annotations

import bisect
from collections.abc import Sequence
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata


# ============================================================================
# Walk-forward 框架
# ============================================================================

def buffered_train_end(
    all_dates: list[pd.Timestamp],
    boundary: pd.Timestamp,
    label_buffer: int,
) -> pd.Timestamp:
    """Exclusive upper bound for training dates: *boundary* stepped back
    *label_buffer* trading days in sorted *all_dates*.  Rows dated in the
    dropped tail carry forward-looking labels (T+1..T+label_buffer) that
    reference prices at or after *boundary*, so training on them would leak
    the evaluation window."""
    i = bisect.bisect_left(all_dates, boundary)
    return all_dates[max(0, i - label_buffer)]


def walk_forward(
    strategy: "LGBStrategy",
    factor_panel: pd.DataFrame,
    forward_returns: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    warmup_days: int = 0,
    label_buffer: int = 20,
    min_train: int = 252,
    train_exclude: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Walk-forward cross-sectional prediction（固定测试集协议：train once on
    data before test_start, predict the whole test period frozen）.

    Training rows within *label_buffer* trading days of the prediction boundary
    are dropped: their forward labels would reference prices from the
    prediction period.

    train_exclude: bool Series on factor_panel.index，True = 该行只从训练集剔除
    （预测照常输出）。2026-08-24 修复：此前封板/ST/退市观测在训练入口整行
    删除，预测行集被连带删掉--回测可交易宇宙被 T+1 信息条件化（回避次日
    开盘跌停的崩盘股，收益乐观偏）。现在排除语义收敛为"仅训练"；下游
    （回测执行层 / 报告 / IC 评估）各自过滤。

    Returns a DataFrame with one column per horizon, indexed by (date, code).
    """
    idx_dates = factor_panel.index.get_level_values("date")
    all_dates = sorted(idx_dates.unique())

    if warmup_days > 0:
        all_dates = all_dates[warmup_days:]

    # drop the label_buffer trading days before the test set: their
    # labels reference test-period prices (label look-ahead buffer)
    train_end = buffered_train_end(all_dates, test_start, label_buffer)
    train_mask = (idx_dates >= all_dates[0]) & (idx_dates < train_end)
    if train_exclude is not None:
        train_mask = train_mask & ~train_exclude.reindex(
            factor_panel.index, fill_value=False).to_numpy()
    X_train = factor_panel.loc[train_mask]
    y_train = forward_returns.loc[train_mask].reindex(columns=list(strategy.horizons))

    if X_train.index.get_level_values("date").nunique() >= min_train:
        strategy.fit(X_train, y_train)

    predictions: dict[pd.Timestamp, pd.DataFrame] = {}
    for dt in all_dates:
        if dt < test_start:
            continue
        if dt > test_end:
            break
        if not strategy.fitted:
            continue
        X_pred = factor_panel.xs(dt, level="date", drop_level=False)
        pred = strategy.predict(X_pred)
        if isinstance(pred.index, pd.MultiIndex):
            pred.index = pred.index.droplevel("date")
        predictions[dt] = pred

    if not predictions:
        return pd.DataFrame(dtype=float)
    return pd.concat(predictions, names=["date"])


# ============================================================================
# IC 评估
# ============================================================================

def rank_ic(predictions: pd.Series, returns: pd.Series) -> pd.Series:
    """
    Cross-sectional Rank IC (Spearman) per date.

    Both arguments are indexed by (date, code).  Returns a Series indexed by date.
    """
    combined = pd.DataFrame({"pred": predictions, "ret": returns}).dropna()
    if combined.empty:
        return pd.Series(dtype=float)

    def _spearman(g: pd.DataFrame) -> float:
        if len(g) < 5:
            return np.nan
        return g["pred"].rank().corr(g["ret"].rank())

    return combined.groupby("date").apply(_spearman).dropna()


def ic_summary(ic_series: pd.Series) -> dict:
    """
    Summarise an IC time-series.

    Returns a dict with mean_ic, std_ic, ir (information ratio),
    hit_rate, n_periods, min_ic, max_ic.
    """
    ic = ic_series.dropna()
    if len(ic) == 0:
        return {"n_periods": 0}
    std = ic.std()
    return {
        "mean_ic": float(ic.mean()),
        "std_ic": float(std),
        "ir": float(ic.mean() / std) if std > 0 else 0.0,
        "hit_rate": float((ic > 0).mean()),
        "n_periods": len(ic),
        "min_ic": float(ic.min()),
        "max_ic": float(ic.max()),
    }


# ============================================================================
# 截面 rank IC 积木（2026-09-03 自 factors/select_factors.py 迁入——挖矿层
# 已删，后续挖矿/因子筛选一律 tmp/ 临时脚本，从本处 import）
# ============================================================================

MIN_STOCKS_PER_DATE = 30


def _rank_ic_np(f_vals, l_vals):
    """Compute rank IC (Spearman) using numpy/scipy rankdata."""
    valid = ~np.isnan(f_vals) & ~np.isnan(l_vals)
    n = valid.sum()
    if n < MIN_STOCKS_PER_DATE:
        return np.nan
    f_r = rankdata(f_vals[valid])
    l_r = rankdata(l_vals[valid])
    f_c = f_r - f_r.mean()
    l_c = l_r - l_r.mean()
    denom = np.sqrt(np.dot(f_c, f_c) * np.dot(l_c, l_c))
    if denom == 0:
        return np.nan
    return np.dot(f_c, l_c) / denom


# ============================================================================
# 融合分
# ============================================================================

def _blend(pairs: list[tuple[pd.Series, float]]) -> pd.Series:
    """Generic weighted blend of prediction Series on the union index.

    pairs: [(pred_series, weight), ...]; weights should sum to 1.
    Missing-side renormalization: rows where a model has no prediction use the
    remaining models' weights renormalized (fillna(0) trick avoids NaN*0).
    """
    w_sum = sum(w for _, w in pairs)
    if abs(w_sum - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1, got {w_sum}")

    idx = None
    for s, _ in pairs:
        idx = s.index.union(idx) if idx is not None else s.index
    num = None
    tot = None
    for s, w in pairs:
        p = s.reindex(idx).astype(float)
        w_eff = w * p.notna()
        term = p.fillna(0.0) * w_eff
        num = term if num is None else num + term
        tot = w_eff if tot is None else tot + w_eff
    score = num / tot.where(tot > 0)
    score.name = "score"
    return score


def combine_scores3(pred_2d: pd.Series, pred_6d: pd.Series, pred_20d: pd.Series,
                    w2d: float = 0.30, w6: float = 0.40, w20: float = 0.30) -> pd.Series:
    """v8 三模型融合分（均 next_open 锚）。缺失侧按可用权重重归一。"""
    return _blend([(pred_2d, w2d), (pred_6d, w6), (pred_20d, w20)])


# ============================================================================
# LightGBM 策略（纯回归；v8 全系 objective=regression_l1）
# ============================================================================

def _horizon_label(h) -> str:
    if isinstance(h, str):
        return h
    return f"{h}d"


def _horizon_pcol(h) -> str:
    if isinstance(h, str):
        return f"pred_{h}"
    return f"pred_{h}d"


def _make_progress_callback(horizon, period: int = 50):
    h_label = _horizon_label(horizon)

    def _cb(env):
        if env.iteration % period == 0 and env.evaluation_result_list:
            metrics = " ".join(
                f"{name}={val:.6f}" for name, _, val, _ in env.evaluation_result_list
            )
            print(f"    [{h_label}] iter {env.iteration:4d}  {metrics}")

    _cb.order = 10
    return _cb


class LGBStrategy:
    """Multi-horizon LightGBM regressor with one booster per horizon.

    Anti-overfitting: early stopping on trailing validation dates, L1+L2
    regularisation, bagging (subsample/colsample), conservative leaf size.
    """

    def __init__(
        self,
        factor_names: Sequence[str],
        horizons: tuple = (5,),
        num_leaves: int = 63,
        max_depth: int | None = None,
        learning_rate: float = 0.02,
        n_estimators: int = 2000,
        min_child_samples: int = 100,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        subsample: float = 0.8,
        subsample_freq: int = 0,
        colsample_bytree: float = 0.8,
        early_stopping: bool = True,
        validation_fraction: float = 0.05,
        n_iter_no_change: int = 50,
        random_state: int = 42,
        n_jobs: int = -1,
        verbosity: int = -1,
        objective: str | None = None,
        categorical_feature: list[str] | None = None,
        name: str | None = None,
    ):
        self.factor_names = list(factor_names) if factor_names is not None else []
        self.name = name or self.__class__.__name__
        self.horizons = horizons
        self._fitted = False
        self._config = dict(
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_samples=min_child_samples,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=verbosity,
            objective=objective,
            categorical_feature=categorical_feature or [],
        )
        self._models: dict = {}
        self._categorical_feature = categorical_feature or []
        self._category_mappings: dict[str, dict[str, int]] = {}

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def horizon_columns(self) -> list[str]:
        return [_horizon_pcol(h) for h in self.horizons]

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "LGBStrategy":
        X_sel = X[self.factor_names].replace([np.inf, -np.inf], np.nan)
        common = X_sel.index.intersection(y.index)
        X_sel = X_sel.loc[common]
        y_sel = y.loc[common]

        mask = y_sel.notna().all(axis=1)
        X_sel, y_sel = X_sel.loc[mask], y_sel.loc[mask]

        if len(X_sel) < max(10, len(self.factor_names) * 10):
            self._fitted = False
            return self

        cfg = self._config

        X_sel = X_sel.sort_index(level="date")
        y_sel = y_sel.loc[X_sel.index]

        # Early-stopping split: trailing `validation_fraction` of the dates
        # contained in the data passed to fit() (no independent date source).
        # walk_forward truncates the label look-ahead buffer before calling
        # fit, which keeps this validation tail free of labels that reference
        # the test period; direct callers of fit must do the same.
        dates = X_sel.index.get_level_values("date").unique()
        n_val_dates = max(1, int(len(dates) * cfg["validation_fraction"]))
        train_dates = set(dates[: len(dates) - n_val_dates])
        val_dates = set(dates[len(dates) - n_val_dates :])

        train_mask = X_sel.index.get_level_values("date").isin(train_dates)
        val_mask = X_sel.index.get_level_values("date").isin(val_dates)

        X_train = X_sel.loc[train_mask]
        X_val = X_sel.loc[val_mask]

        self._models = {}
        for h in self.horizons:
            h_label = _horizon_label(h)
            print(f"  training horizon {h_label} ...")
            y_h_train = y_sel.loc[train_mask, h].values.astype(np.float64)
            y_h_val = y_sel.loc[val_mask, h].values.astype(np.float64)

            callbacks = []
            if cfg["early_stopping"]:
                callbacks.append(
                    lgb.early_stopping(cfg["n_iter_no_change"], verbose=False)
                )
            callbacks.append(_make_progress_callback(h, period=50))

            model_kwargs = dict(
                num_leaves=cfg["num_leaves"],
                learning_rate=cfg["learning_rate"],
                n_estimators=cfg["n_estimators"],
                min_child_samples=cfg["min_child_samples"],
                reg_alpha=cfg["reg_alpha"],
                reg_lambda=cfg["reg_lambda"],
                subsample=cfg["subsample"],
                colsample_bytree=cfg["colsample_bytree"],
                random_state=cfg["random_state"],
                n_jobs=cfg["n_jobs"],
                verbosity=cfg["verbosity"],
            )
            if cfg.get("subsample_freq", 0) > 0:
                model_kwargs["subsample_freq"] = cfg["subsample_freq"]
            if cfg.get("max_depth") is not None:
                model_kwargs["max_depth"] = cfg["max_depth"]
            if cfg.get("objective") is not None:
                model_kwargs["objective"] = cfg["objective"]

            model = lgb.LGBMRegressor(**model_kwargs)
            fit_kwargs = {}
            cat_feature = cfg.get("categorical_feature", [])
            if cat_feature:
                fit_kwargs["categorical_feature"] = cat_feature

            model.fit(
                X_train, y_h_train,
                eval_set=[(X_val, y_h_val)],
                callbacks=callbacks,
                **fit_kwargs,
            )
            self._models[h] = model

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=X.index, columns=self.horizon_columns, dtype=float)

        if not self._models:
            return result

        available = [f for f in self.factor_names if f in X.columns]
        if not available:
            return result

        # Include categorical features as-is (they don't need inf/nan replacement)
        cat_cols = [c for c in self._categorical_feature if c in X.columns]
        all_cols = available + [c for c in cat_cols if c not in available]

        X_sel = X[all_cols].copy()
        for col in available:
            if col in X_sel.columns:
                col_vals = X_sel[col].replace([np.inf, -np.inf], np.nan)
                X_sel[col] = col_vals.values if isinstance(col_vals, pd.Series) else col_vals

        for h in self.horizons:
            pred = self._models[h].predict(X_sel)
            result[_horizon_pcol(h)] = pred

        return result

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        bundle = {
            "models": self._models,
            "factor_names": self.factor_names,
            "horizons": self.horizons,
            "name": self.name,
            "config": self._config,
            "category_mappings": self._category_mappings,
        }
        joblib.dump(bundle, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "LGBStrategy":
        path = Path(path)
        bundle = joblib.load(path)
        strategy = cls(
            factor_names=bundle["factor_names"],
            horizons=bundle.get("horizons", (1, 3, 5, 10)),
            **bundle["config"],
        )
        strategy._models = bundle["models"]
        strategy._category_mappings = bundle.get("category_mappings", {})
        strategy._fitted = True
        return strategy
