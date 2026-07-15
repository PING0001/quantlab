# -*- coding: utf-8 -*-
"""
LightGBM multi-horizon strategy.

Trains one LGBMRegressor per prediction horizon.  Supports both standard
MSE regression and asymmetric peak-loss objective.

Anti-overfitting:
  - early stopping on validation set
  - L1 + L2 regularisation
  - bagging (subsample / colsample)
  - conservative leaf size
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from .base import BaseStrategy


def _peak_loss(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Asymmetric MSE: over-prediction penalised 9x more than under-prediction."""
    residual = y_pred - y_true
    is_over = residual > 0
    grad = np.where(is_over, 18.0 * residual, 2.0 * residual)
    hess = np.where(is_over, 18.0, 2.0)
    return grad, hess


def _peak_metric(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
    residual = y_pred - y_true
    is_over = residual > 0
    loss = np.where(is_over, 9.0 * residual ** 2, residual ** 2)
    return "peak_loss", float(np.mean(loss)), False


def _horizon_label(h) -> str:
    if isinstance(h, str):
        return h
    return f"{h}d"


def _horizon_pcol(h) -> str:
    if isinstance(h, str):
        return f"pred_{h}"
    return f"pred_{h}d"


def _make_progress_callback(horizon: Any, period: int = 50):
    h_label = _horizon_label(horizon)

    def _cb(env):
        if env.iteration % period == 0 and env.evaluation_result_list:
            metrics = " ".join(
                f"{name}={val:.6f}" for name, _, val, _ in env.evaluation_result_list
            )
            print(f"    [{h_label}] iter {env.iteration:4d}  {metrics}")

    _cb.order = 10
    return _cb


class LGBStrategy(BaseStrategy):
    """Multi-horizon LightGBM strategy with one booster per horizon."""

    def __init__(
        self,
        factor_names: Sequence[str],
        horizons: tuple[Any, ...] = (5, 10, 20, 30),
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
        boosting_type: str = "gbdt",
        drop_rate: float = 0.0,
        objective: str | None = None,
        categorical_feature: list[str] | None = None,
        name: str | None = None,
        l1_loss_horizon: Any = None,
    ):
        super().__init__(factor_names=factor_names, name=name)
        self.horizons = horizons
        self._l1_loss_horizon = l1_loss_horizon
        self._config = dict(
            num_leaves=num_leaves,
            max_depth=max_depth,
            boosting_type=boosting_type,
            drop_rate=drop_rate,
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
        self._models: dict[Any, lgb.LGBMRegressor] = {}
        self._categorical_feature = categorical_feature or []

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
                boosting_type=cfg["boosting_type"],
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
            if cfg.get("drop_rate", 0) > 0 and cfg["boosting_type"] == "dart":
                model_kwargs["drop_rate"] = cfg["drop_rate"]
            if cfg.get("objective") is not None:
                model_kwargs["objective"] = cfg["objective"]

            # LightGBM 4.x: pass column names directly to fit()
            cat_feature = cfg.get("categorical_feature", [])

            is_peak = (
                self._l1_loss_horizon is not None and h == self._l1_loss_horizon
            )
            if is_peak:
                model_kwargs["objective"] = "regression_l1"

            model = lgb.LGBMRegressor(**model_kwargs)
            fit_kwargs = {}
            if cat_feature:
                fit_kwargs["categorical_feature"] = cat_feature
            model.fit(
                X_train,
                y_h_train,
                eval_set=[(X_val, y_h_val)],
                callbacks=callbacks,
                eval_metric="mae" if is_peak else None,
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
            "l1_loss_horizon": self._l1_loss_horizon,
        }
        joblib.dump(bundle, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "LGBStrategy":
        path = Path(path)
        bundle = joblib.load(path)

        cfg = bundle["config"]
        strategy = cls(
            factor_names=bundle["factor_names"],
            horizons=bundle.get("horizons", cfg.get("horizons", (1, 3, 5, 10))),
            num_leaves=cfg["num_leaves"],
            max_depth=cfg.get("max_depth"),
            boosting_type=cfg.get("boosting_type", "gbdt"),
            drop_rate=cfg.get("drop_rate", 0.0),
            learning_rate=cfg["learning_rate"],
            n_estimators=cfg["n_estimators"],
            min_child_samples=cfg["min_child_samples"],
            reg_alpha=cfg["reg_alpha"],
            reg_lambda=cfg["reg_lambda"],
            subsample=cfg["subsample"],
            subsample_freq=cfg.get("subsample_freq", 0),
            colsample_bytree=cfg["colsample_bytree"],
            early_stopping=cfg["early_stopping"],
            validation_fraction=cfg["validation_fraction"],
            n_iter_no_change=cfg["n_iter_no_change"],
            random_state=cfg["random_state"],
            n_jobs=cfg.get("n_jobs", -1),
            verbosity=cfg.get("verbosity", -1),
            objective=cfg.get("objective"),
            categorical_feature=cfg.get("categorical_feature"),
            name=bundle.get("name"),
            l1_loss_horizon=bundle.get("l1_loss_horizon"),
        )
        strategy._models = bundle["models"]
        strategy._fitted = True
        return strategy
