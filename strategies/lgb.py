# -*- coding: utf-8 -*-
"""
LightGBM multi-horizon strategy.

Trains one LGBMRegressor per prediction horizon (1d / 3d / 5d / 10d).
All models share the same factor inputs; each learns its own horizon's
forward return independently.

Anti-overfitting:
  - early stopping on validation set
  - L1 + L2 regularisation
  - bagging (subsample / colsample)
  - conservative leaf size
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from .base import BaseStrategy


def _make_progress_callback(horizon: int, period: int = 50):
    """Return a callback that prints training progress every *period* rounds."""
    def _cb(env):
        if env.iteration % period == 0 and env.evaluation_result_list:
            metrics = " ".join(f"{name}={val:.6f}" for name, _, val, _ in env.evaluation_result_list)
            print(f"    [{horizon}d] iter {env.iteration:4d}  {metrics}")
    _cb.order = 10
    return _cb


class LGBStrategy(BaseStrategy):
    """Multi-horizon LightGBM strategy with one booster per horizon."""

    def __init__(
        self,
        factor_names: Sequence[str],
        horizons: tuple[int, ...] = (1, 3, 5, 10),
        num_leaves: int = 63,
        learning_rate: float = 0.02,
        n_estimators: int = 2000,
        min_child_samples: int = 100,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping: bool = True,
        validation_fraction: float = 0.05,
        n_iter_no_change: int = 50,
        random_state: int = 42,
        n_jobs: int = -1,
        verbosity: int = -1,
        name: str | None = None,
    ):
        super().__init__(factor_names=factor_names, name=name)
        self.horizons = horizons
        self._config = dict(
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_samples=min_child_samples,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=verbosity,
        )
        self._models: dict[int, lgb.LGBMRegressor] = {}

    @property
    def horizon_columns(self) -> list[str]:
        return [f"pred_{h}d" for h in self.horizons]

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "LGBStrategy":
        X_sel = X[self.factor_names].replace([np.inf, -np.inf], np.nan).dropna()
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

        X_train_np = X_sel.loc[train_mask].values.astype(np.float64)
        X_val_np = X_sel.loc[val_mask].values.astype(np.float64)

        self._models = {}
        for h in self.horizons:
            print(f"  training horizon {h}d ...")
            y_h_train = y_sel.loc[train_mask, h].values.astype(np.float64)
            y_h_val = y_sel.loc[val_mask, h].values.astype(np.float64)

            callbacks = []
            if cfg["early_stopping"]:
                callbacks.append(lgb.early_stopping(cfg["n_iter_no_change"], verbose=False))
            callbacks.append(_make_progress_callback(h, period=50))

            model = lgb.LGBMRegressor(
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
            model.fit(
                X_train_np, y_h_train,
                eval_set=[(X_val_np, y_h_val)],
                callbacks=callbacks,
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

        X_sel = X[available].replace([np.inf, -np.inf], np.nan)
        valid = X_sel.notna().all(axis=1)
        if not valid.any():
            return result

        X_np = X_sel.loc[valid].values.astype(np.float64)

        for h in self.horizons:
            pred = self._models[h].predict(X_np)
            result.loc[valid, f"pred_{h}d"] = pred

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
            learning_rate=cfg["learning_rate"],
            n_estimators=cfg["n_estimators"],
            min_child_samples=cfg["min_child_samples"],
            reg_alpha=cfg["reg_alpha"],
            reg_lambda=cfg["reg_lambda"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            early_stopping=cfg["early_stopping"],
            validation_fraction=cfg["validation_fraction"],
            n_iter_no_change=cfg["n_iter_no_change"],
            random_state=cfg["random_state"],
            n_jobs=cfg.get("n_jobs", -1),
            verbosity=cfg.get("verbosity", -1),
            name=bundle.get("name"),
        )
        strategy._models = bundle["models"]
        strategy._fitted = True
        return strategy
