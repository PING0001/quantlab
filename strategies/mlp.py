"""
Multi-head MLP strategy backed by PyTorch.

Predicts multiple forward-return horizons simultaneously via shared backbone
+ per-horizon output heads.  Each horizon"s labels are independently z-score
normalised before training (target variance normalisation).

Anti-overfitting:
  - L2 regularisation (weight_decay)
  - Dropout on hidden layers
  - early stopping with validation split
  - ReduceLROnPlateau scheduler
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseStrategy


class _MultiHeadMLP(nn.Module):
    """Feedforward MLP with shared backbone and per-horizon output heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: tuple[int, ...],
        dropout: float,
        horizons: tuple[int, ...],
    ):
        super().__init__()
        self.horizons = horizons

        # -- shared backbone --
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_layer_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev = h
        self.backbone = nn.Sequential(*layers)

        # -- per-horizon heads --
        self.heads = nn.ModuleDict({
            str(h): nn.Linear(prev, 1) for h in horizons
        })

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        """Run forward pass.  Returns dict mapping horizon -> prediction tensor."""
        shared = self.backbone(x)
        return {h: self.heads[str(h)](shared).squeeze(-1) for h in self.horizons}


class MLPStrategy(BaseStrategy):
    """Cross-sectional multi-head MLP mapping factors to multi-horizon forward returns."""

    def __init__(
        self,
        factor_names: Sequence[str],
        horizons: tuple[int, ...] = (1, 3, 5, 10),
        hidden_layer_sizes: tuple[int, ...] = (32, 16),
        dropout: float = 0.0,
        alpha: float = 0.001,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 20,
        learning_rate: float = 0.001,
        max_iter: int = 500,
        random_state: int = 42,
        batch_size: int = 1024,
        name: str | None = None,
    ):
        super().__init__(factor_names=factor_names, name=name)
        self.horizons = horizons
        self.scaler = StandardScaler()
        self._scaler_cache = None
        self._target_scalers: dict[int, tuple[float, float]] = {}  # horizon -> (mean, std)

        self._config = dict(
            horizons=horizons,
            hidden_layer_sizes=hidden_layer_sizes,
            dropout=dropout,
            alpha=alpha,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=random_state,
            batch_size=batch_size,
        )
        self._torch_model: _MultiHeadMLP | None = None
        self._input_dim: int | None = None

    @property
    def horizon_columns(self) -> list[str]:
        """Convenience: column names returned by predict()."""
        return [f"pred_{h}d" for h in self.horizons]

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "MLPStrategy":
        """
        Train on a panel slice.

        X  : factor values, MultiIndex (date, code)
        y  : forward returns, MultiIndex (date, code), one column per horizon.
        """
        X_sel = X[self.factor_names].replace([np.inf, -np.inf], np.nan).dropna()
        common = X_sel.index.intersection(y.index)
        X_sel = X_sel.loc[common]
        y_sel = y.loc[common]

        # Drop rows where ANY horizon target is NaN
        mask = y_sel.notna().all(axis=1)
        X_sel, y_sel = X_sel.loc[mask], y_sel.loc[mask]

        if len(X_sel) < max(10, len(self.factor_names) * 10):
            self._fitted = False
            return self

        # -- per-horizon target variance normalisation (z-score) --
        self._target_scalers = {}
        y_scaled = pd.DataFrame(index=y_sel.index, columns=self.horizons)
        for h in self.horizons:
            col = y_sel[h].values.astype(float)
            mean = float(np.nanmean(col))
            std = float(np.nanstd(col))
            if std < 1e-10:
                std = 1.0
            self._target_scalers[h] = (mean, std)
            y_scaled[h] = (col - mean) / std

        # -- factor scaling --
        X_scaled = self.scaler.fit_transform(X_sel)
        self._scaler_cache = self.scaler

        # -- build model --
        cfg = self._config
        torch.manual_seed(cfg["random_state"])

        self._input_dim = X_scaled.shape[1]
        self._torch_model = _MultiHeadMLP(
            self._input_dim,
            cfg["hidden_layer_sizes"],
            dropout=cfg.get("dropout", 0.0),
            horizons=self.horizons,
        )

        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t_dict = {h: torch.tensor(y_scaled[h].values, dtype=torch.float32) for h in self.horizons}

        # -- train / validation split --
        n_val = max(1, int(len(X_t) * cfg["validation_fraction"]))
        indices = torch.randperm(len(X_t))
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        X_train, X_val = X_t[train_idx], X_t[val_idx]
        y_train = {h: y_t_dict[h][train_idx] for h in self.horizons}
        y_val = {h: y_t_dict[h][val_idx] for h in self.horizons}

        optimizer = torch.optim.Adam(
            self._torch_model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["alpha"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10,
        )
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        self._torch_model.train()
        for epoch in range(cfg["max_iter"]):
            perm = torch.randperm(len(X_train))
            X_tr = X_train[perm]
            y_tr = {h: y_train[h][perm] for h in self.horizons}

            bs = cfg["batch_size"]
            for i in range(0, len(X_tr), bs):
                xb = X_tr[i: i + bs]
                yb = {h: y_tr[h][i: i + bs] for h in self.horizons}
                optimizer.zero_grad()
                preds = self._torch_model(xb)
                loss = sum(loss_fn(preds[h], yb[h]) for h in self.horizons)
                loss.backward()
                optimizer.step()

            # -- validation --
            self._torch_model.eval()
            with torch.no_grad():
                val_preds = self._torch_model(X_val)
                val_loss = sum(loss_fn(val_preds[h], y_val[h]) for h in self.horizons).item()
            self._torch_model.train()

            scheduler.step(val_loss)

            if epoch % 10 == 0 or epoch == 0:
                per_h = {h: f"{loss_fn(self._torch_model(X_val[:256])[h], y_val[h][:256]).item():.6f}" for h in self.horizons}
                loss_str = " / ".join(f"{h}d={per_h[h]}" for h in self.horizons)
                print(f"    epoch {epoch:3d}: val_loss={val_loss:.6f}  ({loss_str})")

            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._torch_model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if cfg["early_stopping"] and no_improve >= cfg["n_iter_no_change"]:
                print(f"    early stop at epoch {epoch + 1} (val_loss={val_loss:.6f}, best={best_val_loss:.6f})")
                break

        if best_state is not None:
            self._torch_model.load_state_dict(best_state)
        print(f"  training done: {epoch + 1} epochs (best_val_loss={best_val_loss:.6f})")

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict forward returns for all horizons.

        Returns DataFrame indexed like X with columns pred_1d, pred_3d, etc.
        """
        if self._torch_model is None:
            return pd.DataFrame(index=X.index, columns=self.horizon_columns, dtype=float)

        available = [f for f in self.factor_names if f in X.columns]
        if not available:
            return pd.DataFrame(index=X.index, columns=self.horizon_columns, dtype=float)

        X_sel = X[available].dropna()
        if len(X_sel) == 0:
            return pd.DataFrame(index=X.index, columns=self.horizon_columns, dtype=float)

        scaler = self._scaler_cache if self._scaler_cache is not None else self.scaler
        X_scaled = scaler.transform(X_sel)

        self._torch_model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_scaled, dtype=torch.float32)
            outputs = self._torch_model(X_t)

        result = pd.DataFrame(index=X_sel.index, columns=self.horizon_columns, dtype=float)
        for h in self.horizons:
            pred = outputs[h].numpy()
            mean, std = self._target_scalers.get(h, (0.0, 1.0))
            result[f"pred_{h}d"] = pred * std + mean

        return result.reindex(X.index)

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Persist model state_dict + scalers + config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        bundle = {
            "model_state": self._torch_model.state_dict() if self._torch_model is not None else None,
            "input_dim": self._input_dim,
            "scaler": self._scaler_cache if self._scaler_cache is not None else self.scaler,
            "target_scalers": {str(h): v for h, v in self._target_scalers.items()},
            "factor_names": self.factor_names,
            "horizons": self.horizons,
            "name": self.name,
            "config": self._config,
        }
        torch.save(bundle, path)
        return path

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "MLPStrategy":
        """Load a saved MLPStrategy from disk."""
        path = Path(path)
        bundle = torch.load(path, map_location=map_location, weights_only=False)

        cfg = bundle["config"]
        horizons = bundle.get("horizons", cfg.get("horizons", (1, 3, 5, 10)))
        dropout = cfg.get("dropout", 0.0)

        strategy = cls(
            factor_names=bundle["factor_names"],
            horizons=horizons,
            hidden_layer_sizes=cfg["hidden_layer_sizes"],
            dropout=dropout,
            alpha=cfg["alpha"],
            learning_rate=cfg.get("learning_rate", 0.001),
            max_iter=cfg["max_iter"],
            random_state=cfg["random_state"],
            batch_size=cfg.get("batch_size", 1024),
            name=bundle.get("name"),
        )

        scaler = bundle["scaler"]
        strategy.scaler = scaler
        strategy._scaler_cache = scaler

        ts_raw = bundle.get("target_scalers", {})
        strategy._target_scalers = {int(k): v for k, v in ts_raw.items()}

        input_dim = bundle.get("input_dim")
        if input_dim is not None and bundle["model_state"] is not None:
            strategy._input_dim = input_dim
            strategy._torch_model = _MultiHeadMLP(
                input_dim, cfg["hidden_layer_sizes"], dropout, horizons=horizons,
            )
            strategy._torch_model.load_state_dict(bundle["model_state"])

        strategy._fitted = True
        return strategy