"""
MLP strategy backed by PyTorch.

    Anti-overfitting:
      - L2 regularization (weight_decay)
      - Dropout on hidden layers
      - early stopping with validation split
      - ReduceLROnPlateau scheduler
      - reduced default hidden layers
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

from .base import BaseStrategy


class _MLPModule(nn.Module):
    """Feedforward MLP with configurable hidden layers."""

    def __init__(self, input_dim: int, hidden_layer_sizes: tuple[int, ...], dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_layer_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPStrategy(BaseStrategy):
    """Cross-sectional MLP mapping factor values to forward returns, powered by PyTorch."""

    def __init__(
        self,
        factor_names: Sequence[str],
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
        self.scaler = StandardScaler()
        self._scaler_cache = None

        self._config = dict(
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
        self._torch_model: _MLPModule | None = None
        self._input_dim: int | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLPStrategy":
        X_sel = X[self.factor_names]
        X_sel = X_sel.replace([np.inf, -np.inf], np.nan).dropna()
        common = X_sel.index.intersection(y.index)
        X_sel = X_sel.loc[common]
        y_sel = y.loc[common]
        mask = y_sel.notna()
        X_sel, y_sel = X_sel.loc[mask], y_sel.loc[mask]

        if len(X_sel) < max(10, len(self.factor_names) * 10):
            self._fitted = False
            return self

        X_scaled = self.scaler.fit_transform(X_sel)
        self._scaler_cache = self.scaler

        cfg = self._config
        torch.manual_seed(cfg["random_state"])

        self._input_dim = X_scaled.shape[1]
        self._torch_model = _MLPModule(self._input_dim, cfg["hidden_layer_sizes"], dropout=cfg.get("dropout", 0.0))

        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(y_sel.values.ravel(), dtype=torch.float32)

        n_val = max(1, int(len(X_t) * cfg["validation_fraction"]))
        indices = torch.randperm(len(X_t))
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        X_train, y_train = X_t[train_idx], y_t[train_idx]
        X_val, y_val = X_t[val_idx], y_t[val_idx]

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
            X_tr, y_tr = X_train[perm], y_train[perm]

            bs = cfg["batch_size"]
            for i in range(0, len(X_tr), bs):
                xb = X_tr[i : i + bs]
                yb = y_tr[i : i + bs]
                optimizer.zero_grad()
                loss = loss_fn(self._torch_model(xb), yb)
                loss.backward()
                optimizer.step()

            self._torch_model.eval()
            with torch.no_grad():
                val_loss = loss_fn(self._torch_model(X_val), y_val).item()
            self._torch_model.train()

            scheduler.step(val_loss)

            if epoch % 10 == 0 or epoch == 0:
                print(f"    epoch {epoch:3d}: val_loss={val_loss:.6f}")

            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._torch_model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if cfg["early_stopping"] and no_improve >= cfg["n_iter_no_change"]:
                print(f"    early stop at epoch {epoch+1} (val_loss={val_loss:.6f}, best_val_loss={best_val_loss:.6f})")
                break

        if best_state is not None:
            self._torch_model.load_state_dict(best_state)
        print(f"  training done: {epoch+1} epochs")

        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._torch_model is None:
            return pd.Series(np.nan, index=X.index)
        available = [f for f in self.factor_names if f in X.columns]
        if not available:
            return pd.Series(np.nan, index=X.index)
        X_sel = X[available].dropna()
        if len(X_sel) == 0:
            return pd.Series(np.nan, index=X.index)

        scaler = self._scaler_cache if self._scaler_cache is not None else self.scaler
        X_scaled = scaler.transform(X_sel)

        self._torch_model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_scaled, dtype=torch.float32)
            pred = self._torch_model(X_t).numpy()

        result = pd.Series(pred, index=X_sel.index, name="prediction")
        return result.reindex(X.index)

    def save(self, path: str | Path) -> Path:
        """Persist model state_dict + scaler + config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        bundle = {
            "model_state": self._torch_model.state_dict() if self._torch_model is not None else None,
            "input_dim": self._input_dim,
            "scaler": self._scaler_cache if self._scaler_cache is not None else self.scaler,
            "factor_names": self.factor_names,
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
        dropout = cfg.get("dropout", 0.0)
        strategy = cls(
            factor_names=bundle["factor_names"],
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

        input_dim = bundle.get("input_dim")
        if input_dim is not None and bundle["model_state"] is not None:
            strategy._input_dim = input_dim
            strategy._torch_model = _MLPModule(input_dim, cfg["hidden_layer_sizes"], dropout=dropout)
            strategy._torch_model.load_state_dict(bundle["model_state"])

        strategy._fitted = True

        return strategy
