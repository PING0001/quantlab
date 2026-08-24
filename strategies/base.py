"""
Strategy base classes and walk-forward framework.

Data convention: arguments are pandas objects with a MultiIndex (date, code).

Multi-head support: y is a DataFrame with horizon columns; predict returns a
DataFrame with one column per horizon.
"""
from __future__ import annotations

import bisect
from abc import ABC, abstractmethod
from collections.abc import Sequence

import pandas as pd


class BaseStrategy(ABC):
    """Abstract strategy that learns a mapping from factors to forward returns."""

    def __init__(self, factor_names: Sequence[str] | None = None, name: str | None = None):
        self.factor_names = list(factor_names) if factor_names is not None else []
        self.name = name or self.__class__.__name__
        self._fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "BaseStrategy":
        """
        Train on a panel slice.  X columns are factor values; y columns are
        forward returns for each prediction horizon (e.g. 1d, 3d, 5d, 10d).
        Both are indexed by (date, code).
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict forward returns for all horizons.
        Returns a DataFrame with one column per horizon, indexed by (date, code).
        """
        ...

    @property
    def fitted(self) -> bool:
        return self._fitted


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
    strategy: BaseStrategy,
    factor_panel: pd.DataFrame,
    forward_returns: pd.DataFrame,
    train_window: int = 252,
    min_train: int = 252,
    start_date: pd.Timestamp | None = None,
    test_start: pd.Timestamp | None = None,
    test_end: pd.Timestamp | None = None,
    warmup_days: int = 0,
    label_buffer: int = 20,
    train_exclude: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Walk-forward cross-sectional prediction.

    When *test_start* and *test_end* are provided, dates in [test_start, test_end]
    are predicted using a single model trained on all data before *test_start*.
    Training rows within *label_buffer* trading days of the prediction boundary
    are dropped in both branches: their forward labels would reference prices
    from the prediction period.

    train_exclude: bool Series on factor_panel.index，True = 该行只从训练集剔除
    （预测照常输出）。2026-08-24 修复：此前封板/ST/退市观测在训练入口整行
    删除，预测行集被连带删掉——回测可交易宇宙被 T+1 信息条件化（回避次日
    开盘跌停的崩盘股，收益乐观偏）。现在排除语义收敛为"仅训练"；下游
    （回测执行层 / 报告 / IC 评估）各自过滤。

    Returns a DataFrame with one column per horizon, indexed by (date, code).
    """
    idx_dates = factor_panel.index.get_level_values("date")
    all_dates = sorted(idx_dates.unique())

    if warmup_days > 0:
        all_dates = all_dates[warmup_days:]

    if start_date is not None:
        all_dates = [d for d in all_dates if d >= start_date]

    has_test = test_start is not None and test_end is not None

    if has_test:
        # -- fixed test-set: train once, predict frozen --
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

    # -- standard walk-forward (no test set) --
    predictions: dict[pd.Timestamp, pd.DataFrame] = {}
    for i, dt in enumerate(all_dates):
        train_start = all_dates[max(0, i - train_window)]
        # labels of the last label_buffer days before dt are not realised yet
        # at prediction time; exclude them the same way as the fixed branch
        train_end = buffered_train_end(all_dates, dt, label_buffer)
        train_mask = (idx_dates >= train_start) & (idx_dates < train_end)
        if train_exclude is not None:
            train_mask = train_mask & ~train_exclude.reindex(
                factor_panel.index, fill_value=False).to_numpy()
        X_train = factor_panel.loc[train_mask]
        y_train = forward_returns.loc[train_mask].reindex(columns=list(strategy.horizons))

        if X_train.index.get_level_values("date").nunique() < min_train:
            continue

        strategy.fit(X_train, y_train)
        X_pred = factor_panel.xs(dt, level="date", drop_level=False)
        pred = strategy.predict(X_pred)
        if isinstance(pred.index, pd.MultiIndex):
            pred.index = pred.index.droplevel("date")
        predictions[dt] = pred
    if not predictions:
        return pd.DataFrame(dtype=float)
    return pd.concat(predictions, names=["date"])