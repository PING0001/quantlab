# -*- coding: utf-8 -*-
"""
Feature DataProxy: wraps Polars columns with overloaded arithmetic/comparison.
Ported from vnpy alpha dataset.
"""

from collections.abc import Callable
from numbers import Real
from typing import Union, cast

import polars as pl

EXPRESSION_FUNCTIONS: dict[str, Callable] = {}


def register_functions(functions: list[Callable]) -> None:
    for func in functions:
        EXPRESSION_FUNCTIONS[func.__name__] = func


class DataProxy:
    """Feature data proxy – chains Polars operations via operator overloading."""

    def __init__(self, df: pl.DataFrame) -> None:
        self.name: str = df.columns[-1]
        self.df: pl.DataFrame = df.rename({self.name: "data"})

    @staticmethod
    def _as_series(value: object) -> pl.Series:
        if isinstance(value, pl.Series):
            return value
        return cast(pl.Series, value)

    def _comparison_series(self, value: object) -> pl.Series:
        if isinstance(value, pl.Series):
            return value.cast(pl.Int32)
        if isinstance(value, bool):
            return pl.Series(name="data", values=[int(value)] * len(self.df))
        if isinstance(value, Real):
            return pl.Series(name="data", values=[int(bool(value))] * len(self.df))
        raise TypeError(f"Unsupported comparison result type: {type(value)!r}")

    def result(self, s: pl.Series) -> "DataProxy":
        result: pl.DataFrame = self.df[["datetime", "vt_symbol"]]
        result = result.with_columns(other=s)
        return DataProxy(result)

    # -- arithmetic --
    def __add__(self, other: Union["DataProxy", Real]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s = self._as_series(self.df["data"] + other.df["data"])
        else:
            s = self._as_series(self.df["data"] + other)
        return self.result(s)

    def __radd__(self, other: Union["DataProxy", Real]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s = self._as_series(other.df["data"] + self.df["data"])
        else:
            s = self._as_series(other + self.df["data"])
        return self.result(s)

    def __sub__(self, other: Union["DataProxy", Real]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s = self._as_series(self.df["data"] - other.df["data"])
        else:
            s = self._as_series(self.df["data"] - other)
        return self.result(s)

    def __rsub__(self, other: Union["DataProxy", Real]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s = self._as_series(other.df["data"] - self.df["data"])
        else:
            s = self._as_series(other - self.df["data"])
        return self.result(s)

    def __mul__(self, other: Union["DataProxy", Real]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s = self._as_series(self.df["data"] * other.df["data"])
        else:
            s = self._as_series(self.df["data"] * other)
        return self.result(s)

    def __rmul__(self, other: Union["DataProxy", Real]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s = self._as_series(self.df["data"] * other.df["data"])
        else:
            s = self._as_series(self.df["data"] * other)
        return self.result(s)

    def __truediv__(self, other: Union["DataProxy", Real]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s = self._as_series(self.df["data"] / other.df["data"])
        else:
            s = self._as_series(self.df["data"] / other)
        return self.result(s)

    def __rtruediv__(self, other: Union["DataProxy", Real]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s = self._as_series(other.df["data"] / self.df["data"])
        else:
            s = self._as_series(other / self.df["data"])
        return self.result(s)

    def __floordiv__(self, other: Union["DataProxy", Real]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s = self._as_series(self.df["data"] // other.df["data"])
        else:
            s = self._as_series(self.df["data"] // other)
        return self.result(s)

    def __mod__(self, other: Union["DataProxy", Real]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s = self._as_series(self.df["data"] % other.df["data"])
        else:
            s = self._as_series(self.df["data"] % other)
        return self.result(s)

    def __pow__(self, other: Union["DataProxy", Real]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s = self._as_series(self.df["data"].pow(other.df["data"]))
        else:
            s = self._as_series(self.df["data"].pow(cast(int | float, other)))
        return self.result(s)

    def __abs__(self) -> "DataProxy":
        s: pl.Series = self.df["data"].abs()
        return self.result(s)

    def __neg__(self) -> "DataProxy":
        s: pl.Series = -self.df["data"]
        return self.result(s)

    # -- comparison --
    def __gt__(self, other: Union["DataProxy", Real]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s: object = self.df["data"] > other.df["data"]
        else:
            s = self.df["data"] > other
        return self.result(self._comparison_series(s))

    def __ge__(self, other: Union["DataProxy", Real]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s: object = self.df["data"] >= other.df["data"]
        else:
            s = self.df["data"] >= other
        return self.result(self._comparison_series(s))

    def __lt__(self, other: Union["DataProxy", Real]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s: object = self.df["data"] < other.df["data"]
        else:
            s = self.df["data"] < other
        return self.result(self._comparison_series(s))

    def __le__(self, other: Union["DataProxy", Real]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s: object = self.df["data"] <= other.df["data"]
        else:
            s = self.df["data"] <= other
        return self.result(self._comparison_series(s))

    def __eq__(self, other: Union["DataProxy", Real]) -> "DataProxy":  # type: ignore[override]
        if isinstance(other, DataProxy):
            s: object = self.df["data"] == other.df["data"]
        else:
            s = self.df["data"] == other
        return self.result(self._comparison_series(s))

    def __ne__(self, other: Union["DataProxy", Real]) -> "DataProxy":  # type: ignore[override]
        if isinstance(other, DataProxy):
            s: object = self.df["data"] != other.df["data"]
        else:
            s = self.df["data"] != other
        return self.result(self._comparison_series(s))
