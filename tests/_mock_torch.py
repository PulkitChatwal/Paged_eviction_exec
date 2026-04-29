"""
Minimal numpy-backed torch mock for CPU-only test environments.
Only implements the subset used by paged_eviction.manager.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np


class Tensor:
    """Thin wrapper around a numpy array."""

    def __init__(self, data: Any) -> None:
        if isinstance(data, np.ndarray):
            self._d = data.astype(np.float32)
        else:
            self._d = np.asarray(data, dtype=np.float32)

    # --- properties ---
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._d.shape  # type: ignore[return-value]

    @property
    def ndim(self) -> int:
        return int(self._d.ndim)

    @property
    def device(self) -> str:
        return "cpu"

    def __len__(self) -> int:
        if self._d.ndim == 0:
            return 1
        return int(self._d.shape[0])

    # --- conversions ---
    def float(self) -> "Tensor":
        return Tensor(self._d.astype(np.float32))

    def item(self) -> float:
        return float(self._d.flat[0])

    def tolist(self) -> Any:
        return self._d.tolist()

    def numel(self) -> int:
        return int(self._d.size)

    # --- indexing ---
    def __getitem__(self, idx: Any) -> "Tensor":
        return Tensor(self._d[idx])

    def __setitem__(self, idx: Any, val: Any) -> None:
        self._d[idx] = val._d if isinstance(val, Tensor) else val

    # --- arithmetic ---
    def _u(self, other: Any) -> Any:
        return other._d if isinstance(other, Tensor) else other

    def __add__(self, other: Any) -> "Tensor":
        return Tensor(self._d + self._u(other))

    def __iadd__(self, other: Any) -> "Tensor":
        self._d = self._d + self._u(other)
        return self

    def __truediv__(self, other: Any) -> "Tensor":
        return Tensor(self._d / self._u(other))

    def __itruediv__(self, other: Any) -> "Tensor":
        self._d = self._d / self._u(other)
        return self

    def __mul__(self, other: Any) -> "Tensor":
        return Tensor(self._d * self._u(other))

    def __rmul__(self, other: Any) -> "Tensor":
        return Tensor(self._u(other) * self._d)

    # --- comparisons ---
    def __gt__(self, other: Any) -> Any:
        result = self._d > self._u(other)
        if result.ndim == 0:
            return bool(result)
        return Tensor(result)

    def __eq__(self, other: Any) -> Any:  # type: ignore[override]
        o = self._u(other)
        if isinstance(o, float) and np.isinf(o):
            result = np.isinf(self._d)
        else:
            result = self._d == o
        if hasattr(result, "ndim") and result.ndim == 0:
            return bool(result)
        return Tensor(result)

    # --- reductions ---
    def norm(self, dim: int = -1) -> "Tensor":
        return Tensor(np.linalg.norm(self._d, axis=dim))

    def mean(self, dim: int = -1) -> "Tensor":
        return Tensor(np.mean(self._d, axis=dim))

    def all(self) -> bool:
        return bool(self._d.all())

    def argmin(self) -> "Tensor":
        return Tensor(np.array(self._d.argmin()))

    def isfinite(self) -> "Tensor":
        return Tensor(np.isfinite(self._d))

    def __repr__(self) -> str:
        return f"Tensor({self._d})"


# --- module-level functions ---


def randn(*shape: int, device: str = "cpu") -> Tensor:
    return Tensor(np.random.randn(*shape).astype(np.float32))


def zeros(*shape: int, dtype: Any = None, device: str = "cpu") -> Tensor:
    return Tensor(np.zeros(shape, dtype=np.float32))


def ones(*shape: int) -> Tensor:
    return Tensor(np.ones(shape, dtype=np.float32))


def tensor(data: Any, dtype: Any = None) -> Tensor:
    return Tensor(np.asarray(data, dtype=np.float32))


def stack(tensors: List[Tensor], dim: int = 0) -> Tensor:
    return Tensor(np.stack([t._d for t in tensors], axis=dim))


def cat(tensors: List[Tensor], dim: int = 0) -> Tensor:
    return Tensor(np.concatenate([t._d for t in tensors], axis=dim))


def isfinite(t: Tensor) -> Tensor:
    return t.isfinite()


def allclose(a: Tensor, b: Tensor) -> bool:
    return bool(np.allclose(a._d, b._d))


def topk(input_t: Tensor, k: int, largest: bool = True) -> Tuple[Tensor, Tensor]:
    arr = input_t._d
    if largest:
        idx = np.argsort(arr)[-k:][::-1]
    else:
        idx = np.argsort(arr)[:k]
    return Tensor(arr[idx]), Tensor(idx.astype(np.float32))


float32 = np.float32
