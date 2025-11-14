"""Shared helpers for deterministic NumPy random number generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["current_numpy_seed", "get_numpy_rng", "seed_numpy_rng"]


@dataclass(slots=True)
class _NumpyRngState:
    """State container for the cached Generator."""

    seed: int | None = None
    generator: np.random.Generator | None = None


_STATE = _NumpyRngState()


def seed_numpy_rng(seed: int) -> np.random.Generator:
    """
    Seed and cache the global NumPy Generator instance.

    Uses the recommended ``default_rng`` constructor which supersedes
    the legacy ``np.random.seed`` / ``RandomState`` APIs.
    """
    _STATE.seed = seed
    _STATE.generator = np.random.default_rng(seed)
    return _STATE.generator


def get_numpy_rng() -> np.random.Generator:
    """Return the cached Generator, creating an unseeded instance."""
    if _STATE.generator is None:
        _STATE.generator = np.random.default_rng()
    return _STATE.generator


def current_numpy_seed() -> int | None:
    """Expose the last configured NumPy seed for observability/testing."""
    return _STATE.seed
