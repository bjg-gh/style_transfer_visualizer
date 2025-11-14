"""Tests for shared NumPy RNG helpers."""

from __future__ import annotations

import numpy as np
import pytest

import style_transfer_visualizer.random_utils as stv_random_utils


@pytest.fixture(autouse=True)
def restore_state() -> None:
    """Restore RNG state after each test."""
    prev_seed = stv_random_utils._STATE.seed  # type: ignore[attr-defined]
    prev_gen = stv_random_utils._STATE.generator  # type: ignore[attr-defined]
    yield
    stv_random_utils._STATE.seed = prev_seed  # type: ignore[attr-defined]
    stv_random_utils._STATE.generator = prev_gen  # type: ignore[attr-defined]


def test_seed_numpy_rng_tracks_state() -> None:
    """Seeding should cache the generator and remember the seed."""
    gen = stv_random_utils.seed_numpy_rng(321)
    assert stv_random_utils.current_numpy_seed() == 321
    assert stv_random_utils.get_numpy_rng() is gen

    # Consuming the generator should mirror standalone default_rng behaviour.
    expected = np.random.default_rng(321).integers(0, 10, size=4)
    np.testing.assert_array_equal(gen.integers(0, 10, size=4), expected)


def test_get_numpy_rng_lazy_initialization() -> None:
    """get_numpy_rng should lazily create a generator when unseeded."""
    stv_random_utils._STATE.seed = None  # type: ignore[attr-defined]
    stv_random_utils._STATE.generator = None  # type: ignore[attr-defined]

    gen = stv_random_utils.get_numpy_rng()

    assert isinstance(gen, np.random.Generator)
    assert stv_random_utils.current_numpy_seed() is None
    # Subsequent calls should reuse the cached generator.
    assert stv_random_utils.get_numpy_rng() is gen
