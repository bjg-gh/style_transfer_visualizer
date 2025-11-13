"""Unit tests for the loss accumulator helper."""

from __future__ import annotations

import pytest
import torch
from pytest_mock import MockerFixture

from style_transfer_visualizer.loss_accumulator import LossAccumulator

CAPACITY_FIVE = 5


def _make_scalar(value: float) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32)


def test_accumulator_batches_host_sync(mocker: MockerFixture) -> None:
    """Only flush device tensors on configured logging cadence."""
    accumulator = LossAccumulator(
        log_every=3,
        history_capacity=6,
        track_history=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    spy = mocker.spy(accumulator, "_to_float")

    logged_steps: list[int] = []
    for step in range(1, 7):
        logged = accumulator.accumulate(
            step,
            _make_scalar(float(step)),
            _make_scalar(float(step + 1)),
            _make_scalar(float(step + 2)),
        )
        if step % 3 == 0:
            assert logged is not None
            logged_steps.append(logged.step)
        else:
            assert logged is None

    assert logged_steps == [3, 6]
    # Each flush converts three tensors (style/content/total).
    assert spy.call_count == len(logged_steps) * 3


def test_accumulator_bounded_history() -> None:
    """Loss history buffers behave as a circular queue."""
    accumulator = LossAccumulator(
        log_every=2,
        history_capacity=4,
        track_history=True,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    for step in range(1, 7):
        accumulator.accumulate(
            step,
            _make_scalar(float(step)),
            _make_scalar(float(step + 1)),
            _make_scalar(float(step + 2)),
        )

    history = accumulator.export_history()
    assert history["style_loss"] == [3.0, 4.0, 5.0, 6.0]
    assert history["content_loss"] == [4.0, 5.0, 6.0, 7.0]
    assert history["total_loss"] == [5.0, 6.0, 7.0, 8.0]
    assert accumulator.history_truncated is True


def test_accumulator_metadata_and_latest() -> None:
    """Capacity, latest, and empty history paths are exercised."""
    accumulator = LossAccumulator(
        log_every=1,
        history_capacity=CAPACITY_FIVE,
        track_history=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert accumulator.capacity == CAPACITY_FIVE
    assert accumulator.tracks_history is False
    assert accumulator.latest() is None
    assert accumulator.export_history() == {
        "style_loss": [],
        "content_loss": [],
        "total_loss": [],
    }

    logged = accumulator.accumulate(
        1,
        _make_scalar(1.0),
        _make_scalar(2.0),
        _make_scalar(3.0),
        force=True,
    )
    assert logged is not None
    assert accumulator.latest() == logged


def test_gather_history_and_sync_pending_edge_cases() -> None:
    """Internal helpers cope with missing buffers and pending data."""
    accumulator = LossAccumulator(
        log_every=10,
        history_capacity=2,
        track_history=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    empty = accumulator._gather_history(None)  # noqa: SLF001
    assert empty.numel() == 0
    assert accumulator._sync_pending() is None  # noqa: SLF001

    tracked = LossAccumulator(
        log_every=10,
        history_capacity=2,
        track_history=True,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    buf = tracked._gather_history(tracked._style_history)  # noqa: SLF001
    assert buf.numel() == 0


def test_gather_history_orders_entries_without_wrap() -> None:
    """_gather_history returns contiguous slices when no wrap occurs."""
    accumulator = LossAccumulator(
        log_every=1,
        history_capacity=4,
        track_history=True,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    for step in range(1, 3):
        accumulator.accumulate(
            step,
            _make_scalar(float(step)),
            _make_scalar(float(step)),
            _make_scalar(float(step)),
        )

    style = accumulator._gather_history(accumulator._style_history)  # noqa: SLF001
    assert style.tolist() == [1.0, 2.0]


def test_gather_history_orders_entries_with_wrap() -> None:
    """_gather_history concatenates wrapped segments in chronological order."""
    accumulator = LossAccumulator(
        log_every=1,
        history_capacity=3,
        track_history=True,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    for step in range(1, 5):
        accumulator.accumulate(
            step,
            _make_scalar(float(step)),
            _make_scalar(float(step)),
            _make_scalar(float(step)),
        )

    style = accumulator._gather_history(accumulator._style_history)  # noqa: SLF001
    assert style.tolist() == [2.0, 3.0, 4.0]


def test_record_history_raises_when_uninitialized() -> None:
    """_record_history guards against missing history buffers."""
    accumulator = LossAccumulator(
        log_every=5,
        history_capacity=CAPACITY_FIVE,
        track_history=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    with pytest.raises(RuntimeError, match="uninitialized"):
        accumulator._record_history(  # noqa: SLF001
            _make_scalar(1.0),
            _make_scalar(2.0),
            _make_scalar(3.0),
        )
