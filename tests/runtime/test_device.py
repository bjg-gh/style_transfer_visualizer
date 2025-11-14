"""Tests for runtime.device helpers."""

from __future__ import annotations

import logging
import random

import numpy as np
import pytest
import torch

from style_transfer_visualizer import random_utils as stv_random
from style_transfer_visualizer.runtime import device as runtime_device


class TestSetupDevice:
    """Validate device selection logic and logging."""

    def test_cpu_device_selected(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO)
        result = runtime_device.setup_device("cpu")
        assert result.type == "cpu"
        assert "Using device: cpu" in caplog.text

    def test_cuda_device_when_available(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        caplog.set_level(logging.INFO)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        result = runtime_device.setup_device("cuda")
        assert result.type == "cuda"
        assert "Using device: cuda" in caplog.text

    def test_cuda_fallback_to_cpu(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        caplog.set_level(logging.INFO)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        result = runtime_device.setup_device("cuda")
        assert result.type == "cpu"
        assert "CUDA requested but not available" in caplog.text

    def test_invalid_device_raises(self) -> None:
        with pytest.raises(RuntimeError):
            runtime_device.setup_device("invalid")


def test_setup_random_seed_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Random seed helper should make torch and random deterministic."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    runtime_device.setup_random_seed(123)
    torch_first = torch.rand(2)
    py_first = [random.random() for _ in range(2)]
    np_first = stv_random.get_numpy_rng().random(2)

    runtime_device.setup_random_seed(123)
    torch_second = torch.rand(2)
    py_second = [random.random() for _ in range(2)]
    np_second = stv_random.get_numpy_rng().random(2)

    assert torch.allclose(torch_first, torch_second)
    assert py_first == py_second
    assert np.array_equal(np_first, np_second)
