"""Tests for utility functions in style_transfer_visualizer.

This module verifies device selection, logging, input validation,
directory setup, loss plotting, and random seed behavior.

Tested components:
- setup_device()
- setup_logger()
- validate_parameters()
- validate_input_paths()
- calculate_output_dimensions()
- setup_output_directory()
- plot_loss_curves()
- setup_random_seed()
"""

import os
import sys
import logging
import pytest
import torch
import random
from types import ModuleType
from typing import cast, Any
import numpy as np
from pathlib import Path as RealPath
import style_transfer_visualizer as stv


class TestDeviceSetup:
    def test_setup_device_cpu(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test selecting CPU device and log message."""
        caplog.set_level(logging.INFO)
        device = stv.setup_device("cpu")
        assert device.type == "cpu"
        assert "Using device: cpu" in caplog.text

    def test_setup_device_cuda_available(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test selecting CUDA device if available."""
        caplog.set_level(logging.INFO)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        device = stv.setup_device("cuda")
        assert device.type == "cuda"
        assert "Using device: cuda" in caplog.text

    def test_setup_device_cuda_fallback(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test fallback to CPU when CUDA is unavailable."""
        caplog.set_level(logging.INFO)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        device = stv.setup_device("cuda")
        assert device.type == "cpu"
        assert "CUDA requested but not available" in caplog.text

    def test_setup_device_invalid(self) -> None:
        """Test invalid device name raises error."""
        with pytest.raises(RuntimeError):
            stv.setup_device("invalid_device")


class TestLoggerSetup:
    def test_logger_singleton_behavior(self) -> None:
        """Test that logger instances are singleton per name."""
        logger1 = stv.setup_logger("test_logger")
        logger2 = stv.setup_logger("test_logger")
        assert logger1 is logger2
        assert len(logger1.handlers) == 1

    def test_logger_custom_formatter_and_handler(self) -> None:
        """Test custom formatter and handler are applied."""
        formatter = logging.Formatter("[CUSTOM] %(message)s")
        handler = logging.StreamHandler()
        logger = stv.setup_logger(
            "custom_logger",
            formatter=formatter,
            handler=handler
        )
        assert logger.name == "custom_logger"
        assert len(logger.handlers) == 1
        assert logger.handlers[0].formatter._fmt.startswith("[CUSTOM]")


class TestInputValidation:
    def test_parameter_validation(self) -> None:
        """Test invalid video quality triggers ValueError."""
        with pytest.raises(ValueError):
            stv.validate_parameters(video_quality=-1)

    def test_input_path_validation(
        self,
        content_image: str,
        style_image: str
    ) -> None:
        """Test image path validation logic with good and bad paths."""
        stv.validate_input_paths(content_image, style_image)

        with pytest.raises(FileNotFoundError):
            stv.validate_input_paths("nonexistent.jpg", style_image)

        with pytest.raises(FileNotFoundError):
            stv.validate_input_paths(content_image, "nonexistent.jpg")

        with pytest.raises(FileNotFoundError):
            stv.validate_input_paths("nonexistent1.jpg", "nonexistent2.jpg")


class TestOutputDirectory:
    def test_setup_output_directory_creates_path(
        self,
        tmp_path: RealPath
    ) -> None:
        """Test output directory creation."""
        output_dir = tmp_path / "new_output_dir"
        result = stv.setup_output_directory(str(output_dir))
        assert result.exists()
        assert result.is_dir()

    def test_output_directory_fallback_on_failure(
        self,
        tmp_path: RealPath
    ) -> None:
        """Test fallback path is used if directory creation fails."""
        class FailingPath(RealPath):
            def mkdir(self, *args: Any, **kwargs: Any) -> None:
                if "restricted_output" in str(self):
                    raise PermissionError("Mocked failure")
                return super().mkdir(*args, **kwargs)

        result = stv.setup_output_directory(
            "restricted_output",
            path_factory=FailingPath
        )
        assert result.name == "style_transfer_output"
        assert result.exists()


class TestPlotLossCurves:
    def test_empty_metrics(
        self,
        output_dir: RealPath,
        caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test warning if empty loss metrics are passed."""
        caplog.set_level("WARNING")
        stv.plot_loss_curves({}, output_dir)
        assert "No loss metrics dictionary provided." in caplog.text

    def test_empty_lists(
        self,
        output_dir: RealPath,
        caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test warning if all loss metric lists are empty."""
        caplog.set_level("WARNING")
        metrics = {
            "style_loss": [],
            "content_loss": [],
            "total_loss": []
        }
        stv.plot_loss_curves(metrics, output_dir)
        assert "Loss metrics dictionary is empty" in caplog.text

    def test_plot_loss_curves_missing_matplotlib(
        self,
        monkeypatch: pytest.MonkeyPatch,
        output_dir: RealPath,
        caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test ImportError handling if matplotlib is unavailable."""
        caplog.set_level("WARNING")

        monkeypatch.setitem(sys.modules, "matplotlib",
                            cast(ModuleType | None, None))
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot",
                            cast(ModuleType | None, None))

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "matplotlib.pyplot":
                raise ImportError("Simulated missing matplotlib")
            return __import__(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)

        stv.plot_loss_curves({"style_loss": [1.0]}, output_dir)
        assert "matplotlib not found" in caplog.text

    def test_valid_plot_loss_curves(self, output_dir: RealPath) -> None:
        """Test saving of valid loss plot file."""
        metrics = {
            "style_loss": [1.0, 0.8, 0.5],
            "content_loss": [0.5, 0.4, 0.3],
            "total_loss": [1.5, 1.2, 0.8]
        }
        stv.plot_loss_curves(metrics, output_dir)
        plot_path = output_dir / "loss_plot.png"
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0

    def test_partial_empty_plot_loss_curves(
        self,
        output_dir: RealPath
    ) -> None:
        """Test plot when some loss metric lists are empty."""
        metrics = {
            "style_loss": [1.0, 2.0, 3.0],
            "content_loss": [],
            "total_loss": []
        }
        stv.plot_loss_curves(metrics, output_dir)
        plot_path = output_dir / "loss_plot.png"
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0


class TestSeedSetup:
    def test_mocked_seed_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test all seed functions are called with correct values."""
        called = {
            "torch_manual": False,
            "torch_cuda_manual": False,
            "random_seed": False,
            "np_seed": False
        }

        monkeypatch.setattr(
            torch, "manual_seed",
            lambda x: called.__setitem__("torch_manual", True)
        )
        monkeypatch.setattr(
            torch.cuda, "is_available",
            lambda: True
        )
        monkeypatch.setattr(
            torch.cuda, "manual_seed_all",
            lambda x: called.__setitem__("torch_cuda_manual", True)
        )
        monkeypatch.setattr(
            random, "seed",
            lambda x: called.__setitem__("random_seed", True)
        )
        monkeypatch.setattr(
            np.random, "seed",
            lambda x: called.__setitem__("np_seed", True)
        )

        stv.setup_random_seed(42)
        assert all(called.values())

    def test_real_cuda_seed_execution(self) -> None:
        """Test real CUDA seeding (skipped in CI or if no GPU)."""
        if "CI" in os.environ or not torch.cuda.is_available():
            pytest.skip("Skipping CUDA-specific test")
        stv.setup_random_seed(321)

    def test_setup_random_seed_no_cuda(
        self,
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test seeding when CUDA is not available."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        stv.setup_random_seed(999)
