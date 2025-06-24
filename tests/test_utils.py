"""Tests for utility functions in style_transfer_visualizer.

This module verifies device selection, input validation,
directory setup, loss plotting, and random seed behavior.
"""
import logging
import os
import random
import shutil
import sys
from pathlib import Path as RealPath, Path
from types import ModuleType
from typing import cast, Any

import numpy as np
import pytest
import torch

from style_transfer_visualizer import utils as stv_utils


class TestDeviceSetup:
    def test_setup_device_cpu(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test selecting CPU device and log message."""
        caplog.set_level(logging.INFO)
        device = stv_utils.setup_device("cpu")
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
        device = stv_utils.setup_device("cuda")
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
        device = stv_utils.setup_device("cuda")
        assert device.type == "cpu"
        assert "CUDA requested but not available" in caplog.text

    def test_setup_device_invalid(self) -> None:
        """Test invalid device name raises error."""
        with pytest.raises(RuntimeError):
            stv_utils.setup_device("invalid_device")


class TestInputValidation:
    def test_parameter_validation(self) -> None:
        """Test invalid video quality triggers ValueError."""
        with pytest.raises(ValueError):
            stv_utils.validate_parameters(video_quality=-1)

    def test_input_path_validation(
        self,
        content_image: str,
        style_image: str
    ) -> None:
        """Test image path validation logic with good and bad paths."""
        stv_utils.validate_input_paths(content_image, style_image)

        with pytest.raises(FileNotFoundError):
            stv_utils.validate_input_paths("nonexistent.jpg", style_image)

        with pytest.raises(FileNotFoundError):
            stv_utils.validate_input_paths(content_image, "nonexistent.jpg")

        with pytest.raises(FileNotFoundError):
            stv_utils.validate_input_paths("nonexistent1.jpg",
                                           "nonexistent2.jpg")


class TestOutputDirectory:
    def test_setup_output_directory_creates_path(
        self,
        tmp_path: RealPath
    ) -> None:
        """Test output directory creation."""
        output_dir = tmp_path / "new_output_dir"
        result = stv_utils.setup_output_directory(str(output_dir))
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

        result = stv_utils.setup_output_directory(
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
        stv_utils.plot_loss_curves({}, output_dir)
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
        stv_utils.plot_loss_curves(metrics, output_dir)
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

        stv_utils.plot_loss_curves({"style_loss": [1.0]}, output_dir)
        assert "matplotlib not found" in caplog.text

    def test_valid_plot_loss_curves(self, output_dir: RealPath) -> None:
        """Test saving of valid loss plot file."""
        metrics = {
            "style_loss": [1.0, 0.8, 0.5],
            "content_loss": [0.5, 0.4, 0.3],
            "total_loss": [1.5, 1.2, 0.8]
        }
        stv_utils.plot_loss_curves(metrics, output_dir)
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
        stv_utils.plot_loss_curves(metrics, output_dir)
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

        stv_utils.setup_random_seed(42)
        assert all(called.values())

    def test_real_cuda_seed_execution(self) -> None:
        """Test real CUDA seeding (skipped in CI or if no GPU)."""
        if "CI" in os.environ or not torch.cuda.is_available():
            pytest.skip("Skipping CUDA-specific test")
        stv_utils.setup_random_seed(321)

    def test_setup_random_seed_no_cuda(
        self,
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test seeding when CUDA is not available."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        stv_utils.setup_random_seed(999)


class TestSaveOutputs:
    def test_creates_final_image(self, output_dir: Path):
        """Test final image is saved correctly."""
        input_img = torch.rand(1, 3, 64, 64)
        loss_metrics = {
            "style_loss": [1.0],
            "content_loss": [0.5],
            "total_loss": [1.5]
        }

        stv_utils.save_outputs(
            input_img=input_img,
            loss_metrics=loss_metrics,
            output_dir=output_dir,
            elapsed=12.3,
            content_name="dog",
            style_name="mosaic",
            video_name="timelapse_dog_x_mosaic.mp4",
            normalize=False,
            video_created=True
        )

        final_path = output_dir / "stylized_dog_x_mosaic.png"
        assert final_path.exists()
        assert final_path.stat().st_size > 0

    def test_logs_video_path(
        self,
        output_dir: Path,
        caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test video path logging message is recorded."""
        input_img = torch.rand(1, 3, 64, 64)
        loss_metrics = {
            "style_loss": [1.0],
            "content_loss": [0.5],
            "total_loss": [1.5]
        }

        caplog.set_level("INFO")
        stv_utils.save_outputs(
            input_img=input_img,
            loss_metrics=loss_metrics,
            output_dir=output_dir,
            elapsed=5.0,
            content_name="cat",
            style_name="wave",
            video_name="timelapse_cat_x_wave.mp4",
            normalize=False,
            video_created=True
        )

        assert "Video saved to:" in caplog.text

    def test_logs_creation(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture
    ):
        """Force logger.info for directory creation to execute."""
        caplog.set_level("INFO")

        # Force .exists() to return False so the mkdir logic runs
        class CustomPath(Path):
            def exists(self, **_kwargs) -> bool:
                return False

        custom_path = CustomPath(tmp_path)

        input_img = torch.rand(1, 3, 64, 64)
        loss_metrics = {"style_loss": [1.0], "content_loss": [0.5],
                        "total_loss": [1.5]}

        stv_utils.save_outputs(
            input_img=input_img,
            loss_metrics=loss_metrics,
            output_dir=custom_path,
            elapsed=1.23,
            content_name="test",
            style_name="coverage",
            video_name="timelapse_test_x_coverage.mp4",
            normalize=False,
            video_created=True
        )

        assert "Created output directory" in caplog.text

    def test_handles_fallback(self, caplog: pytest.LogCaptureFixture):
        """Simulate failure to create output directory and use fallback."""
        class BrokenPath(Path):
            def mkdir(self, *args: Any, **kwargs: Any) -> None:
                raise PermissionError("Mocked mkdir failure")

        fallback_path = Path.cwd() / "style_transfer_output"
        if fallback_path.exists():
            for file in fallback_path.iterdir():
                file.unlink()
            fallback_path.rmdir()

        input_img = torch.rand(1, 3, 64, 64)
        loss_metrics = {
            "style_loss": [1.0],
            "content_loss": [0.5],
            "total_loss": [1.5]
        }

        stv_utils.save_outputs(
            input_img=input_img,
            loss_metrics=loss_metrics,
            output_dir=BrokenPath("/bad/path/"),
            elapsed=1.0,
            content_name="test",
            style_name="fallback",
            video_name=None,
            normalize=False,
            video_created=False
        )

        fallback_img = fallback_path / "stylized_test_x_fallback.png"
        assert fallback_img.exists()
        assert "Mocked mkdir failure" in caplog.text

        # Cleanup
        if fallback_path.exists():
            shutil.rmtree(fallback_path)
