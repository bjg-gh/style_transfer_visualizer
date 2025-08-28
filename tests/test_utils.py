"""
Tests for utility functions in style_transfer_visualizer.

This module verifies device selection, input validation,
directory setup, loss plotting, and random seed behavior.
"""
import logging
import os
import random
import shutil
import sys
from enum import Enum
from pathlib import Path, Path as RealPath
from types import ModuleType
from typing import cast

import pytest
import torch
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch

from style_transfer_visualizer import utils as stv_utils
from style_transfer_visualizer.type_defs import SaveOptions


class TestDeviceSetup:
    """Tests device selection and fallback behavior."""

    def test_setup_device_cpu(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test selecting CPU device and log message."""
        caplog.set_level(logging.INFO)
        device = stv_utils.setup_device("cpu")
        assert device.type == "cpu"
        assert "Using device: cpu" in caplog.text

    def test_setup_device_cuda_available(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
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
        monkeypatch: pytest.MonkeyPatch,
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
    """Tests validation of input paths and video parameters."""

    def test_parameter_validation(self) -> None:
        """Test invalid video quality triggers ValueError."""
        with pytest.raises(
            ValueError,
            match=r"(?i)video quality.*between 1 and 10",
        ):
            stv_utils.validate_parameters(video_quality=-1)

    def test_input_path_validation(
        self,
        content_image: str,
        style_image: str,
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
    """Tests output directory creation and fallback logic."""

    def test_setup_output_directory_creates_path(
        self,
        tmp_path: RealPath,
    ) -> None:
        """Test output directory creation."""
        output_dir = tmp_path / "new_output_dir"
        result = stv_utils.setup_output_directory(str(output_dir))
        assert result.exists()
        assert result.is_dir()

    def test_output_directory_fallback_on_failure(self) -> None:
        """Test fallback path is used if directory creation fails."""

        class FailingPath(RealPath):
            def mkdir(
                self,
                mode: int = 0o777,
                parents: bool = False,  # noqa: FBT001, FBT002
                exist_ok: bool = False,  # noqa: FBT001, FBT002
            ) -> None:
                if "restricted_output" in str(self):
                    msg = "Mocked failure"
                    raise PermissionError(msg)
                return super().mkdir(
                    mode=mode, parents=parents, exist_ok=exist_ok,
                )

        result = stv_utils.setup_output_directory(
            "restricted_output",
            path_factory=FailingPath,
        )
        assert result.name == "style_transfer_output"
        assert result.exists()


class TestPlotLossCurves:
    """Tests matplotlib loss plotting behavior and error handling."""

    def test_empty_metrics(
        self,
        output_dir: RealPath,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test warning if empty loss metrics are passed."""
        caplog.set_level("WARNING")
        stv_utils.plot_loss_curves({}, output_dir)
        assert "No loss metrics dictionary provided." in caplog.text

    def test_empty_lists(
        self,
        output_dir: RealPath,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test warning if all loss metric lists are empty."""
        caplog.set_level("WARNING")
        metrics = {
            "style_loss": [],
            "content_loss": [],
            "total_loss": [],
        }
        stv_utils.plot_loss_curves(metrics, output_dir)
        assert "Loss metrics dictionary is empty" in caplog.text

    def test_plot_loss_curves_missing_matplotlib(
        self,
        monkeypatch: pytest.MonkeyPatch,
        output_dir: RealPath,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test ImportError handling if matplotlib is unavailable."""
        caplog.set_level("WARNING")

        monkeypatch.setitem(sys.modules, "matplotlib",
                            cast("ModuleType | None", None))
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot",
                            cast("ModuleType | None", None))

        def fake_import(
            name: str,
            globals: dict[str, object] | None = None,  # noqa: A002
            locals: dict[str, object] | None = None,  # noqa: A002
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> ModuleType:
            if name == "matplotlib.pyplot":
                msg = "Simulated missing matplotlib"
                raise ImportError(msg)
            return __import__(name, globals, locals, fromlist, level)

        monkeypatch.setattr("builtins.__import__", fake_import)

        stv_utils.plot_loss_curves({"style_loss": [1.0]}, output_dir)
        assert "matplotlib not found" in caplog.text

    def test_valid_plot_loss_curves(self, output_dir: RealPath) -> None:
        """Test saving of valid loss plot file."""
        metrics = {
            "style_loss": [1.0, 0.8, 0.5],
            "content_loss": [0.5, 0.4, 0.3],
            "total_loss": [1.5, 1.2, 0.8],
        }
        stv_utils.plot_loss_curves(metrics, output_dir)
        plot_path = output_dir / "loss_plot.png"
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0

    def test_partial_empty_plot_loss_curves(
        self,
        output_dir: RealPath,
    ) -> None:
        """Test plot when some loss metric lists are empty."""
        metrics = {
            "style_loss": [1.0, 2.0, 3.0],
            "content_loss": [],
            "total_loss": [],
        }
        stv_utils.plot_loss_curves(metrics, output_dir)
        plot_path = output_dir / "loss_plot.png"
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0


class TestSeedSetup:
    """Tests random seed setup across libraries and devices."""

    def test_mocked_seed_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test all seed functions are called with correct values."""
        called = {
            "torch_manual": False,
            "torch_cuda_manual": False,
            "random_seed": False,
        }

        monkeypatch.setattr(
            torch,
            "manual_seed",
            lambda _: called.__setitem__("torch_manual", True),  # noqa: FBT003
        )
        monkeypatch.setattr(
            torch.cuda,
            "is_available",
            lambda: True,
        )
        monkeypatch.setattr(
            torch.cuda,
            "manual_seed_all",
            lambda _: called.__setitem__("torch_cuda_manual", True),  # noqa: FBT003
        )
        monkeypatch.setattr(
            random,
            "seed",
            lambda _: called.__setitem__("random_seed", True),  # noqa: FBT003
        )

        stv_utils.setup_random_seed(42)
        assert all(called.values()) is True

    def test_real_cuda_seed_execution(self) -> None:
        """Test real CUDA seeding (skipped in CI or if no GPU)."""
        if "CI" in os.environ or not torch.cuda.is_available():
            pytest.skip("Skipping CUDA-specific test")
        stv_utils.setup_random_seed(321)

    def test_setup_random_seed_no_cuda(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test seeding when CUDA is not available."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        stv_utils.setup_random_seed(999)


class TestSaveOutputs:
    """Tests saving final images, loss plots, and video logging."""

    def test_creates_final_image(self, output_dir: RealPath) -> None:
        """Test final image is saved correctly."""
        input_img = torch.rand(1, 3, 64, 64)
        loss_metrics = {
            "style_loss": [1.0],
            "content_loss": [0.5],
            "total_loss": [1.5],
        }

        opts = SaveOptions(
            content_name="dog",
            style_name="mosaic",
            video_name="timelapse_dog_x_mosaic.mp4",
            normalize=False,
            video_created=True,
            plot_losses=True,
        )

        stv_utils.save_outputs(
            input_img=input_img,
            loss_metrics=loss_metrics,
            output_dir=output_dir,
            elapsed=12.3,
            opts=opts,
        )

        final_path = output_dir / "stylized_dog_x_mosaic.png"
        assert final_path.exists()
        assert final_path.stat().st_size > 0

    def test_save_outputs_no_plot(self, output_dir: RealPath) -> None:
        """Test save_outputs skips plotting when plot_losses=False."""
        input_img = torch.rand(1, 3, 64, 64)
        loss_metrics = {
            "style_loss": [1.0],
            "content_loss": [0.5],
            "total_loss": [1.5],
        }

        opts = SaveOptions(
            content_name="cat",
            style_name="wave",
            video_name=None,
            normalize=False,
            video_created=False,
            plot_losses=False,
        )

        stv_utils.save_outputs(
            input_img=input_img,
            loss_metrics=loss_metrics,
            output_dir=output_dir,
            elapsed=2.5,
            opts=opts,
        )
        final_path = output_dir / "stylized_cat_x_wave.png"
        assert final_path.exists()

    def test_logs_video_path(
        self,
        output_dir: RealPath,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test video path logging message is recorded."""
        input_img = torch.rand(1, 3, 64, 64)
        loss_metrics = {
            "style_loss": [1.0],
            "content_loss": [0.5],
            "total_loss": [1.5],
        }

        caplog.set_level("INFO")
        opts = SaveOptions(
            content_name="cat",
            style_name="wave",
            video_name="timelapse_cat_x_wave.mp4",
            normalize=False,
            video_created=True,
            plot_losses=True,
        )
        stv_utils.save_outputs(
            input_img=input_img,
            loss_metrics=loss_metrics,
            output_dir=output_dir,
            elapsed=5.0,
            opts=opts,
        )

        assert "Video saved to:" in caplog.text

    def test_logs_creation(
        self,
        tmp_path: RealPath,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Force logger.info for directory creation to execute."""
        caplog.set_level("INFO")

        # Force .exists() to return False so the mkdir logic runs
        class CustomPath(RealPath):
            def exists(self, **_kwargs: object) -> bool:
                return False

        custom_path = CustomPath(tmp_path)

        input_img = torch.rand(1, 3, 64, 64)
        loss_metrics = {"style_loss": [1.0], "content_loss": [0.5],
                        "total_loss": [1.5]}

        opts = SaveOptions(
            content_name="test",
            style_name="coverage",
            video_name="timelapse_test_x_coverage.mp4",
            normalize=False,
            video_created=True,
            plot_losses=True,
        )

        stv_utils.save_outputs(
            input_img=input_img,
            loss_metrics=loss_metrics,
            output_dir=custom_path,
            elapsed=1.23,
            opts=opts,
        )

        assert "Created output directory" in caplog.text

    def test_handles_fallback(self, caplog: pytest.LogCaptureFixture) -> None:
        """Simulate failure to create output directory and use fallback."""
        class BrokenPath(RealPath):
            def mkdir(self, *_args: object, **_kwargs: object) -> None:
                msg = "Mocked mkdir failure"
                raise PermissionError(msg)

        fallback_path = RealPath.cwd() / "style_transfer_output"
        if fallback_path.exists():
            for file in fallback_path.iterdir():
                file.unlink()
            fallback_path.rmdir()

        input_img = torch.rand(1, 3, 64, 64)
        loss_metrics = {
            "style_loss": [1.0],
            "content_loss": [0.5],
            "total_loss": [1.5],
        }

        opts = SaveOptions(
            content_name="test",
            style_name="fallback",
            video_name=None,
            normalize=False,
            video_created=False,
            plot_losses=True,
        )

        stv_utils.save_outputs(
            input_img=input_img,
            loss_metrics=loss_metrics,
            output_dir=BrokenPath("/bad/path/"),
            elapsed=1.0,
            opts=opts,
        )

        fallback_img = fallback_path / "stylized_test_x_fallback.png"
        assert fallback_img.exists()
        assert "Mocked mkdir failure" in caplog.text

        # Cleanup
        if fallback_path.exists():
            shutil.rmtree(fallback_path)


@pytest.fixture
def fake_dist_missing(monkeypatch: MonkeyPatch) -> None:
    """Force importlib.metadata.version to raise PackageNotFoundError."""
    def missing(_name: str) -> None:
        raise stv_utils.importlib_metadata.PackageNotFoundError

    monkeypatch.setattr(stv_utils.importlib_metadata, "version", missing)


@pytest.fixture
def temp_pkg_root(tmp_path: Path, monkeypatch: MonkeyPatch) -> Path:
    """Create an isolated package root and point module __file__ to it."""
    root = tmp_path / "pkgroot"
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(stv_utils, "__file__", str(root / "pkg" / "utils.py"))
    return root


# ---------------------------------- Tests ------------------------------------

def test_resolve_project_version_from_distribution(
    monkeypatch: MonkeyPatch,
) -> None:
    """Return distribution version and do not read pyproject."""
    monkeypatch.setattr(stv_utils.importlib_metadata, "version",
                        lambda _name: "9.9.9")

    def boom(_fh: object) -> None:
        msg = "pyproject should not be read"
        raise RuntimeError(msg)

    # Guard that pyproject is not consulted when distro lookup succeeds
    monkeypatch.setattr(stv_utils.tomllib, "load", boom)

    assert stv_utils.resolve_project_version() == "9.9.9"


class Warn(Enum):
    """Sentinel for whether a warning is expected in a test case."""

    NO = 0
    YES = 1


@pytest.mark.usefixtures("fake_dist_missing")
@pytest.mark.parametrize(
    ("toml_body", "stub_loader", "expected", "expect_warn"),
    [
        (
            "[project]\nname='style-transfer-visualizer'\nversion='1.2.3'\n",
            None,
            "1.2.3",
            Warn.NO,
        ),
        (
            "[project]\nname='style-transfer-visualizer'\n",
            None,
            "0.0.0",
            Warn.NO,
        ),
        (
            "[project]\n",
            lambda _fh: (_ for _ in ()).throw(OSError("boom")),
            "0.0.0",
            Warn.YES,
        ),
    ],
    ids=["has_version", "missing_version", "oserror"],
)
def test_resolve_project_version_pyproject_paths(  # noqa: PLR0913
    temp_pkg_root: Path,
    toml_body: str,
    stub_loader: object | None,
    expected: str,
    expect_warn: Warn,
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
) -> None:
    """Fallback behavior with present, missing, or unreadable TOML."""
    pyproj = temp_pkg_root / "pyproject.toml"
    pyproj.write_text(toml_body, encoding="utf-8")

    if callable(stub_loader):
        # Force tomllib.load to raise OSError in the oserror case.
        monkeypatch.setattr(stv_utils.tomllib, "load", stub_loader)  # type: ignore[arg-type]

    with caplog.at_level("WARNING"):
        assert stv_utils.resolve_project_version() == expected

    if expect_warn is Warn.YES:
        assert any("Error reading" in r.getMessage() for r in caplog.records)
