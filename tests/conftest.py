"""
Test configuration and shared fixtures for style_transfer_visualizer.

This module defines reusable pytest fixtures for image and tensor
generation, mocking models, and managing test directories. These
fixtures support all test modules in the test suite.

Note:
    This file is automatically loaded by pytest and should not be
    renamed.

"""
import shutil
import tempfile
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image
from torch import Tensor

from style_transfer_visualizer.config import StyleTransferConfig
from style_transfer_visualizer.constants import COLOR_MODE_RGB
from style_transfer_visualizer.logging_utils import logger
from style_transfer_visualizer.type_defs import InputPaths

STYLE_CONFIG_VARIANTS = [
    pytest.param({"device": "cpu", "mode": "realtime"}, id="cpu-realtime"),
    pytest.param({"device": "cpu", "mode": "postprocess"}, id="cpu-postprocess"),
]
if torch.cuda.is_available():
    STYLE_CONFIG_VARIANTS.append(
        pytest.param({"device": "cuda", "mode": "realtime"}, id="cuda-realtime"),
    )


@pytest.fixture
def test_device() -> torch.device:
    """Provides a PyTorch device (CPU or CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def test_dir() -> Generator[str, None, None]:
    """Provides a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def make_output_subdir(test_dir: str) -> Callable[[str], Path]:
    """Create (and reset) named subdirectories under the shared test_dir."""
    base = Path(test_dir)

    def _make(sub_name: str) -> Path:
        target = base / sub_name
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        return target

    return _make


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Provide a reusable temporary directory for output files."""
    return tmp_path


@pytest.fixture
def make_style_transfer_config(
    tmp_path: Path,
    test_device: torch.device,
) -> Callable[..., StyleTransferConfig]:
    """
    Build StyleTransferConfig instances with optional section overrides.

    Ensures each config uses an isolated output directory under tmp_path and
    defaults the device to the active test device.
    """
    default_output = tmp_path / "stv_outputs"
    default_output.mkdir(exist_ok=True)

    def _build(
        *,
        optimization: dict[str, Any] | None = None,
        video: dict[str, Any] | None = None,
        output: dict[str, Any] | None = None,
        hardware: dict[str, Any] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> StyleTransferConfig:
        data: dict[str, Any] = {}
        if optimization:
            data["optimization"] = dict(optimization)
        if video:
            data["video"] = dict(video)
        effective_output = dict(output or {})
        if "output" in effective_output:
            effective_output["output"] = str(effective_output["output"])
        else:
            effective_output["output"] = str(default_output)
        data["output"] = effective_output

        effective_hardware = dict(hardware or {})
        device_value = effective_hardware.get("device", str(test_device))
        effective_hardware["device"] = str(device_value)
        data["hardware"] = effective_hardware

        if extras:
            for section, section_data in extras.items():
                if isinstance(section_data, dict):
                    data[section] = dict(section_data)
                else:
                    data[section] = section_data

        return StyleTransferConfig.model_validate(data)

    return _build


@pytest.fixture(params=STYLE_CONFIG_VARIANTS)
def style_config_variant(
    make_style_transfer_config: Callable[..., StyleTransferConfig],
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> StyleTransferConfig:
    """Provide StyleTransferConfig variants covering device and video modes."""
    variant: dict[str, str] = request.param
    output_root = tmp_path / f"stv_{variant['device']}_{variant['mode']}"
    output_root.mkdir(exist_ok=True)
    cfg = make_style_transfer_config(
        video={"mode": variant["mode"]},
        hardware={"device": variant["device"]},
        output={"output": output_root},
    )
    cfg.video.mode_override = True
    return cfg


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a sample 100x100 red RGB PIL image."""
    return Image.new(COLOR_MODE_RGB, (100, 100), color="red")


@pytest.fixture
def sample_tensor() -> Tensor:
    """Create a random normalized PyTorch tensor [1, 3, 100, 100]."""
    return torch.randn(1, 3, 100, 100)


@pytest.fixture
def style_image(test_dir: str) -> Path:
    """
    Create and save a blue RGB style image.

    Returns:
        str: Path to the saved image.

    """
    img = Image.new(COLOR_MODE_RGB, (64, 64), color="blue")
    path = Path(test_dir) / "style.jpg"
    img.save(path)
    return path


@pytest.fixture
def content_image(test_dir: str) -> Path:
    """
    Create and save a green RGB content image.

    Returns:
        str: Path to the saved image.

    """
    img = Image.new(COLOR_MODE_RGB, (64, 64), color="green")
    path = Path(test_dir) / "content.jpg"
    img.save(path)
    return path


@pytest.fixture
def input_paths(content_image: Path, style_image: Path) -> InputPaths:
    """Typed helper for passing content/style paths to the pipeline."""
    return InputPaths(content_path=str(content_image),
                      style_path=str(style_image))


@pytest.fixture
def make_input_paths(
    content_image: Path,
    style_image: Path,
) -> Callable[..., InputPaths]:
    """Factory for building InputPaths with optional overrides."""

    def _build(
        *,
        content: str | Path | None = None,
        style: str | Path | None = None,
    ) -> InputPaths:
        content_path = Path(content) if content is not None else content_image
        style_path = Path(style) if style is not None else style_image
        return InputPaths(content_path=str(content_path),
                          style_path=str(style_path))

    return _build


@pytest.fixture
def style_layers() -> list[int]:
    """Return example indices for style feature layers."""
    return [0, 2, 4]


@pytest.fixture
def content_layers() -> list[int]:
    """Return example indices for content feature layers."""
    return [1, 3]


@pytest.fixture(autouse=True)
def enable_logger_propagation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable propagation for visualizer logger to allow caplog to work."""
    monkeypatch.setattr(logger, "propagate", True)
