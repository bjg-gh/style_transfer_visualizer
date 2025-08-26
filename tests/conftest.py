"""
Test configuration and shared fixtures for style_transfer_visualizer.

This module defines reusable pytest fixtures for image and tensor
generation, mocking models, and managing test directories. These
fixtures support all test modules in the test suite.

Note:
    This file is automatically loaded by pytest and should not be
    renamed.

"""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import Tensor
from torch.optim import Optimizer

from style_transfer_visualizer.constants import COLOR_MODE_RGB
from style_transfer_visualizer.logging_utils import logger
from style_transfer_visualizer.type_defs import InputPaths


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
def output_dir(tmp_path: Path) -> Path:
    """Provide a reusable temporary directory for output files."""
    return tmp_path


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
def mock_vgg() -> torch.nn.Module:
    """Provide a mock VGG-style model with a small Conv+ReLU stack."""
    class MockVGG(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, padding=1),
                torch.nn.ReLU(inplace=True),
            )

        @staticmethod
        def forward(_: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                "relu1_1": torch.randn(1, 64, 32, 32),
                "relu2_1": torch.randn(1, 128, 16, 16),
                "relu3_1": torch.randn(1, 256, 8, 8),
            }

    return MockVGG()


@pytest.fixture
def style_layers() -> list[int]:
    """Return example indices for style feature layers."""
    return [0, 2, 4]


@pytest.fixture
def content_layers() -> list[int]:
    """Return example indices for content feature layers."""
    return [1, 3]


@pytest.fixture
def mock_feature_maps() -> Tensor:
    """Create a mock feature map tensor."""
    return torch.randn(1, 64, 32, 32)


@pytest.fixture
def mock_style_targets() -> list[Tensor]:
    """Create a list of mock style target Gram matrices."""
    return [torch.randn(1, 64, 64) for _ in range(3)]


@pytest.fixture
def mock_content_targets() -> list[Tensor]:
    """Create a list of mock content feature maps."""
    return [torch.randn(1, 64, 32, 32) for _ in range(2)]


@pytest.fixture
def mock_vgg_features() -> dict:
    """Create a dictionary of mock VGG feature maps."""
    return {
        "relu1_1": torch.randn(1, 64, 32, 32),
        "relu2_1": torch.randn(1, 128, 16, 16),
        "relu3_1": torch.randn(1, 256, 8, 8),
    }


@pytest.fixture
def input_image_size() -> tuple[int, int]:
    """Return a standard input image size (256, 256)."""
    return 256, 256


@pytest.fixture
def mock_optimizer(sample_tensor: Tensor) -> Optimizer:
    """Create a mock optimizer over the sample tensor."""
    return torch.optim.Adam([sample_tensor.requires_grad_(True)], lr=0.01)  # noqa: FBT003


@pytest.fixture
def video_path(test_dir: str) -> Path:
    """
    Return the expected path to a test video file.

    Note:
        This does not create a video file.

    """
    return Path(test_dir) / "test_video.mp4"


@pytest.fixture(autouse=True)
def enable_logger_propagation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable propagation for visualizer logger to allow caplog to work."""
    monkeypatch.setattr(logger, "propagate", True)
