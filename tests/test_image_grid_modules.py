"""Smoke tests covering the public image_grid submodules."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from style_transfer_visualizer.constants import COLOR_GREY, COLOR_WHITE
from style_transfer_visualizer.image_grid import core, layouts, naming

pytestmark = pytest.mark.visual


def test_core_builds_panel(sample_image: Image.Image) -> None:
    """Core primitives can frame an image without error."""
    panel, anchor = core.build_framed_panel(
        sample_image,
        (64, 64),
        core.FrameParams(label=None),
        wall_color=COLOR_GREY,
    )
    assert panel.size == (64, 64)
    assert anchor == (32, 64)


def test_layouts_horizontal_grid(sample_image: Image.Image) -> None:
    """Layouts module composes a simple horizontal grid."""
    grid = layouts.make_horizontal_grid(
        [sample_image, sample_image],
        target_height=sample_image.height,
        pad=8,
        bg_color=COLOR_WHITE,
    )
    assert grid.size[1] == sample_image.height + 16  # top/bottom padding
    assert grid.mode == "RGB"


def test_naming_default_name(tmp_path: Path) -> None:
    """Naming helpers derive comparison filenames deterministically."""
    path = naming.default_comparison_name(
        Path("content sample.png"),
        Path("style-image.jpg"),
        tmp_path,
    )
    assert path.name == "comparison_content_sample_x_style-image.png"
