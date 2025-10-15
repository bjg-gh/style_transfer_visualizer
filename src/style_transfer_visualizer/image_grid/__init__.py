"""
Image grid utilities split into core primitives, layouts, and naming helpers.

The package exposes the most commonly used entry points directly to keep
the public API compatible with the previous single-module layout.
"""

from __future__ import annotations

from . import core, layouts, naming
from .core import (
    DEFAULT_HEIGHT,
    DEFAULT_PAD,
    FrameParams,
    Rect,
    build_framed_panel,
    make_wall_canvas,
    to_rgb,
)
from .layouts import (
    make_gallery_comparison,
    make_horizontal_grid,
)
from .naming import (
    default_comparison_name,
    save_comparison_grid,
    save_gallery_comparison,
)

__all__ = [
    "DEFAULT_HEIGHT",
    "DEFAULT_PAD",
    "FrameParams",
    "Rect",
    "build_framed_panel",
    "core",
    "default_comparison_name",
    "layouts",
    "make_gallery_comparison",
    "make_horizontal_grid",
    "make_wall_canvas",
    "naming",
    "save_comparison_grid",
    "save_gallery_comparison",
    "to_rgb",
]
