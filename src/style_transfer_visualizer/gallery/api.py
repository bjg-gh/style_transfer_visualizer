"""
Reusable comparison grid rendering API shared by CLI, tools, and tests.

The module exposes a dataclass-based configuration object alongside
helpers for parsing CLI-style arguments. Rendering is delegated to the
existing ``style_transfer_visualizer.image_grid`` helpers so callers can
invoke a single entry point regardless of whether they need the simple
three-panel grid or the gallery wall layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from style_transfer_visualizer.constants import COLOR_GREY, RESOLUTION_FULL_HD
from style_transfer_visualizer.image_grid import (
    default_comparison_name,
    save_comparison_grid,
    save_gallery_comparison,
)
from style_transfer_visualizer.logging_utils import logger

GalleryLayout = Literal["gallery-stacked-left", "gallery-two-across"]
FrameStyle = Literal["gold", "oak", "black"]

LAYOUT_CHOICES: tuple[GalleryLayout, ...] = (
    "gallery-stacked-left",
    "gallery-two-across",
)
FRAME_CHOICES: tuple[FrameStyle, ...] = ("gold", "oak", "black")

_TARGET_SIZE_PARTS = 2
_HEX_RGB_LENGTH = 6


@dataclass(slots=True)
class ComparisonRenderOptions:
    """
    Configuration for comparison rendering.

    The dataclass deliberately mirrors the options exposed by
    ``tools/compare_grid.py`` so existing consumers can pass parsed CLI
    arguments directly to :func:`render_comparison`.
    """

    content_path: Path
    style_path: Path
    result_path: Path | None = None
    out_path: Path | None = None
    target_height: int = 512
    pad: int = 16
    border_px: int = 0
    target_size: tuple[int, int] | None = None
    layout: GalleryLayout | None = None
    wall_color: tuple[int, int, int] = COLOR_GREY
    frame_style: FrameStyle = "gold"
    show_labels: bool = False


def positive_int(text: str) -> int:
    """Argparse-style validator that enforces a strictly positive integer."""
    try:
        value = int(text)
    except ValueError as exc:
        msg = "must be an integer"
        raise ValueError(msg) from exc
    if value <= 0:
        msg = "must be positive"
        raise ValueError(msg)
    return value


def size_2d(text: str) -> tuple[int, int]:
    """Parse ``WxH`` strings into integer tuples and validate positivity."""
    parts = text.lower().split("x")
    if len(parts) != _TARGET_SIZE_PARTS:
        msg = "must look like WxH, e.g., 1920x1080"
        raise ValueError(msg)
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError as exc:
        msg = "width and height must be integers"
        raise ValueError(msg) from exc
    if width <= 0 or height <= 0:
        msg = "width and height must be positive"
        raise ValueError(msg)
    return width, height


def parse_wall_color(text: str) -> tuple[int, int, int]:
    """Parse ``#rrggbb`` strings into RGB triples."""
    stripped = text.strip().lstrip("#")
    if len(stripped) != _HEX_RGB_LENGTH:
        msg = "wall color must look like #rrggbb"
        raise ValueError(msg)
    try:
        red = int(stripped[0:2], 16)
        green = int(stripped[2:4], 16)
        blue = int(stripped[4:6], 16)
    except ValueError as exc:
        msg = "wall color contains invalid hex digits"
        raise ValueError(msg) from exc
    return red, green, blue


def _ensure_png(path: Path) -> Path:
    """Return a path that ends with ``.png`` for output consistency."""
    return path if path.suffix.lower() == ".png" else path.with_suffix(".png")


def _resolve_output_path(
    *,
    content_path: Path,
    style_path: Path,
    out_path: Path | None,
) -> Path:
    """Determine the output path, falling back to deterministic naming."""
    if out_path is None:
        return default_comparison_name(content_path, style_path, Path())
    return Path(out_path)


def render_comparison(options: ComparisonRenderOptions) -> Path:
    """
    Render a comparison grid or gallery based on ``options``.

    Returns the saved ``Path``. Errors are surfaced as :class:`ValueError`
    when options are inconsistent (for example requesting a grid without a
    result image). The underlying image helpers raise their own exceptions
    for I/O or Pillow-specific failures.
    """
    content_path = Path(options.content_path)
    style_path = Path(options.style_path)
    result_path = Path(options.result_path) if options.result_path else None

    out_path = _ensure_png(
        _resolve_output_path(
            content_path=content_path,
            style_path=style_path,
            out_path=options.out_path,
        ),
    )

    if options.layout is None:
        if result_path is None:
            msg = "result_path is required when layout is None"
            raise ValueError(msg)
        grid_target_height = (
            options.target_height if options.target_size is None else None
        )
        saved = save_comparison_grid(
            content_path=content_path,
            style_path=style_path,
            result_path=result_path,
            out_path=out_path,
            target_height=grid_target_height,
            target_size=options.target_size,
            pad=options.pad,
            border_px=options.border_px,
        )
    else:
        layout = options.layout
        target_size = options.target_size or RESOLUTION_FULL_HD
        effective_result = (
            None if layout == "gallery-two-across" else result_path
        )
        saved = save_gallery_comparison(
            content_path=content_path,
            style_path=style_path,
            result_path=effective_result,
            out_path=out_path,
            target_size=target_size,
            layout=layout,
            wall_color=options.wall_color,
            frame_tone=options.frame_style,
            show_labels=options.show_labels,
        )

    logger.info("Comparison image saved to: %s", saved)
    return saved


__all__ = [
    "FRAME_CHOICES",
    "LAYOUT_CHOICES",
    "ComparisonRenderOptions",
    "FrameStyle",
    "GalleryLayout",
    "parse_wall_color",
    "positive_int",
    "render_comparison",
    "size_2d",
]
