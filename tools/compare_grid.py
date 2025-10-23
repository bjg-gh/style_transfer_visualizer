
"""Compatibility wrapper around the shared gallery CLI."""

from __future__ import annotations

from style_transfer_visualizer.gallery import (
    parse_wall_color as _parse_hex_color,
    positive_int,
    size_2d,
)
from style_transfer_visualizer.gallery.cli import build_parser, main

__all__ = [
    "_parse_hex_color",
    "build_parser",
    "main",
    "positive_int",
    "size_2d",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
