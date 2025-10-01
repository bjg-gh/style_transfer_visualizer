
"""
Standalone runner for the comparison image grid.

This is a helper for quick manual testing without running the full
pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from style_transfer_visualizer.constants import RESOLUTION_FULL_HD

try:
    from style_transfer_visualizer.image_grid import (
        default_comparison_name,
        save_comparison_grid,
        save_gallery_comparison,
    )
except Exception as exc:  # pragma: no cover
    msg = "Failed to import image_grid. Ensure package is on PYTHONPATH."
    raise SystemExit(msg) from exc
from style_transfer_visualizer.logging_utils import logger

_TARGET_SIZE_PARTS: int = 2
_HEX_RGB_LENGTH: int = 6

def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for the tool."""
    p = argparse.ArgumentParser(
        description=(
            "Build a 3-panel comparison from content, style, and result. "
            "Optionally render as a framed gallery wall."
        ),
    )
    p.add_argument("--content", required=True, type=Path)
    p.add_argument("--style", required=True, type=Path)
    p.add_argument("--result", required=False, type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--target-height", type=positive_int, default=512)
    p.add_argument("--pad", type=positive_int, default=16)
    p.add_argument("--border-px", type=positive_int, default=0)
    p.add_argument(
        "--target-size",
        type=size_2d,
        default=None,
        help="Exact WxH for video safe output, e.g., 1920x1080.",
    )
    # gallery options
    p.add_argument(
        "--layout",
        type=str,
        default=None,
        choices=["gallery-stacked-left", "gallery-two-across"],
        help=(
            "If provided, render as a gallery wall. "
            "Select two or three panel layout."
        ),
    )
    p.add_argument(
        "--wall",
        type=str,
        default="#3c434a",
        help="Wall color as hex like #3c434a.",
    )
    p.add_argument(
        "--frame-style",
        type=str,
        default="gold",
        choices=["gold", "oak", "black"],
        help="Quick frame tone preset.",
    )
    p.add_argument(
        "--show-labels",
        action="store_true",
        help="Draw Content, Style, and Final labels.",
    )
    return p


def positive_int(text: str) -> int:
    """Argparse type that enforces a strictly positive int."""
    try:
        value = int(text)
    except ValueError as e:
        msg_int = "must be an integer"
        raise argparse.ArgumentTypeError(msg_int) from e
    if value <= 0:
        msg_positive = "must be positive"
        raise argparse.ArgumentTypeError(msg_positive)
    return value


def size_2d(text: str) -> tuple[int, int]:
    """Argparse type that parses WxH and enforces positivity."""
    parts = text.lower().split("x")
    if len(parts) != _TARGET_SIZE_PARTS:
        msg_dimensions = "must look like WxH, e.g., 1920x1080"
        raise argparse.ArgumentTypeError(msg_dimensions)
    try:
        w, h = int(parts[0]), int(parts[1])
    except ValueError as e:
        msg_int = "width and height must be integers"
        raise argparse.ArgumentTypeError(msg_int) from e
    if w <= 0 or h <= 0:
        msg_positive = "width and height must be positive"
        raise argparse.ArgumentTypeError(msg_positive)
    return w, h


def _parse_hex_color(text: str) -> tuple[int, int, int]:
    """Parse #rrggbb into an RGB tuple."""
    t = text.strip().lstrip("#")
    if len(t) != _HEX_RGB_LENGTH:
        err_msg = "wall color must look like #rrggbb"
        raise argparse.ArgumentTypeError(err_msg)
    try:
        r = int(t[0:2], 16)
        g = int(t[2:4], 16)
        b = int(t[4:6], 16)
    except ValueError as e:
        err_msg = "wall color contains invalid hex digits"
        raise argparse.ArgumentTypeError(err_msg) from e
    return r, g, b


def main() -> int:
    """Parse args, build grid or gallery, save, return exit code."""
    parser = build_parser()
    args = parser.parse_args()

    out_path = args.out
    if out_path is None:
        out_dir = Path()
        out_path = default_comparison_name(args.content, args.style, out_dir)
    if out_path.suffix.lower() != ".png":
        out_path = out_path.with_suffix(".png")

    size = args.target_size

    # Branch by layout. None keeps current simple grid behavior.
    if args.layout is None:
        if args.result is None:
            err_msg = "result is required when not using gallery layout"
            raise SystemExit(err_msg)

        save_comparison_grid(
            content_path=args.content,
            style_path=args.style,
            result_path=cast("Path", args.result),
            out_path=out_path,
            target_height=args.target_height if size is None else None,
            target_size=size,
            pad=args.pad,
            border_px=args.border_px,
        )
    else:
        wall_rgb = _parse_hex_color(args.wall)
        # When using two across we ignore result even if given
        result_path = (
            args.result if args.layout != "gallery-two-across" else None
        )
        ts = size if size is not None else RESOLUTION_FULL_HD
        save_gallery_comparison(
            content_path=args.content,
            style_path=args.style,
            result_path=result_path,
            out_path=out_path,
            target_size=ts,
            layout=args.layout,
            wall_color=wall_rgb,
            frame_tone=args.frame_style,
            show_labels=args.show_labels,
        )

    logger.info("Comparison image saved to: %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
