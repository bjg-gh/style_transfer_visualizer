"""Command-line entry point for comparison grid rendering."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from style_transfer_visualizer.gallery import (
    FRAME_CHOICES,
    LAYOUT_CHOICES,
    ComparisonRenderOptions,
    parse_wall_color,
    positive_int,
    render_comparison,
    size_2d,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Sequence

T = TypeVar("T")


def _wrap_validator[T](
    validator: Callable[[str], T],
    error_cls: type[argparse.ArgumentTypeError] = argparse.ArgumentTypeError,
) -> Callable[[str], T]:
    """Convert ``ValueError`` from a validator into ``ArgumentTypeError``."""

    def wrapper(text: str) -> T:
        try:
            return validator(text)
        except ValueError as exc:
            raise error_cls(str(exc)) from exc

    return wrapper


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the comparison tool."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a 3-panel comparison from content, style, and result. "
            "Optionally render as a framed gallery wall."
        ),
    )
    parser.add_argument("--content", required=True, type=Path)
    parser.add_argument("--style", required=True, type=Path)
    parser.add_argument("--result", required=False, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--target-height",
        type=_wrap_validator(positive_int),
        default=512,
    )
    parser.add_argument("--pad", type=_wrap_validator(positive_int),
                         default=16)
    parser.add_argument(
        "--border-px",
        type=_wrap_validator(positive_int),
        default=0,
    )
    parser.add_argument(
        "--target-size",
        type=_wrap_validator(size_2d),
        default=None,
        help="Exact WxH for video safe output, e.g., 1920x1080.",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default=None,
        choices=list(LAYOUT_CHOICES),
        help=(
            "If provided, render as a gallery wall. "
            "Select two or three panel layout."
        ),
    )
    parser.add_argument(
        "--wall",
        type=str,
        default="#3c434a",
        help="Wall color as hex like #3c434a.",
    )
    parser.add_argument(
        "--frame-style",
        type=str,
        default="gold",
        choices=list(FRAME_CHOICES),
        help="Quick frame tone preset.",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Draw Content, Style, and Final labels.",
    )
    return parser


def _build_options(args: argparse.Namespace) -> ComparisonRenderOptions:
    """Map argparse namespace to :class:`ComparisonRenderOptions`."""
    return ComparisonRenderOptions(
        content_path=args.content,
        style_path=args.style,
        result_path=args.result,
        out_path=args.out,
        target_height=args.target_height,
        pad=args.pad,
        border_px=args.border_px,
        target_size=args.target_size,
        layout=args.layout,
        wall_color=parse_wall_color(args.wall),
        frame_style=args.frame_style,
        show_labels=args.show_labels,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments and render the comparison image."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.layout is None and args.result is None:
        parser.error("result is required when not using gallery layout")

    options = _build_options(args)

    try:
        render_comparison(options)
    except ValueError as exc:
        parser.error(str(exc))

    return 0


__all__ = ["build_parser", "main"]
