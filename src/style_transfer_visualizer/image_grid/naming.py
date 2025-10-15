"""Path and persistence helpers for comparison grid outputs."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from style_transfer_visualizer.constants import (
    COLOR_GREY,
    COLOR_WHITE,
    RESOLUTION_FULL_HD,
)
from style_transfer_visualizer.image_grid.core import (
    DEFAULT_HEIGHT,
    DEFAULT_PAD,
    FrameParams,
    to_rgb,
)
from style_transfer_visualizer.image_grid.layouts import (
    make_gallery_comparison,
    make_horizontal_grid,
)

if TYPE_CHECKING:  # pragma: no cover
    from style_transfer_visualizer.type_defs import LayoutName
else:
    LayoutName = str  # type: ignore[assignment]

_RGB = tuple[int, int, int]


def save_comparison_grid(  # noqa: PLR0913
    content_path: Path,
    style_path: Path,
    result_path: Path,
    out_path: Path,
    *,
    target_height: int | None = DEFAULT_HEIGHT,
    target_size: tuple[int, int] | None = None,
    pad: int = DEFAULT_PAD,
    bg_color: _RGB = COLOR_WHITE,
    border_px: int = 0,
) -> Path:
    """Open three images, build a grid, and save to out_path."""
    if not isinstance(out_path, Path):
        msg = "out_path must be a pathlib.Path"
        raise TypeError(msg)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(content_path) as content, \
         Image.open(style_path) as style, \
         Image.open(result_path) as result:

        grid = make_horizontal_grid(
            [
                to_rgb(content, bg_color=bg_color),
                to_rgb(style, bg_color=bg_color),
                to_rgb(result, bg_color=bg_color),
            ],
            target_height=target_height,
            target_size=target_size,
            pad=pad,
            bg_color=bg_color,
            border_px=border_px,
        )
        grid.save(out_path, format="PNG")
    return out_path


def default_comparison_name(
    content_path: Path,
    style_path: Path,
    out_dir: Path,
) -> Path:
    """Build a deterministic filename for the comparison image."""
    def stem(p: Path) -> str:
        return p.stem.replace(" ", "_")

    name = f"comparison_{stem(content_path)}_x_{stem(style_path)}.png"
    return out_dir / name


def save_gallery_comparison(  # noqa: PLR0913
    content_path: Path,
    style_path: Path,
    result_path: Path | None,
    out_path: Path,
    *,
    target_size: tuple[int, int] = RESOLUTION_FULL_HD,
    layout: LayoutName = "gallery-stacked-left",
    wall_color: _RGB = COLOR_GREY,
    frame_tone: str = "gold",
    show_labels: bool = True,
) -> Path:
    """Open images, build a gallery wall, and save to out_path."""
    if not isinstance(out_path, Path):
        msg = "out_path must be a pathlib.Path"
        raise TypeError(msg)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # noinspection PyAbstractClass
    with ExitStack() as stack:
        content = stack.enter_context(Image.open(content_path))
        style = stack.enter_context(Image.open(style_path))
        result = (
            stack.enter_context(Image.open(result_path))
            if result_path
            else None
        )

        fparams = FrameParams(
            frame_tone=frame_tone,
            label="on" if show_labels else None,
        )
        img = make_gallery_comparison(
            content=content,
            style=style,
            result=result,
            target_size=target_size,
            layout=layout,
            wall_color=wall_color,
            frame=fparams,
        )
        img.save(out_path, format="PNG")
    return out_path
