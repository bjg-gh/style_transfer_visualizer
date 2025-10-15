"""Layout orchestration for image comparison grids and gallery walls."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from PIL import Image

from style_transfer_visualizer.constants import (
    COLOR_BLACK,
    COLOR_GREY,
    COLOR_WHITE,
    RESOLUTION_FULL_HD,
)
from style_transfer_visualizer.image_grid.core import (
    DEFAULT_HEIGHT,
    DEFAULT_PAD,
    FRAME_TEXTURE_MAX,
    FrameParams,
    Rect,
    build_framed_panel,
    content_dimensions,
    draw_border,
    draw_label,
    fit_box_by_inner_aspect,
    make_wall_canvas,
    paste_horizontally,
    scale_images_to_fit_canvas,
    scale_images_to_target,
    to_rgb,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from style_transfer_visualizer.type_defs import LayoutName
else:
    Sequence = tuple  # type: ignore[assignment]
    LayoutName = str  # type: ignore[assignment]

_RGB = tuple[int, int, int]

# Index constants for gallery panels
_CONTENT_IDX = 0
_STYLE_IDX = 1
_RESULT_IDX = 2

# Layout proportions and spacing
_GAP_FRACTION = 0.02              # used for panel gaps
_LEFT_COL_FRACTION = 0.42         # stacked-left layout left column width
_RESULT_INSET_FRACTION = 0.06     # inset for final panel box


@dataclass(frozen=True)
class GridParams:
    """Parameters for the grid layout."""

    target_height: int | None = DEFAULT_HEIGHT
    target_size: tuple[int, int] | None = None
    pad: int = DEFAULT_PAD
    bg_color: _RGB = COLOR_WHITE
    border_px: int = 0


def make_horizontal_grid(  # noqa: PLR0913
    images: Sequence[Image.Image],
    *,
    target_height: int | None = DEFAULT_HEIGHT,
    target_size: tuple[int, int] | None = None,
    pad: int = DEFAULT_PAD,
    bg_color: _RGB = COLOR_WHITE,
    border_px: int = 0,
) -> Image.Image:
    """
    Build a horizontal grid with N panels.

    If target_size is given, scale the composed content to fit within
    that size and center on a canvas of exact dimensions. Stretching
    above 1.0 scale is not performed. If only target_height is given,
    scale panels to that height, keep aspect, and size tightly.
    """
    if not images:
        msg = "No images provided"
        raise ValueError(msg)

    rgb_imgs = [to_rgb(im, bg_color=bg_color) for im in images]
    work_imgs = scale_images_to_target(rgb_imgs, target_height, target_size)
    work_imgs = [draw_border(im, border_px) for im in work_imgs]

    inner_gap = pad
    outer_pad = pad
    content_w, content_h, _, _ = content_dimensions(work_imgs, inner_gap)
    tight_w = content_w + 2 * outer_pad
    tight_h = content_h + 2 * outer_pad

    # canvas sizing and optional downscale
    if target_size is None:
        canvas_w, canvas_h = tight_w, tight_h
    else:
        work_imgs, content_w, content_h = scale_images_to_fit_canvas(
            work_imgs, inner_gap, tight_w, tight_h, target_size,
        )
        canvas_w, canvas_h = target_size

    # compose
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    if target_size is None:
        start_x = outer_pad
        y = outer_pad
    else:
        start_x = (canvas_w - content_w) // 2
        y = (canvas_h - content_h) // 2

    paste_horizontally(canvas, work_imgs, inner_gap, (start_x, y), content_h)
    return canvas


def _layout_two_across(
    w: int,
    h: int,
    *,
    lr_margin: int,
    tb_margin: int,
    gap_frac: float,
) -> list[Rect]:
    """Return outer boxes for two side by side panels."""
    gap = int(w * gap_frac)
    avail_w = w - 2 * lr_margin - gap
    avail_h = h - 2 * tb_margin
    panel_w = avail_w // 2
    panel_h = avail_h
    y0 = (h - panel_h) // 2

    left = Rect(lr_margin, y0, lr_margin + panel_w, y0 + panel_h)
    right = Rect(lr_margin + panel_w + gap, y0,
                 lr_margin + panel_w + gap + panel_w, y0 + panel_h)
    return [left, right]


def _layout_stacked_left(  # noqa: PLR0913
    w: int,
    h: int,
    *,
    lr_margin: int,
    tb_margin: int,
    gap_frac: float,
    left_col_frac: float,
) -> list[Rect]:
    """Return outer boxes for stacked left plus tall right panel."""
    gap = int(w * gap_frac)
    col_w = int((w - 2 * lr_margin - gap) * left_col_frac)
    right_w = w - 2 * lr_margin - gap - col_w
    avail_h = h - 2 * tb_margin
    top_h = (avail_h - gap) // 2
    bottom_h = avail_h - gap - top_h

    x0 = lr_margin
    y0 = tb_margin
    return [
        Rect(x0, y0, x0 + col_w, y0 + top_h),
        Rect(x0, y0 + top_h + gap, x0 + col_w, y0 + top_h + gap + bottom_h),
        Rect(x0 + col_w + gap, y0, x0 + col_w + gap + right_w, y0 + avail_h),
    ]


def _render_panels(  # noqa: PLR0913
    canvas: Image.Image,
    images: list[Image.Image],
    boxes: list[Rect],
    fparams: FrameParams,
    *,
    wall_color: _RGB,
    two_image: bool,
) -> list[tuple[int, int]]:
    """Render framed panels and paste to canvas. Return label anchors."""
    anchors: list[tuple[int, int]] = []
    for idx, (im, box) in enumerate(zip(images, boxes, strict=True)):
        w_box, h_box = box.size()
        local_params = fparams
        if two_image or idx == _RESULT_IDX:
            local_params = replace(fparams, fit_mode="contain")

        panel, anchor = build_framed_panel(
            to_rgb(im, bg_color=COLOR_BLACK),
            (w_box, h_box),
            local_params,
            wall_color=wall_color,
        )
        anchors.append((box.x0 + anchor[0], box.y0 + anchor[1]))
        canvas.paste(panel, (box.x0, box.y0))
    return anchors


def make_gallery_comparison(  # noqa: PLR0913
    content: Image.Image,
    style: Image.Image,
    result: Image.Image | None,
    *,
    target_size: tuple[int, int] = RESOLUTION_FULL_HD,
    layout: LayoutName = "gallery-stacked-left",
    wall_color: _RGB = COLOR_GREY,
    frame: FrameParams | None = None,
    labels: tuple[str, str, str] = ("Content", "Style", "Final"),
    left_right_wall_margin: int = 48,
    top_bottom_wall_margin: int = 48,
) -> Image.Image:
    """
    Build the gallery wall comparison image.

    Supports two panel and three panel layouts. If result is None the
    two panel layout is used regardless of layout name.
    """
    two_image = (result is None) or (layout == "gallery-two-across")

    w, h = target_size
    if w <= 0 or h <= 0:
        msg = "target_size must be positive"
        raise ValueError(msg)

    # clamp texture strength into a safe range
    fparams = frame or FrameParams()
    if fparams.frame_texture_strength < 0:
        fparams = replace(fparams, frame_texture_strength=0)
    elif fparams.frame_texture_strength > FRAME_TEXTURE_MAX:
        fparams = replace(fparams, frame_texture_strength=FRAME_TEXTURE_MAX)

    wall = make_wall_canvas((w, h), wall_color, vignette=True, noise=True)

    # Layout outer boxes
    if two_image:
        boxes = _layout_two_across(
            w, h, lr_margin=left_right_wall_margin,
            tb_margin=top_bottom_wall_margin, gap_frac=_GAP_FRACTION,
        )
        imgs: list[Image.Image] = [content, style]
        labs: tuple[str, ...] = labels[:2]
        # Fit each panel by image aspect, include a small inset
        boxes = [
            fit_box_by_inner_aspect(b, im, fparams, _RESULT_INSET_FRACTION)
            for b, im in zip(boxes, imgs, strict=True)
        ]
    else:
        boxes = _layout_stacked_left(
            w, h, lr_margin=left_right_wall_margin,
            tb_margin=top_bottom_wall_margin, gap_frac=_GAP_FRACTION,
            left_col_frac=_LEFT_COL_FRACTION,
        )
        imgs = [content, style, result]  # type: ignore[list-item]
        labs = labels
        # Inset and fit the result column only
        boxes[_RESULT_IDX] = fit_box_by_inner_aspect(
            boxes[_RESULT_IDX], imgs[_RESULT_IDX],
            fparams, _RESULT_INSET_FRACTION,
        )

    # render and paste
    canvas = wall.copy()
    anchors = _render_panels(
        canvas, imgs, boxes, fparams, wall_color=wall_color,
        two_image=two_image,
    )

    # labels
    if fparams.label is not None:
        for text, center in zip(labs, anchors, strict=True):
            draw_label(
                canvas,
                center=center,
                text=text,
                px=fparams.label_px,
                fill=fparams.label_fill,
                y_offset=fparams.label_offset_px,
            )

    return canvas
