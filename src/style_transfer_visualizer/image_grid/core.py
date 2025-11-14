"""Core rendering primitives for building image comparison grids."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from style_transfer_visualizer import random_utils as stv_random
from style_transfer_visualizer.constants import (
    COLOR_BEIGE,
    COLOR_BLACK,
    COLOR_WHITE,
)

_RGB = tuple[int, int, int]

# Texture and rendering parameters
FRAME_TEXTURE_MAX = 100
_MIN_FRAME_OUTER_PX = 3
_MIN_FRAME_INNER_PX = 2
_BEVEL_ALPHA_MAX = 120
_SHADOW_ALPHA = 130
_NOISE_EFFECT_SCALE = 8.0
_NOISE_GAUSS_RADIUS = 2
_BLEND_MAX = 0.25
_ASPECT_SOLVE_ITERS = 6

# Defaults shared with layouts
DEFAULT_HEIGHT = 512
DEFAULT_PAD = 16

# Wall rendering
_WALL_GRADIENT_CENTER = 220       # center luminance for vertical gradient
_WALL_GRADIENT_RANGE = 20         # range around center
_VIGNETTE_MARGIN_FRAC = 0.06      # vignette rectangle margin fraction


def to_rgb(img: Image.Image, *, bg_color: _RGB) -> Image.Image:
    """Convert PIL image to RGB, alpha compositing if needed."""
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", img.size, (*bg_color, 255))
        comp = Image.alpha_composite(bg, img.convert("RGBA"))
        return comp.convert("RGB")
    return img.convert("RGB")


def _resize_to_height(img: Image.Image, height: int) -> Image.Image:
    """Resize keeping aspect so that resulting height matches."""
    w, h = img.size
    if h <= 0:
        msg = "Input image has zero height"
        raise ValueError(msg)
    scale = height / h
    new_w = max(1, round(w * scale))
    return img.resize((new_w, height), Image.Resampling.LANCZOS)


def draw_border(img: Image.Image, border_px: int) -> Image.Image:
    """Add a thin border around the image if requested."""
    if border_px <= 0:
        return img
    return ImageOps.expand(img, border=border_px, fill=COLOR_BLACK)


def scale_images_to_target(
    images: list[Image.Image],
    target_height: int | None,
    target_size: tuple[int, int] | None,
) -> list[Image.Image]:
    """Resize images by target height unless exact target_size is set."""
    if target_size is not None and target_height is None:
        return images
    work_h = target_height or DEFAULT_HEIGHT
    return [_resize_to_height(im, work_h) for im in images]


def content_dimensions(
    images: list[Image.Image],
    pad: int,
) -> tuple[int, int, list[int], list[int]]:
    """Return tight content width, height and per-panel dims."""
    widths = [im.size[0] for im in images]
    heights = [im.size[1] for im in images]
    content_w = sum(widths) + pad * (len(images) - 1)
    content_h = max(heights) if heights else 0
    return content_w, content_h, widths, heights


def scale_images_to_fit_canvas(
    images: list[Image.Image],
    pad: int,
    tight_w: int,
    tight_h: int,
    target_size: tuple[int, int],
) -> tuple[list[Image.Image], int, int]:
    """
    Scale images down uniformly so the tight layout fits target canvas.

    Returns updated images and recomputed content width and height.
    """
    target_w, target_h = target_size
    scale_w = target_w / tight_w
    scale_h = target_h / tight_h
    scale = min(1.0, scale_w, scale_h)
    if scale >= 1.0:
        cw, ch, _, _ = content_dimensions(images, pad)
        return images, cw, ch

    def scale_im(im: Image.Image) -> Image.Image:
        w, h = im.size
        return im.resize(
            (max(1, round(w * scale)), max(1, round(h * scale))),
            Image.Resampling.LANCZOS,
        )

    scaled = [scale_im(im) for im in images]
    cw, ch, _, _ = content_dimensions(scaled, pad)
    return scaled, cw, ch


def paste_horizontally(
    canvas: Image.Image,
    images: list[Image.Image],
    pad: int,
    start_xy: tuple[int, int],
    row_height: int,
) -> None:
    """Paste images onto canvas with a fixed vertical center."""
    x, y = start_xy
    for im in images:
        im_h = im.size[1]
        y_offset = y + (row_height - im_h) // 2
        canvas.paste(im, (x, y_offset))
        x += im.size[0] + pad


@dataclass(frozen=True)
class FrameParams:
    """Appearance configuration for a framed panel."""

    matte_frac: float = 0.0
    frame_outer_frac: float = 0.035
    frame_inner_frac: float = 0.02
    bevel_px: int = 3
    shadow_radius: int = 12
    shadow_offset: tuple[int, int] = (6, 6)
    frame_tone: str = "gold"  # gold, oak, black

    # Control how the image fills the frame opening
    fit_mode: Literal["cover", "contain"] = "cover"

    # Subtle textures
    frame_texture_strength: int = 18   # 0 disables
    label: str | None = None
    label_px: int = 30
    label_fill: _RGB = (235, 235, 235)
    label_offset_px: int = 2


@dataclass(frozen=True)
class _FrameThickness:
    """Per-side thickness of matte and frame bands in pixels."""

    matte: int
    outer: int
    inner: int

    @property
    def total(self) -> int:
        """Return total per-side thickness."""
        return self.matte + self.outer + self.inner


def _frame_thickness(
    panel_w: int,
    panel_h: int,
    params: FrameParams,
) -> _FrameThickness:
    """Compute per-side thicknesses for matte and frame bands."""
    s = min(panel_w, panel_h)
    matte = int(max(0, round(params.matte_frac * s)))
    outer = int(max(_MIN_FRAME_OUTER_PX, round(params.frame_outer_frac * s)))
    inner = int(max(_MIN_FRAME_INNER_PX, round(params.frame_inner_frac * s)))
    return _FrameThickness(matte=matte, outer=outer, inner=inner)


def _panel_margin_px(params: FrameParams, panel_w: int, panel_h: int) -> int:
    """Return total per side thickness in pixels for frame plus matte."""
    return _frame_thickness(panel_w, panel_h, params).total


@dataclass(frozen=True)
class Rect:
    """Simple rectangle with convenience accessors."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        """Width."""
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        """Height."""
        return self.y1 - self.y0

    def size(self) -> tuple[int, int]:
        """Return (w, h)."""
        return self.w, self.h

    def move_to(self, x: int, y: int) -> Rect:
        """Return a copy moved so its top left is at (x, y)."""
        return Rect(x, y, x + self.w, y + self.h)

    def inset(self, dx: int, dy: int) -> Rect:
        """Return a copy inset by (dx, dy) on all sides."""
        return Rect(self.x0 + dx, self.y0 + dy, self.x1 - dx, self.y1 - dy)


def _tone_colors(tone: str) -> tuple[_RGB, _RGB, _RGB]:
    """Return three band colors for the frame."""
    t = tone.lower()
    if t == "oak":
        return (115, 85, 45), (150, 115, 70), (90, 65, 35)
    if t == "black":
        return (25, 25, 25), (40, 40, 40), (15, 15, 15)
    # default gold
    return (110, 85, 35), (170, 140, 70), (80, 60, 25)


def _fit_panel_box_to_inner_aspect(
    avail_box: tuple[int, int, int, int],
    target_aspect: float,
    params: FrameParams,
) -> tuple[int, int, int, int]:
    """
    Fit panel box to inner aspect ratio.

    Within avail_box, choose a panel box whose inner opening matches
    target_aspect after subtracting frame and matte thickness.
    """
    ax0, ay0, ax1, ay1 = avail_box
    aw, ah = ax1 - ax0, ay1 - ay0

    # Start from the full available area and solve by fixed point
    # iteration. Converges fast in practice.
    pw, ph = aw, ah
    for _ in range(_ASPECT_SOLVE_ITERS):
        margin = _panel_margin_px(params, pw, ph)
        iw_max = max(1, aw - 2 * margin)
        ih_max = max(1, ah - 2 * margin)

        # Fit the largest inner rectangle with the target aspect
        if iw_max / ih_max >= target_aspect:
            ih = ih_max
            iw = round(ih * target_aspect)
        else:
            iw = iw_max
            ih = round(iw / target_aspect)

        new_pw = iw + 2 * margin
        new_ph = ih + 2 * margin

        # Clamp and break if stable
        new_pw = min(new_pw, aw)
        new_ph = min(new_ph, ah)
        if new_pw == pw and new_ph == ph:
            break
        pw, ph = new_pw, new_ph

    nx0 = ax0 + (aw - pw) // 2
    ny0 = ay0 + (ah - ph) // 2
    return nx0, ny0, nx0 + pw, ny0 + ph


def fit_box_by_inner_aspect(
    box: Rect,
    img: Image.Image,
    params: FrameParams,
    inset_frac: float,
) -> Rect:
    """Return a panel box fitted so the inner opening matches image aspect."""
    if img.size[1] <= 0:
        msg = "Image height must be positive"
        raise ValueError(msg)
    aspect = img.size[0] / img.size[1]
    avail = box.inset(int(box.w * inset_frac / 2),
                      int(box.h * inset_frac / 2))
    return Rect(*_fit_panel_box_to_inner_aspect(
        (avail.x0, avail.y0, avail.x1, avail.y1), aspect, params,
    ))


def _fit_and_place(
    img: Image.Image,
    inner_size: tuple[int, int],
    matte_px: int,
    *,
    fit_mode: str = "cover",
) -> Image.Image:
    """
    Resize img to fill inner_size, then place on a matte.

    fit_mode:
        - "cover": fills inner box and crops to avoid letterbox
        - "contain": letterboxes to preserve all content
    """
    iw, ih = img.size
    mw = max(1, inner_size[0] + 2 * matte_px)
    mh = max(1, inner_size[1] + 2 * matte_px)

    if fit_mode == "cover":
        cropped = ImageOps.fit(
            img,
            inner_size,
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
    else:  # "contain"
        scale = min(inner_size[0] / iw, inner_size[1] / ih)
        rw, rh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = img.resize((rw, rh), Image.Resampling.LANCZOS)
        cropped = Image.new("RGB", inner_size, COLOR_BEIGE)
        cx = (inner_size[0] - rw) // 2
        cy = (inner_size[1] - rh) // 2
        cropped.paste(resized, (cx, cy))

    matte = Image.new("RGB", (mw, mh), COLOR_BEIGE)
    matte.paste(cropped, (matte_px, matte_px))
    return matte


def build_framed_panel(
    image: Image.Image,
    panel_box: tuple[int, int],
    params: FrameParams,
    *,
    wall_color: _RGB,
) -> tuple[Image.Image, tuple[int, int]]:
    """
    Render a single framed panel for the given area.

    Returns the composed panel and the label anchor offset.
    """
    panel_w, panel_h = panel_box
    base = Image.new("RGBA", (panel_w, panel_h), (*wall_color, 0))

    # sizes
    t = _frame_thickness(panel_w, panel_h, params)
    bevel = max(0, params.bevel_px)

    inner_w = panel_w - 2 * t.total
    inner_h = panel_h - 2 * t.total
    inner_w = max(8, inner_w)
    inner_h = max(8, inner_h)

    # image placed to fit inner opening
    matte_img = _fit_and_place(
        image,
        (inner_w, inner_h),
        t.matte,
        fit_mode=params.fit_mode,
    )

    # frame bands
    frame_img = Image.new("RGBA", (panel_w, panel_h), COLOR_BLACK)
    draw = ImageDraw.Draw(frame_img)
    c1, c2, c3 = _tone_colors(params.frame_tone)

    def rect(x0: int, y0: int, x1: int, y1: int, color: _RGB) -> None:
        draw.rectangle([x0, y0, x1, y1], outline=color, width=1, fill=color)

    rect(0, 0, panel_w - 1, panel_h - 1, c1)
    rect(t.outer, t.outer, panel_w - t.outer - 1, panel_h - t.outer - 1, c2)
    rect(
        t.outer + t.inner,
        t.outer + t.inner,
        panel_w - t.outer - t.inner - 1,
        panel_h - t.outer - t.inner - 1,
        c3,
    )

    # bevel highlight and shadow
    if bevel > 0:
        hl = Image.new("RGBA", (panel_w, panel_h), (*COLOR_WHITE, 0))
        hl_draw = ImageDraw.Draw(hl)
        inset = t.outer + t.inner
        for i in range(bevel):
            alpha = int(_BEVEL_ALPHA_MAX * (1 - i / max(1, bevel)))
            hl_draw.rectangle(
                [inset + i, inset + i, panel_w - inset - 1 - i, inset + i],
                fill=(*COLOR_WHITE, alpha),
            )
        for i in range(bevel):
            alpha = int(_BEVEL_ALPHA_MAX * (1 - i / max(1, bevel)))
            hl_draw.rectangle(
                [inset + i, inset + i, inset + i, panel_h - inset - 1 - i],
                fill=(*COLOR_WHITE, alpha),
            )
            hl_draw.rectangle(
                [
                    inset + i,
                    panel_h - inset - 1 - i,
                    panel_w - inset - 1 - i,
                    panel_h - inset - 1 - i,
                ],
                fill=(*COLOR_BLACK, alpha // 2),
            )
            hl_draw.rectangle(
                [
                    panel_w - inset - 1 - i,
                    inset + i,
                    panel_w - inset - 1 - i,
                    panel_h - inset - 1 - i,
                ],
                fill=(*COLOR_BLACK, alpha // 2),
            )
        frame_img = Image.alpha_composite(frame_img, hl)

    # apply texture to frame bands (Change 8 clamps strength in caller)
    frame_img = _add_frame_texture(frame_img, params.frame_texture_strength)

    # paste matte plus image
    matte_xy = (t.outer + t.inner, t.outer + t.inner)
    frame_img.paste(matte_img, matte_xy)

    # drop shadow: compose once under the frame to reduce blends
    shadow_box = Image.new(
        "RGBA", (panel_w, panel_h), (*COLOR_BLACK, _SHADOW_ALPHA),
    )
    shadow_box = shadow_box.filter(
        ImageFilter.GaussianBlur(radius=params.shadow_radius),
    )
    sx, sy = params.shadow_offset
    base.alpha_composite(shadow_box, dest=(sx, sy))
    base = Image.alpha_composite(base, frame_img)

    label_anchor = (panel_w // 2, panel_h)
    return base.convert("RGB"), label_anchor


@lru_cache(maxsize=8)
def _get_font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a font at the given pixel size with fallback; cached."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", px)
    except OSError:
        return ImageFont.load_default()


def _try_font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to get a readable font, fallback to default (cached)."""
    return _get_font(px)


def draw_label(  # noqa: PLR0913
    canvas: Image.Image,
    center: tuple[int, int],
    text: str,
    px: int,
    fill: _RGB,
    *,
    y_offset: int = 0,
) -> None:
    """Draw a small centered label at a given canvas point."""
    draw = ImageDraw.Draw(canvas)
    font = _try_font(px)
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    x = center[0] - (w // 2)
    y = center[1] + y_offset
    draw.text((x + 1, y + 1), text, font=font, fill=COLOR_BLACK)
    draw.text((x, y), text, font=font, fill=fill)


def make_wall_canvas(
    size: tuple[int, int],
    color: _RGB,
    *,
    vignette: bool = True,
    noise: bool = False,
) -> Image.Image:
    """Build a wall background with optional vignette and noise."""
    w, h = size
    wall = Image.new("RGB", (w, h), color)

    # Soft vertical gradient for wall lighting
    grad = Image.new("L", (1, h))
    for y in range(h):
        v = int(
            _WALL_GRADIENT_CENTER
            - _WALL_GRADIENT_RANGE * (abs((y - h / 2) / (h / 2))),
        )
        grad.putpixel((0, y), max(0, min(255, v)))
    grad = grad.resize((w, h))
    wall = Image.composite(wall, Image.new("RGB", (w, h), COLOR_BLACK), grad)

    if vignette:
        overlay = Image.new("L", (w, h), 0)
        ov_draw = ImageDraw.Draw(overlay)
        margin = int(min(w, h) * _VIGNETTE_MARGIN_FRAC)
        ov_draw.rectangle([margin, margin, w - margin, h - margin], fill=255)
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=margin // 2))
        wall = Image.composite(
            wall, Image.new("RGB", (w, h), COLOR_BLACK), overlay,
        )

    if noise:
        noise_small = _gaussian_noise_image(
            (max(1, w // 4), max(1, h // 4)),
            sigma=_NOISE_EFFECT_SCALE,
            rng=stv_random.get_numpy_rng(),
        )
        noise_big = noise_small.resize(
            (w, h),
            Image.Resampling.BILINEAR,
        ).filter(ImageFilter.GaussianBlur(radius=_NOISE_GAUSS_RADIUS))
        texture = ImageOps.colorize(noise_big, (0, 0, 0), color)
        wall = Image.blend(wall, texture, 0.05)

    return wall


def _gaussian_noise_image(
    size: tuple[int, int],
    *,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    """Return a grayscale noise image generated via the shared RNG."""
    width, height = size
    eff_width = max(1, int(width))
    eff_height = max(1, int(height))
    generator = rng or stv_random.get_numpy_rng()
    samples = generator.normal(
        loc=127.5,
        scale=float(sigma),
        size=(eff_height, eff_width),
    )
    clipped = np.clip(np.rint(samples), 0, 255).astype(np.uint8)
    return Image.fromarray(clipped, mode="L")


def _add_frame_texture(
    frame_img: Image.Image,
    strength: int = 18,
) -> Image.Image:
    """Overlay a faint texture to reduce flatness of frame bands."""
    if strength <= 0:
        return frame_img

    alpha = frame_img.getchannel("A") if frame_img.mode == "RGBA" else None
    base_rgb = frame_img.convert("RGB")

    w, h = base_rgb.size
    streaks = _gaussian_noise_image(
        (max(1, w // 3), 1),
        sigma=25.0,
        rng=stv_random.get_numpy_rng(),
    ).resize(
        (w, h),
        Image.Resampling.BILINEAR,
    )
    streaks = streaks.filter(ImageFilter.GaussianBlur(radius=1))
    streaks_rgb = ImageOps.colorize(streaks, COLOR_BLACK, COLOR_WHITE)

    blend_amount = min(_BLEND_MAX, max(0.0, strength) / 100.0)
    blended_rgb = Image.blend(base_rgb, streaks_rgb, blend_amount)

    if alpha is not None:
        blended_rgba = blended_rgb.convert("RGBA")
        blended_rgba.putalpha(alpha)
        return blended_rgba
    return blended_rgb
