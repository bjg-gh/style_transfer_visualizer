# tests/test_image_grid.py
"""
Tests for image_grid.py covering grid and gallery utilities.

The suite focuses on public entry points and adds narrow calls to
private helpers only where needed to reach full coverage.
"""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING, cast

import pytest
from PIL import Image, ImageFont

import style_transfer_visualizer.image_grid as ig

if TYPE_CHECKING:  # typing-only imports
    import os
    from pathlib import Path


# Constants to avoid magic numbers in assertions
PAD_SMALL = 2
PAD_MED = 4
BORDER_PX = 1
CANVAS_W = 120
CANVAS_H = 60
BOX_W = 100
BOX_H = 60
IMG_W = 30
IMG_H = 20
MAX_CW = 100
MAX_CH = 50
MAX_EDGE = 60


# --------------------------
# Small helpers for the tests
# --------------------------

class _DummyImg:
    """Duck-typed minimal image used for edge-case size testing."""

    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size

    @staticmethod
    def resize(
        new_size: tuple[int, int],
        *_: object,
        **__: object,
    ) -> _DummyImg:
        return _DummyImg(new_size)


def _mk_rgb(w: int, h: int, color: str = "white") -> Image.Image:
    """Make a solid RGB image of arbitrary size."""
    return Image.new("RGB", (w, h), color=color)


def _mk_rgba(w: int, h: int) -> Image.Image:
    """Make an RGBA image with a transparent corner."""
    img = Image.new("RGBA", (w, h), (255, 0, 0, 0))
    img.putpixel((0, 0), (0, 255, 0, 255))
    return img


# -------------
# Format helpers
# -------------

def test_to_rgb_conversions(sample_image: Image.Image) -> None:
    """RGB stays RGB; RGBA alpha-composites; LA converts."""
    # Hit the final else-path: non-RGB, non-alpha (mode "L")
    l_img = Image.new("L", (4, 4), 128)
    l_rgb = ig._to_rgb(l_img, bg_color=ig.COLOR_WHITE)  # noqa: SLF001
    assert l_rgb.mode == "RGB"

    # Existing coverage
    rgb_else = ig._to_rgb(_mk_rgb(5, 5), bg_color=ig.COLOR_WHITE)  # noqa: SLF001
    assert rgb_else.mode == "RGB"

    rgb = ig._to_rgb(sample_image, bg_color=ig.COLOR_WHITE)  # noqa: SLF001
    assert rgb.mode == "RGB"

    rgba = ig._to_rgb(_mk_rgba(8, 8), bg_color=ig.COLOR_WHITE)  # noqa: SLF001
    assert rgba.mode == "RGB"
    assert isinstance(rgba.getpixel((0, 0)), tuple)

    la = Image.new("LA", (4, 4), (128, 255))
    la_rgb = ig._to_rgb(la, bg_color=ig.COLOR_WHITE)  # noqa: SLF001
    assert la_rgb.mode == "RGB"


def test_resize_to_height_and_error() -> None:
    """Resize keeps aspect; zero height raises ValueError."""
    im = _mk_rgb(10, 20)
    out = ig._resize_to_height(im, 40)  # noqa: SLF001
    assert out.size == (20, 40)

    with pytest.raises(ValueError, match="zero height"):
        ig._resize_to_height(  # noqa: SLF001
            _DummyImg((10, 0)),  # type: ignore[arg-type]
            10,
        )


def test_draw_border_and_dimensions(sample_image: Image.Image) -> None:
    """Border expands image; dimension helpers compute correctly."""
    with_border = ig._draw_border(sample_image, BORDER_PX)  # noqa: SLF001
    assert with_border.size == (
        sample_image.size[0] + 2 * BORDER_PX,
        sample_image.size[1] + 2 * BORDER_PX,
    )

    w, h, ws, hs = ig._content_dimensions(  # noqa: SLF001
        [_mk_rgb(5, 7), _mk_rgb(3, 4)],
        pad=1,
    )
    assert (w, h) == (5 + 1 + 3, 7)
    assert ws == [5, 3]
    assert hs == [7, 4]


def test_scale_images_to_target_and_fit() -> None:
    """Height scaling and canvas-fit downscale behave as documented."""
    imgs = [_mk_rgb(50, 100), _mk_rgb(20, 100)]
    scaled = ig._scale_images_to_target(imgs, 50, None)  # noqa: SLF001
    assert [im.size for im in scaled] == [(25, 50), (10, 50)]

    # When target_size is set and target_height None, images are not touched
    untouched = ig._scale_images_to_target(  # noqa: SLF001
        imgs,
        None,
        (100, 50),
    )
    assert untouched is imgs

    # Fit two 60x60 images with pad into a smaller canvas
    ims = [_mk_rgb(60, 60), _mk_rgb(60, 60)]
    tight_w, tight_h = ig._content_dimensions(  # noqa: SLF001
        ims,
        PAD_SMALL,
    )[:2]
    fitted, cw, ch = ig._scale_images_to_fit_canvas(  # noqa: SLF001
        ims,
        PAD_SMALL,
        tight_w,
        tight_h,
        (MAX_CW, MAX_CH),
    )
    assert cw <= MAX_CW
    assert ch <= MAX_CH
    assert all(im.size[0] <= MAX_EDGE for im in fitted)


def test_paste_horizontally_no_error() -> None:
    """Pasting positions images without exception."""
    canvas = _mk_rgb(100, 40, "gray")
    ims = [_mk_rgb(10, 20, "red"), _mk_rgb(10, 30, "blue")]
    ig._paste_horizontally(canvas, ims, 5, (10, 5), 30)  # noqa: SLF001


# -------------
# Public grid IO
# -------------

def test_make_horizontal_grid_variants() -> None:
    """Tight and fixed-size grid modes work and center content."""
    ims = [_mk_rgb(30, 20), _mk_rgb(10, 20), _mk_rgb(5, 20)]
    tight = ig.make_horizontal_grid(
        ims,
        target_height=20,
        pad=PAD_SMALL,
        border_px=BORDER_PX,
    )
    assert tight.size[1] == 20 + 2 * BORDER_PX + 2 * PAD_SMALL

    fixed = ig.make_horizontal_grid(
        ims,
        target_height=None,
        target_size=(CANVAS_W, CANVAS_H),
        pad=PAD_MED,
    )
    assert fixed.size == (CANVAS_W, CANVAS_H)

    with pytest.raises(ValueError, match="No images provided"):
        ig.make_horizontal_grid([], target_height=10)


def test_save_comparison_grid(
    tmp_path: Path,
    content_image: Path,
    style_image: Path,
) -> None:
    """Comparison grid saves a PNG and validates types."""
    out = tmp_path / "cmp.png"
    res = tmp_path / "result.png"
    _mk_rgb(64, 64, "purple").save(res)

    # Intentionally wrong type: craft a fake Path typed object
    bad_out: Path = cast("Path", "not-a-path")
    with pytest.raises(TypeError, match=r"out_path must be a pathlib.Path"):
        ig.save_comparison_grid(
            content_image,
            style_image,
            res,
            bad_out,
        )

    path = ig.save_comparison_grid(
        content_image,
        style_image,
        res,
        out,
        target_height=64,
        pad=PAD_MED,
        border_px=BORDER_PX,
    )
    assert path == out
    assert out.is_file()


def test_default_comparison_name_paths(tmp_path: Path) -> None:
    """Default name is deterministic and space-safe."""
    c = tmp_path / "content img.png"
    s = tmp_path / "style pic.png"
    c.write_bytes(b"")
    s.write_bytes(b"")
    name = ig.default_comparison_name(c, s, tmp_path)
    assert name.name == "comparison_content_img_x_style_pic.png"


# --------------------
# Geometry and framing
# --------------------

def test_rect_and_thickness_and_tones() -> None:
    """Rect helpers, frame thickness clamp, and tone colors."""
    r = ig.Rect(1, 2, 6, 7)
    assert r.size() == (5, 5)
    assert r.move_to(0, 0) == ig.Rect(0, 0, 5, 5)
    assert r.inset(1, 1) == ig.Rect(2, 3, 5, 6)

    fp = ig.FrameParams(frame_outer_frac=0.0, frame_inner_frac=0.0)
    t = ig._frame_thickness(20, 10, fp)  # noqa: SLF001
    assert t.outer >= 3  # noqa: PLR2004
    assert t.inner >= 2  # noqa: PLR2004
    assert t.total >= 5  # noqa: PLR2004

    assert ig._tone_colors("oak") != ig._tone_colors("black")  # noqa: SLF001
    assert isinstance(ig._tone_colors("gold"), tuple)  # noqa: SLF001


def test_fit_panel_box_and_inner_aspect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aspect-fitting converges; early-break hits with zero margin."""
    box = ig.Rect(0, 0, BOX_W, BOX_H)
    nx0, ny0, nx1, ny1 = ig._fit_panel_box_to_inner_aspect(  # noqa: SLF001
        (box.x0, box.y0, box.x1, box.y1),
        2.0,
        ig.FrameParams(),
    )
    assert (nx1 - nx0) <= box.w
    assert (ny1 - ny0) <= box.h

    # Force margin to zero so new_pw/new_ph equal the current pw/ph.
    monkeypatch.setattr(
        ig, "_panel_margin_px", lambda *_args, **_kw: 0,
    )
    aspect = BOX_W / BOX_H
    nx0b, ny0b, nx1b, ny1b = ig._fit_panel_box_to_inner_aspect(  # noqa: SLF001
        (box.x0, box.y0, box.x1, box.y1),
        aspect,
        ig.FrameParams(
            matte_frac=0.0,
            frame_outer_frac=0.0,
            frame_inner_frac=0.0,
            bevel_px=0,
            shadow_radius=0,
        ),
    )
    assert (nx1b - nx0b) == BOX_W
    assert (ny1b - ny0b) == BOX_H


def test_fit_panel_box_loop_completes_no_break(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Make the solver never stabilize so the for-loop fully completes."""
    # Oscillate margin: when at full box size, return 1; otherwise 0.
    def oscillating_margin(_params: ig.FrameParams, pw: int, ph: int) -> int:
        return 1 if (pw == BOX_W and ph == BOX_H) else 0

    monkeypatch.setattr(ig, "_panel_margin_px", oscillating_margin)

    box = (0, 0, BOX_W, BOX_H)
    aspect = BOX_W / BOX_H
    nx0, ny0, nx1, ny1 = ig._fit_panel_box_to_inner_aspect(  # noqa: SLF001
        box, aspect, ig.FrameParams(),
    )
    # Valid box within bounds; we mainly needed the "no-break" branch.
    assert 0 <= nx0 < nx1 <= BOX_W
    assert 0 <= ny0 < ny1 <= BOX_H


def test_fit_box_by_inner_aspect_zero_height_raises() -> None:
    """Zero image height triggers the guard clause."""
    with pytest.raises(ValueError, match="Image height must be positive"):
        ig._fit_box_by_inner_aspect(  # noqa: SLF001
            ig.Rect(0, 0, 10, 10),
            _DummyImg((10, 0)),  # type: ignore[arg-type]
            ig.FrameParams(),
            0.1,
        )


def test_fit_and_place_cover_and_contain(sample_image: Image.Image) -> None:
    """Cover crops to fill; contain letterboxes on a matte."""
    out_cover = ig._fit_and_place(  # noqa: SLF001
        sample_image,
        (50, 30),
        PAD_MED,
        fit_mode="cover",
    )
    assert out_cover.size == (50 + PAD_MED * 2, 30 + PAD_MED * 2)

    tall = _mk_rgb(20, 60, "navy")
    out_contain = ig._fit_and_place(  # noqa: SLF001
        tall,
        (50, 30),
        PAD_SMALL,
        fit_mode="contain",
    )
    assert out_contain.size == (50 + PAD_SMALL * 2, 30 + PAD_SMALL * 2)


def test_add_frame_texture_and_wall_canvas() -> None:
    """Texture passthrough when disabled and noisy wall when enabled."""
    # RGBA path (returns with alpha preserved)
    base = Image.new("RGBA", (40, 20), (128, 128, 128, 255))
    no_tex = ig._add_frame_texture(base, strength=0)  # noqa: SLF001
    assert no_tex.size == base.size

    yes_tex = ig._add_frame_texture(base, strength=30)  # noqa: SLF001
    assert yes_tex.size == base.size

    # RGB path (hits the blended_rgb return)
    base_rgb = Image.new("RGB", (40, 20), (200, 200, 200))
    out_rgb = ig._add_frame_texture(base_rgb, strength=15)  # noqa: SLF001
    assert out_rgb.mode == "RGB"

    wall_plain = ig._make_wall_canvas(  # noqa: SLF001
        (80, 40),
        ig.COLOR_GREY,
        vignette=False,
        noise=False,
    )
    assert wall_plain.size == (80, 40)

    wall_decor = ig._make_wall_canvas(  # noqa: SLF001
        (80, 40),
        ig.COLOR_GREY,
        vignette=True,
        noise=True,
    )
    assert wall_decor.size == (80, 40)


def test_font_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force truetype to fail only for DejaVuSans, exercising fallback."""
    ig._get_font.cache_clear()  # type: ignore[attr-defined]  # noqa: SLF001

    original_truetype = ig.ImageFont.truetype

    def fake_truetype(
        fake_font: str
                   | bytes
                   | os.PathLike[str]
                   | os.PathLike[bytes]
                   | typing.BinaryIO,
        size: int,
        index: int = 0,
        encoding: str = "",
        layout_engine: ImageFont.Layout | None = None,
    ) -> ImageFont.FreeTypeFont:
        """Shim that fails for DejaVuSans then defers to real truetype."""
        if fake_font == "DejaVuSans.ttf":
            raise OSError
        return original_truetype(
            fake_font,
            size,
            index=index,
            encoding=encoding,
            layout_engine=layout_engine,
        )

    monkeypatch.setattr(ig.ImageFont, "truetype", fake_truetype)

    font = ig._try_font(14)  # noqa: SLF001
    assert hasattr(font, "getsize") or hasattr(font, "getbbox")


def test_build_framed_panel_and_labels() -> None:
    """Panel rendering returns an RGB panel and a label anchor."""
    img = _mk_rgb(IMG_W, IMG_H, "orange")
    panel, anchor = ig.build_framed_panel(
        img,
        (CANVAS_W, CANVAS_H),
        ig.FrameParams(label="on"),
        wall_color=ig.COLOR_GREY,
    )
    assert panel.mode == "RGB"
    assert panel.size == (CANVAS_W, CANVAS_H)
    assert anchor[0] == panel.size[0] // 2

    # Also cover bevel <= 0 branch
    panel2, _ = ig.build_framed_panel(
        img,
        (CANVAS_W, CANVAS_H),
        ig.FrameParams(bevel_px=0),
        wall_color=ig.COLOR_GREY,
    )
    assert panel2.size == (CANVAS_W, CANVAS_H)

    lab_canvas = _mk_rgb(CANVAS_W, CANVAS_H + 10, "white")
    ig._draw_label(  # noqa: SLF001
        lab_canvas,
        (CANVAS_W // 2, CANVAS_H),
        "Test",
        14,
        ig.COLOR_BLACK,
    )


# -------------------
# Gallery composition
# -------------------

def test_layout_generators() -> None:
    """Two-across and stacked-left layouts cover gap logic."""
    two = ig._layout_two_across(  # noqa: SLF001
        200,
        100,
        lr_margin=10,
        tb_margin=8,
        gap_frac=0.02,
    )
    assert len(two) == 2  # noqa: PLR2004
    assert all(isinstance(r, ig.Rect) for r in two)

    stk = ig._layout_stacked_left(  # noqa: SLF001
        200,
        150,
        lr_margin=12,
        tb_margin=10,
        gap_frac=0.03,
        left_col_frac=0.4,
    )
    assert len(stk) == 3  # noqa: PLR2004
    assert stk[0].x0 == 12  # noqa: PLR2004


def test_render_panels_two_and_three() -> None:
    """Render covers two-image contain and three-image result-contain."""
    canvas = _mk_rgb(300, 200, "gray")
    boxes = ig._layout_two_across(  # noqa: SLF001
        300,
        200,
        lr_margin=10,
        tb_margin=10,
        gap_frac=0.02,
    )
    imgs = [_mk_rgb(60, 40, "red"), _mk_rgb(50, 30, "blue")]
    anchors = ig._render_panels(  # noqa: SLF001
        canvas,
        imgs,
        boxes,
        ig.FrameParams(),
        wall_color=ig.COLOR_GREY,
        two_image=True,
    )
    assert len(anchors) == 2  # noqa: PLR2004

    canvas2 = _mk_rgb(320, 200, "gray")
    boxes2 = ig._layout_stacked_left(  # noqa: SLF001
        320,
        200,
        lr_margin=12,
        tb_margin=12,
        gap_frac=0.02,
        left_col_frac=0.42,
    )
    imgs2 = [
        _mk_rgb(40, 40, "red"),
        _mk_rgb(40, 50, "blue"),
        _mk_rgb(80, 60, "green"),
    ]
    anchors2 = ig._render_panels(  # noqa: SLF001
        canvas2,
        imgs2,
        boxes2,
        ig.FrameParams(),
        wall_color=ig.COLOR_GREY,
        two_image=False,
    )
    assert len(anchors2) == 3  # noqa: PLR2004


def test_make_gallery_comparison_two_and_three() -> None:
    """Gallery supports both two and three image variants and guards."""
    c = _mk_rgb(80, 60, "red")
    s = _mk_rgb(70, 50, "blue")
    r = _mk_rgb(90, 70)

    two = ig.make_gallery_comparison(
        c,
        s,
        None,
        frame=ig.FrameParams(label="on", frame_texture_strength=150),
    )
    assert two.size == ig.RESOLUTION_FULL_HD

    three = ig.make_gallery_comparison(
        c,
        s,
        r,
        frame=ig.FrameParams(label=None, frame_texture_strength=-10),
    )
    assert three.size == ig.RESOLUTION_FULL_HD

    with pytest.raises(ValueError, match="target_size must be positive"):
        ig.make_gallery_comparison(c, s, r, target_size=(0, 1080))


def test_save_gallery_comparison(
    tmp_path: Path,
    content_image: Path,
    style_image: Path,
) -> None:
    """Saving gallery wall writes a PNG and validates out_path type."""
    res = tmp_path / "res.png"
    _mk_rgb(64, 64, "purple").save(res)

    # Fake bad path with correct static type to satisfy the checker
    bad_out: Path = cast("Path", "not-a-path")
    with pytest.raises(TypeError, match=r"out_path must be a pathlib.Path"):
        ig.save_gallery_comparison(
            content_image,
            style_image,
            res,
            bad_out,
        )

    out = tmp_path / "gallery.png"
    p = ig.save_gallery_comparison(
        content_image,
        style_image,
        res,
        out,
        layout="gallery-stacked-left",
        frame_tone="black",
        show_labels=True,
    )
    assert p == out
    assert out.is_file()

    out2 = tmp_path / "gallery2.png"
    p2 = ig.save_gallery_comparison(
        content_image,
        style_image,
        None,
        out2,
        layout="gallery-two-across",
        show_labels=False,
    )
    assert p2 == out2
    assert out2.is_file()


def test_type_checking_block_for_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-import module with TYPE_CHECKING True to execute that block."""
    import importlib.util  # noqa: PLC0415
    import sys  # noqa: PLC0415
    import typing  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location(
        "stv_image_grid_tc",
        ig.__file__,
    )
    assert spec is not None
    assert spec.loader is not None

    tmp_mod = importlib.util.module_from_spec(spec)
    sys.modules["stv_image_grid_tc"] = tmp_mod  # required by dataclasses

    # Flip TYPE_CHECKING only for this exec
    old_tc = typing.TYPE_CHECKING
    monkeypatch.setattr(typing, "TYPE_CHECKING", True, raising=False)
    try:
        spec.loader.exec_module(tmp_mod)  # type: ignore[assignment]
    finally:
        monkeypatch.setattr(typing, "TYPE_CHECKING", old_tc, raising=False)
        sys.modules.pop("stv_image_grid_tc", None)

    assert hasattr(tmp_mod, "Rect")
