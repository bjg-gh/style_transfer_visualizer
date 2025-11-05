"""Tests for the reusable comparison gallery API and CLI."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import style_transfer_visualizer.gallery.api as gallery_api
import style_transfer_visualizer.gallery.cli as gallery_cli
from style_transfer_visualizer import gallery
from style_transfer_visualizer.constants import RESOLUTION_FULL_HD
from style_transfer_visualizer.gallery import ComparisonRenderOptions
from style_transfer_visualizer.gallery.cli import build_parser, main

pytestmark = pytest.mark.visual

GRID_TARGET_HEIGHT = 256
GRID_PAD = 8
GRID_BORDER = 2
ARGPARSE_USAGE_CODE = 2
CLI_PAD_VALUE = 32
CUSTOM_OUT_NAME = "custom-output.jpg"


def test_positive_int_valid_and_errors() -> None:
    """``positive_int`` accepts positive values and rejects others."""
    assert gallery.positive_int("3") == 3  # noqa: PLR2004
    with pytest.raises(ValueError, match="must be an integer"):
        gallery.positive_int("nope")
    with pytest.raises(ValueError, match="must be positive"):
        gallery.positive_int("0")


def test_size_2d_valid_and_errors() -> None:
    """``size_2d`` parses width/height strings and validates them."""
    assert gallery.size_2d("100x200") == (100, 200)
    with pytest.raises(ValueError, match="must look like WxH"):
        gallery.size_2d("100")
    with pytest.raises(ValueError, match="width and height must be integers"):
        gallery.size_2d("1xnope")
    with pytest.raises(ValueError, match="width and height must be positive"):
        gallery.size_2d("0x10")


def test_parse_wall_color_valid_and_errors() -> None:
    """Hex wall colors are converted to RGB tuples."""
    assert gallery.parse_wall_color("#0a0b0c") == (10, 11, 12)
    with pytest.raises(ValueError, match="must look like #rrggbb"):
        gallery.parse_wall_color("#fff")
    with pytest.raises(ValueError, match="wall color contains invalid hex digits"):
        gallery.parse_wall_color("#xx0000")


def test_render_comparison_requires_result_for_grid(tmp_path: Path) -> None:
    """Grid rendering without a result image raises a ValueError."""
    options = ComparisonRenderOptions(
        content_path=tmp_path / "c.jpg",
        style_path=tmp_path / "s.jpg",
        layout=None,
    )
    with pytest.raises(ValueError, match="result_path is required"):
        gallery.render_comparison(options)


def test_render_comparison_grid_passes_expected_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Grid rendering delegates to ``save_comparison_grid`` with options."""
    called: dict[str, Any] = {}

    def fake_save_comparison_grid(  # noqa: PLR0913
        *,
        content_path: Path,
        style_path: Path,
        result_path: Path,
        out_path: Path,
        target_height: int | None,
        target_size: tuple[int, int] | None,
        pad: int,
        border_px: int,
    ) -> Path:
        called.update(
            {
                "content_path": content_path,
                "style_path": style_path,
                "result_path": result_path,
                "out_path": out_path,
                "target_height": target_height,
                "target_size": target_size,
                "pad": pad,
                "border_px": border_px,
            },
        )
        return out_path

    monkeypatch.setattr(
        gallery_api,
        "save_comparison_grid",
        fake_save_comparison_grid,
    )

    options = ComparisonRenderOptions(
        content_path=tmp_path / "c.jpg",
        style_path=tmp_path / "s.jpg",
        result_path=tmp_path / "r.jpg",
        target_height=GRID_TARGET_HEIGHT,
        pad=GRID_PAD,
        border_px=GRID_BORDER,
        layout=None,
    )

    saved = gallery.render_comparison(options)
    assert saved == Path("comparison_c_x_s.png")
    assert called["target_height"] == GRID_TARGET_HEIGHT
    assert called["target_size"] is None
    assert called["pad"] == GRID_PAD
    assert called["border_px"] == GRID_BORDER


def test_render_comparison_gallery_two_across_ignores_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Two-across layout drops any provided result path."""
    recorded: dict[str, Any] = {}

    def fake_save_gallery_comparison(  # noqa: PLR0913
        *,
        content_path: Path,
        style_path: Path,
        result_path: Path | None,
        out_path: Path,
        target_size: tuple[int, int] | None,
        layout: str,
        wall_color: tuple[int, int, int],
        frame_tone: str,
        show_labels: bool,
    ) -> Path:
        recorded.update(
            {
                "content_path": content_path,
                "style_path": style_path,
                "result_path": result_path,
                "out_path": out_path,
                "target_size": target_size,
                "layout": layout,
                "wall_color": wall_color,
                "frame_tone": frame_tone,
                "show_labels": show_labels,
            },
        )
        return out_path

    monkeypatch.setattr(
        gallery_api,
        "save_gallery_comparison",
        fake_save_gallery_comparison,
    )

    options = ComparisonRenderOptions(
        content_path=tmp_path / "c.jpg",
        style_path=tmp_path / "s.jpg",
        result_path=tmp_path / "should-ignore.png",
        layout="gallery-two-across",
        target_size=(1280, 720),
        show_labels=True,
    )

    saved = gallery.render_comparison(options)
    assert saved == Path("comparison_c_x_s.png")
    assert recorded["layout"] == "gallery-two-across"
    assert recorded["result_path"] is None
    assert recorded["target_size"] == (1280, 720)
    assert recorded["show_labels"] is True


def test_render_comparison_coerces_string_out_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Explicit string outputs are converted to Paths and normalized to PNG."""
    recorded: dict[str, Any] = {}

    def fake_save_comparison_grid(  # noqa: PLR0913
        *,
        content_path: Path,
        style_path: Path,
        result_path: Path,
        out_path: Path,
        target_height: int | None,
        target_size: tuple[int, int] | None,
        pad: int,
        border_px: int,
    ) -> Path:
        recorded.update(
            {
                "content_path": content_path,
                "style_path": style_path,
                "result_path": result_path,
                "out_path": out_path,
                "target_height": target_height,
                "target_size": target_size,
                "pad": pad,
                "border_px": border_px,
            },
        )
        return out_path

    monkeypatch.setattr(
        gallery_api,
        "save_comparison_grid",
        fake_save_comparison_grid,
    )

    out = gallery.render_comparison(
        ComparisonRenderOptions(
            content_path=tmp_path / "content.jpg",
            style_path=tmp_path / "style.jpg",
            result_path=tmp_path / "result.jpg",
            out_path=Path(CUSTOM_OUT_NAME),
            layout=None,
        ),
    )

    assert isinstance(recorded["out_path"], Path)
    assert recorded["out_path"].name == "custom-output.png"
    assert out == recorded["out_path"]


def test_render_comparison_gallery_stacked_forwards_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Stacked gallery layout forwards the result path."""
    recorded: dict[str, Any] = {}

    def fake_save_gallery_comparison(  # noqa: PLR0913
        *,
        content_path: Path,
        style_path: Path,
        result_path: Path | None,
        out_path: Path,
        target_size: tuple[int, int] | None,
        layout: str,
        wall_color: tuple[int, int, int],
        frame_tone: str,
        show_labels: bool,
    ) -> Path:
        recorded.update(
            {
                "content_path": content_path,
                "style_path": style_path,
                "result_path": result_path,
                "out_path": out_path,
                "target_size": target_size,
                "layout": layout,
                "wall_color": wall_color,
                "frame_tone": frame_tone,
                "show_labels": show_labels,
            },
        )
        return out_path

    monkeypatch.setattr(
        gallery_api,
        "save_gallery_comparison",
        fake_save_gallery_comparison,
    )

    result = tmp_path / "result.png"
    options = ComparisonRenderOptions(
        content_path=tmp_path / "c.jpg",
        style_path=tmp_path / "s.jpg",
        result_path=result,
        layout="gallery-stacked-left",
        target_size=None,
    )

    saved = gallery.render_comparison(options)
    assert saved == Path("comparison_c_x_s.png")
    assert recorded["layout"] == "gallery-stacked-left"
    assert recorded["result_path"] == result
    assert recorded["target_size"] == RESOLUTION_FULL_HD


def test_cli_main_requires_result_when_layout_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI exits with an error when a result path is not provided."""
    monkeypatch.setattr(sys, "argv", ["prog", "--content", "a.jpg", "--style", "b.jpg"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == ARGPARSE_USAGE_CODE


def test_cli_main_passes_options_to_render(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI builds ComparisonRenderOptions and calls render_comparison."""
    captured: dict[str, ComparisonRenderOptions] = {}

    def fake_render(options: ComparisonRenderOptions) -> Path:
        captured["options"] = options
        return Path("out.png")

    monkeypatch.setattr(gallery, "render_comparison", fake_render)
    monkeypatch.setattr(gallery_cli, "render_comparison", fake_render)
    argv = [
        "--content",
        "content.jpg",
        "--style",
        "style.jpg",
        "--result",
        "result.jpg",
        "--layout",
        "gallery-stacked-left",
        "--pad",
        str(CLI_PAD_VALUE),
        "--target-size",
        "1920x1080",
        "--show-labels",
    ]
    assert main(argv) == 0
    options = captured["options"]
    assert options.show_labels is True
    assert options.pad == CLI_PAD_VALUE
    assert options.layout == "gallery-stacked-left"
    assert options.target_size == (1920, 1080)


def test_cli_main_converts_render_errors_to_system_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ValueError from render_comparison is surfaced via parser.error."""

    def boom(_options: ComparisonRenderOptions) -> Path:
        raise ValueError("bad size")

    monkeypatch.setattr(gallery, "render_comparison", boom)
    monkeypatch.setattr(gallery_cli, "render_comparison", boom)

    argv = [
        "--content",
        "content.jpg",
        "--style",
        "style.jpg",
        "--layout",
        "gallery-two-across",
    ]
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == ARGPARSE_USAGE_CODE


def test_build_parser_validates_arguments() -> None:
    """Parser wires validators that surface friendly errors."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--content", "c.jpg", "--style", "s.jpg", "--pad", "-1"])


@pytest.mark.integration
def test_tools_compare_grid_wrapper_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    """tools/compare_grid.py delegates to the shared CLI main."""
    root = Path(__file__).parent.parent
    script = root / "tools" / "compare_grid.py"

    module = ModuleType("fake_cli")
    called: dict[str, Any] = {}

    def fake_main(argv: list[str] | None = None) -> int:
        called["ran"] = True
        return 0

    def fake_build_parser() -> None:
        pytest.fail("build_parser should not be used by wrapper")

    module.main = fake_main  # type: ignore[attr-defined]
    module.build_parser = fake_build_parser  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "style_transfer_visualizer.gallery.cli", module)

    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(script), run_name="__main__")

    assert exc.value.code == 0
    assert called["ran"] is True
