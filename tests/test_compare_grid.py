"""
Unit tests for tools/compare_grid.py.

These tests validate argument-parsing helpers, error handling, and the
top-level `main()` behavior for both grid and gallery modes without
invoking the heavy image pipeline. The module is imported directly from
the tools directory to keep the project layout unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import runpy
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

# Pair length constant to avoid magic numbers in assertions.
PAIR_LEN = 2


class CompareGridModule(Protocol):
    """Protocol describing the compare_grid functions used by the tests."""

    def positive_int(self, s: str) -> int: ...
    def size_2d(self, s: str) -> tuple[int, int]: ...
    def _parse_hex_color(self, s: str) -> tuple[int, int, int]: ...
    def build_parser(self) -> argparse.ArgumentParser: ...
    def main(self) -> int: ...


def _import_compare_grid() -> CompareGridModule:
    """
    Import tools/compare_grid.py as a standalone module object.

    Returns
    -------
    CompareGridModule
        The imported module instance with functions under test.

    """
    root = Path(__file__).parent.parent
    path = root / "tools" / "compare_grid.py"

    spec = importlib.util.spec_from_file_location("tools_compare_grid", path)
    assert spec is not None
    assert spec.loader is not None

    mod = importlib.util.module_from_spec(spec)

    # Ensure project src is on sys.path for style_transfer_visualizer imports.
    src_path = str(root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    spec.loader.exec_module(mod)  # type: ignore[assignment]
    return cast("CompareGridModule", mod)


class SaveGridKwargs(TypedDict):
    """Keyword arguments recorded for save_comparison_grid stubs."""

    content_path: Path
    style_path: Path
    result_path: Path
    out_path: Path
    target_height: int | None
    target_size: tuple[int, int] | None
    pad: int
    border_px: int


class SaveGalleryKwargs(TypedDict):
    """Keyword arguments recorded for save_gallery_comparison stubs."""

    content_path: Path
    style_path: Path
    result_path: Path | None
    out_path: Path
    target_size: tuple[int, int] | None
    layout: str
    wall_color: tuple[int, int, int]
    frame_tone: str
    show_labels: bool


def _make_fake_save_gallery(
    calls: list[SaveGalleryKwargs],
) -> Callable[..., Path]:
    """
    Create a stub for save_gallery_comparison that records calls.

    Parameters
    ----------
    calls
        A list that will collect the keyword arguments from each call.

    Returns
    -------
    Callable[..., Path]
        A function that mirrors save_gallery_comparison and returns `out_path`.

    """
    def fake_save_gallery_comparison(  # noqa: PLR0913
        *,
        content_path: Path,
        style_path: Path,
        result_path: Path | None,
        out_path: Path,
        target_size: tuple[int, int] | None = None,
        layout: str = "gallery-stacked-left",
        wall_color: tuple[int, int, int] = (0, 0, 0),
        frame_tone: str = "gold",
        show_labels: bool = True,
    ) -> Path:
        calls.append(
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

    return fake_save_gallery_comparison


# -------------
# Parser tests
# -------------

def test_positive_int_valid_and_errors() -> None:
    """positive_int enforces integer type and strict positivity."""
    cg = _import_compare_grid()
    assert cg.positive_int("7") == 7  # noqa: PLR2004

    with pytest.raises(
        argparse.ArgumentTypeError,
        match="must be an integer",
    ):
        cg.positive_int("x")

    with pytest.raises(
        argparse.ArgumentTypeError,
        match="must be positive",
    ):
        cg.positive_int("0")


def test_size_2d_valid_and_errors() -> None:
    """size_2d parses WxH and validates both parts are positive integers."""
    cg = _import_compare_grid()
    assert cg.size_2d("1920x1080") == (1920, 1080)

    with pytest.raises(
        argparse.ArgumentTypeError,
        match="look like WxH",
    ):
        cg.size_2d("1920")

    with pytest.raises(
        argparse.ArgumentTypeError,
        match="must be integers",
    ):
        cg.size_2d("ax1080")

    with pytest.raises(
        argparse.ArgumentTypeError,
        match="must be positive",
    ):
        cg.size_2d("1920x0")


def test_parse_hex_color_valid_and_errors() -> None:
    """_parse_hex_color parses #rrggbb and rejects malformed values."""
    cg = _import_compare_grid()
    assert cg._parse_hex_color("#3c434a") == (60, 67, 74)  # noqa: SLF001

    with pytest.raises(
        argparse.ArgumentTypeError,
        match="look like #rrggbb",
    ):
        cg._parse_hex_color("#12345")  # noqa: SLF001

    with pytest.raises(
        argparse.ArgumentTypeError,
        match="invalid hex",
    ):
        cg._parse_hex_color("#zzzzzz")  # noqa: SLF001


def test_build_parser_defaults() -> None:
    """build_parser provides expected defaults for optional flags."""
    cg = _import_compare_grid()
    parser = cg.build_parser()
    args = parser.parse_args([
        "--content", "c.jpg",
        "--style", "s.jpg",
    ])

    assert args.result is None
    assert args.out is None
    assert args.layout is None
    assert args.target_height == 512  # noqa: PLR2004
    assert args.pad == 16  # noqa: PLR2004
    assert args.border_px == 0


# ------------
# main() tests
# ------------

def test_main_grid_mode_requires_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no layout is supplied, --result is required for grid mode."""
    cg = _import_compare_grid()

    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--content", "c.jpg",
        "--style", "s.jpg",
    ])

    with pytest.raises(SystemExit, match="result is required"):
        cg.main()


def test_main_grid_mode_invokes_save_with_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Grid mode calls save_comparison_grid with parsed and defaulted args."""
    cg = _import_compare_grid()

    calls: list[SaveGridKwargs] = []

    def fake_default_name(
        _content: Path,
        _style: Path,
        _out_dir: Path,
    ) -> Path:
        # Return without suffix to ensure main() appends .png
        return tmp_path / "outname"

    def fake_save_grid(  # noqa: PLR0913
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
        calls.append({
            "content_path": content_path,
            "style_path": style_path,
            "result_path": result_path,
            "out_path": out_path,
            "target_height": target_height,
            "target_size": target_size,
            "pad": pad,
            "border_px": border_px,
        })
        return out_path

    monkeypatch.setattr(cg, "default_comparison_name", fake_default_name)
    monkeypatch.setattr(cg, "save_comparison_grid", fake_save_grid)

    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--content", "c.jpg",
        "--style", "s.jpg",
        "--result", "r.jpg",
        "--pad", "8",
        "--border-px", "2",
        "--target-height", "256",
    ])

    rc = cg.main()
    assert rc == 0
    assert len(calls) == 1

    saved = calls[0]
    assert saved["out_path"].suffix == ".png"
    assert saved["target_height"] == 256  # noqa: PLR2004
    assert saved["target_size"] is None
    assert saved["pad"] == 8  # noqa: PLR2004
    assert saved["border_px"] == 2  # noqa: PLR2004


def test_main_gallery_two_across_ignores_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Gallery two-across ignores a provided result and honors wall and size.

    Verifies that:
    - layout selection is `gallery-two-across`
    - result_path becomes None even if passed via CLI
    - wall color hex is parsed to an RGB tuple
    - target size is parsed from WxH
    """
    cg = _import_compare_grid()

    calls: list[SaveGalleryKwargs] = []

    def fake_default_name(
        _content: Path,
        _style: Path,
        _out_dir: Path,
    ) -> Path:
        return tmp_path / "gallery_out"

    monkeypatch.setattr(cg, "default_comparison_name", fake_default_name)

    fake_gallery = _make_fake_save_gallery(calls)
    monkeypatch.setattr(cg, "save_gallery_comparison", fake_gallery)

    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--content", "c.jpg",
        "--style", "s.jpg",
        "--result", "unused.jpg",
        "--layout", "gallery-two-across",
        "--wall", "#112233",
        "--target-size", "320x200",
    ])

    rc = cg.main()
    assert rc == 0
    assert len(calls) == 1

    saved = calls[0]
    assert saved["result_path"] is None
    assert saved["layout"] == "gallery-two-across"
    assert saved["wall_color"] == (17, 34, 51)
    assert saved["target_size"] == (320, 200)


def test_main_gallery_stacked_left_uses_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Gallery stacked-left forwards the result path and frame tone.

    Also verifies that when no target-size is supplied, the module uses a
    built-in default canvas size (tuple of two ints).
    """
    cg = _import_compare_grid()

    calls: list[SaveGalleryKwargs] = []

    def fake_default_name(
        _content: Path,
        _style: Path,
        _out_dir: Path,
    ) -> Path:
        # Return with suffix to confirm main() preserves extensions
        return tmp_path / "gallery_out2.png"

    monkeypatch.setattr(cg, "default_comparison_name", fake_default_name)

    fake_gallery = _make_fake_save_gallery(calls)
    monkeypatch.setattr(cg, "save_gallery_comparison", fake_gallery)

    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--content", "c.jpg",
        "--style", "s.jpg",
        "--result", "r.jpg",
        "--layout", "gallery-stacked-left",
        "--frame-style", "black",
        "--show-labels",
    ])

    rc = cg.main()
    assert rc == 0
    assert len(calls) == 1

    saved = calls[0]
    assert saved["result_path"] == Path("r.jpg")
    assert saved["frame_tone"] == "black"
    assert isinstance(saved["target_size"], tuple)
    assert len(saved["target_size"]) == PAIR_LEN


def test_main_respects_explicit_out_and_suffix_variants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    When --out is given, default_comparison_name is not used.

    The suffix is normalized to .png if needed.
    """
    cg = _import_compare_grid()

    calls: list[SaveGalleryKwargs] = []
    fake_gallery = _make_fake_save_gallery(calls)
    monkeypatch.setattr(cg, "save_gallery_comparison", fake_gallery)

    # Guard: default name should not be consulted when --out is provided.
    def _should_not_call(*_a: object, **_k: object) -> Path:
        msg = "default_comparison_name should not be called"
        raise AssertionError(msg)

    monkeypatch.setattr(cg, "default_comparison_name", _should_not_call)

    # Case A: explicit .png is preserved
    out_png = tmp_path / "explicit.png"
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--content", "c.jpg", "--style", "s.jpg",
         "--layout", "gallery-two-across", "--out", str(out_png)],
    )
    rc = cg.main()
    assert rc == 0
    assert calls[-1]["out_path"].suffix == ".png"

    # Case B: non-png suffix is converted to .png
    out_jpg = tmp_path / "explicit.jpg"
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--content", "c.jpg", "--style", "s.jpg",
         "--layout", "gallery-two-across", "--out", str(out_jpg)],
    )
    rc = cg.main()
    assert rc == 0
    assert calls[-1]["out_path"].suffix == ".png"



def test_script_entry_guard_executes_with_stubbed_image_grid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Execute tools/compare_grid.py as __main__ to cover the entry guard.

    The real style_transfer_visualizer.image_grid is stubbed so the script
    can run without heavy image I/O.
    """
    # Ensure src is importable for constants/logging imports.
    root = Path(__file__).parent.parent
    src_path = str(root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    class FakeImageGrid(types.ModuleType):
        """Minimal image_grid replacement used for entry-guard coverage."""

        def __init__(self, name: str, base: Path) -> None:
            super().__init__(name)
            self.base = base

        def default_comparison_name(
            self,
            _content: Path,
            _style: Path,
            _out_dir: Path,
        ) -> Path:
            return self.base / "from_default.png"

        @staticmethod
        def save_gallery_comparison(  # noqa: PLR0913
            *,
            content_path: Path,
            style_path: Path,
            result_path: Path | None,
            out_path: Path,
            target_size: tuple[int, int] | None = None,
            layout: str = "gallery-two-across",
            wall_color: tuple[int, int, int] = (0, 0, 0),
            frame_tone: str = "gold",
            show_labels: bool = True,
        ) -> Path:
            # Reference args to avoid ARG001 warnings.
            _ = (
                content_path, style_path, result_path, target_size,
                layout, wall_color, frame_tone, show_labels,
            )
            # Return the path; no disk I/O here.
            return out_path

        @staticmethod
        def save_comparison_grid(  # noqa: PLR0913
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
            _ = (
                content_path, style_path, result_path, target_height,
                target_size, pad, border_px,
            )
            return out_path

    fake_img_grid = FakeImageGrid(
        "style_transfer_visualizer.image_grid",
        tmp_path,
    )

    # Inject stub module before the script import path is executed.
    monkeypatch.setitem(
        sys.modules,
        "style_transfer_visualizer.image_grid",
        fake_img_grid,
    )

    # Provide explicit --out so default naming is not used.
    explicit = tmp_path / "explicit.png"
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--content", "c.jpg", "--style", "s.jpg",
         "--layout", "gallery-two-across", "--out", str(explicit)],
    )

    # Run the file as a script to trigger the __main__ guard.
    script_path = str(root / "tools" / "compare_grid.py")
    with pytest.raises(SystemExit) as ei:
        runpy.run_path(script_path, run_name="__main__")
    assert ei.value.code == 0
