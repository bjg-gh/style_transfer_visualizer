"""Tests for runtime comparison helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import pytest
from PIL import Image

from style_transfer_visualizer.gallery import ComparisonRenderOptions
from style_transfer_visualizer.runtime import comparison
from style_transfer_visualizer.runtime.comparison import ComparisonRequest


@pytest.fixture
def sample_images(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create sample content/style/result images for comparison tests."""
    content = tmp_path / "content.jpg"
    style = tmp_path / "style.jpg"
    result = tmp_path / "result.png"
    Image.new("RGB", (32, 24), "red").save(content)
    Image.new("RGB", (16, 16), "blue").save(style)
    Image.new("RGB", (32, 24), "green").save(result)
    return content, style, result


def test_render_comparison_image_inputs_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_images: tuple[Path, Path, Path],
) -> None:
    """Inputs-only comparisons render with the horizontal gallery layout."""
    content, style, _ = sample_images
    recorded: dict[str, ComparisonRenderOptions] = {}

    def fake_render(options: ComparisonRenderOptions) -> Path:
        recorded["options"] = options
        assert options.out_path is not None
        return options.out_path

    monkeypatch.setattr(
        comparison,
        "render_comparison",
        fake_render,
    )

    out = comparison.render_comparison_image(
        content_path=content,
        style_path=style,
        output_dir=tmp_path,
        include_result=False,
    )

    expected = comparison.comparison_output_path(
        tmp_path, content, style, include_result=False,
    )
    assert out == expected
    options = recorded["options"]
    assert options.layout == "gallery-two-across"
    assert options.result_path is None
    assert options.target_size == (32, 24)


def test_render_comparison_image_with_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_images: tuple[Path, Path, Path],
) -> None:
    """Result comparisons use the stacked layout and pass the result path."""
    content, style, result = sample_images
    recorded: dict[str, ComparisonRenderOptions] = {}

    def fake_render(options: ComparisonRenderOptions) -> Path:
        recorded["options"] = options
        assert options.out_path is not None
        return options.out_path

    monkeypatch.setattr(
        comparison,
        "render_comparison",
        fake_render,
    )

    out = comparison.render_comparison_image(
        content_path=content,
        style_path=style,
        output_dir=tmp_path,
        include_result=True,
        result_path=result,
    )

    expected = comparison.comparison_output_path(
        tmp_path, content, style, include_result=True,
    )
    assert out == expected
    options = recorded["options"]
    assert options.layout == "gallery-stacked-left"
    assert options.result_path == result


class RenderCall(TypedDict):
    """Call arguments captured for render_comparison_image."""

    content_path: Path
    style_path: Path
    output_dir: Path
    include_result: bool
    result_path: Path | None


EXPECTED_RENDER_CALLS = 2


def test_render_requested_comparisons_calls_renderer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_images: tuple[Path, Path, Path],
) -> None:
    """Requested comparisons delegate to render_comparison_image."""
    content, style, result = sample_images
    calls: list[RenderCall] = []

    def fake_render_comparison_image(
        content_path: Path,
        style_path: Path,
        *,
        output_dir: Path,
        include_result: bool,
        result_path: Path | None = None,
    ) -> Path:
        calls.append(
            {
                "content_path": content_path,
                "style_path": style_path,
                "output_dir": output_dir,
                "include_result": include_result,
                "result_path": result_path,
            },
        )
        return output_dir / ("result" if include_result else "inputs")

    monkeypatch.setattr(
        comparison,
        "render_comparison_image",
        fake_render_comparison_image,
    )
    monkeypatch.setattr(
        comparison,
        "stylized_image_path_from_paths",
        lambda *_: result,
    )

    paths = comparison.render_requested_comparisons(
        content_path=content,
        style_path=style,
        output_dir=tmp_path,
        request=ComparisonRequest(
            include_inputs=True,
            include_result=True,
        ),
    )

    assert len(calls) == EXPECTED_RENDER_CALLS
    assert calls[0]["include_result"] is False
    assert calls[0]["result_path"] is None
    assert calls[1]["include_result"] is True
    assert calls[1]["result_path"] == result
    assert paths == [
        tmp_path / "inputs",
        tmp_path / "result",
    ]


def test_render_requested_comparisons_uses_supplied_result_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_images: tuple[Path, Path, Path],
) -> None:
    """Explicit result paths should bypass stylized_image_path_from_paths."""
    content, style, _ = sample_images
    provided = tmp_path / "provided.png"
    Image.new("RGB", (10, 10), "purple").save(provided)

    calls: list[RenderCall] = []

    def fake_render_comparison_image(
        content_path: Path,
        style_path: Path,
        *,
        output_dir: Path,
        include_result: bool,
        result_path: Path | None = None,
    ) -> Path:
        calls.append(
            {
                "content_path": content_path,
                "style_path": style_path,
                "output_dir": output_dir,
                "include_result": include_result,
                "result_path": result_path,
            },
        )
        return output_dir / "provided"

    def fail_stylized_image_path_from_paths(*_args: object) -> Path:
        pytest.fail("stylized_image_path_from_paths should not run")

    monkeypatch.setattr(
        comparison,
        "render_comparison_image",
        fake_render_comparison_image,
    )
    monkeypatch.setattr(
        comparison,
        "stylized_image_path_from_paths",
        fail_stylized_image_path_from_paths,
    )

    paths = comparison.render_requested_comparisons(
        content_path=content,
        style_path=style,
        output_dir=tmp_path,
        request=ComparisonRequest(
            include_inputs=False,
            include_result=True,
            result_path=provided,
        ),
    )

    assert len(calls) == 1
    assert calls[0]["result_path"] == provided
    assert paths == [tmp_path / "provided"]


def test_render_requested_comparisons_skips_result_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_images: tuple[Path, Path, Path],
) -> None:
    """Result rendering is omitted when include_result is False."""
    content, style, _ = sample_images
    calls: list[RenderCall] = []

    def fake_render_comparison_image(
        content_path: Path,
        style_path: Path,
        *,
        output_dir: Path,
        include_result: bool,
        result_path: Path | None = None,
    ) -> Path:
        calls.append(
            {
                "content_path": content_path,
                "style_path": style_path,
                "output_dir": output_dir,
                "include_result": include_result,
                "result_path": result_path,
            },
        )
        return output_dir / ("result" if include_result else "inputs")

    monkeypatch.setattr(
        comparison,
        "render_comparison_image",
        fake_render_comparison_image,
    )

    paths = comparison.render_requested_comparisons(
        content_path=content,
        style_path=style,
        output_dir=tmp_path,
        request=ComparisonRequest(
            include_inputs=True,
            include_result=False,
        ),
    )

    assert calls == [
        {
            "content_path": content,
            "style_path": style,
            "output_dir": tmp_path,
            "include_result": False,
            "result_path": None,
        },
    ]
    assert paths == [tmp_path / "inputs"]


def test_render_requested_comparisons_warns_when_missing_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_images: tuple[Path, Path, Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A warning is emitted when the expected stylized image is absent."""
    content, style, _ = sample_images
    expected = tmp_path / "stylized.png"

    monkeypatch.setattr(
        comparison,
        "stylized_image_path_from_paths",
        lambda *_: expected,
    )
    def fail_render_comparison_image(
        content_path: Path,
        style_path: Path,
        *,
        output_dir: Path,
        include_result: bool,
        result_path: Path | None = None,
    ) -> Path:
        _ = (content_path, style_path, output_dir, include_result, result_path)
        pytest.fail("render_comparison_image should not run")
        return output_dir  # pragma: no cover

    monkeypatch.setattr(
        comparison,
        "render_comparison_image",
        fail_render_comparison_image,
    )

    caplog.set_level("WARNING")
    result = comparison.render_requested_comparisons(
        content_path=content,
        style_path=style,
        output_dir=tmp_path,
        request=ComparisonRequest(
            include_inputs=False,
            include_result=True,
        ),
    )

    assert result == []
    assert "Expected stylized result missing" in caplog.text
