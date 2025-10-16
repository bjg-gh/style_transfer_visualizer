"\"\"\"Tests for runtime.validation helpers.\"\"\""

from __future__ import annotations

from pathlib import Path

import pytest

from style_transfer_visualizer.runtime import validation as runtime_validation


def test_validate_parameters_out_of_range() -> None:
    with pytest.raises(ValueError, match="Video quality"):
        runtime_validation.validate_parameters(video_quality=0)


def test_validate_input_paths_success(
    content_image: Path,
    style_image: Path,
) -> None:
    runtime_validation.validate_input_paths(
        str(content_image),
        str(style_image),
    )


@pytest.mark.parametrize(
    ("content_path", "style_path"),
    [
        ("missing.png", "missing.png"),
        ("missing.png", __file__),
        (__file__, "missing.png"),
    ],
)
def test_validate_input_paths_failure(
    content_path: str,
    style_path: str,
) -> None:
    with pytest.raises(FileNotFoundError):
        runtime_validation.validate_input_paths(content_path, style_path)
