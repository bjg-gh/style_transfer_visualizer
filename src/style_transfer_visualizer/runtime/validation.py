"""Input validation helpers for runtime configuration."""

from __future__ import annotations

from pathlib import Path

from style_transfer_visualizer.constants import (
    VIDEO_QUALITY_MAX,
    VIDEO_QUALITY_MIN,
)


def validate_input_paths(content_path: str, style_path: str) -> None:
    """Ensure the provided content and style paths point to files."""
    if not Path(content_path).is_file():
        msg = f"Content image not found: {content_path}"
        raise FileNotFoundError(msg)
    if not Path(style_path).is_file():
        msg = f"Style image not found: {style_path}"
        raise FileNotFoundError(msg)


def validate_parameters(video_quality: int) -> None:
    """Validate that runtime parameters fall within supported ranges."""
    if video_quality < VIDEO_QUALITY_MIN or video_quality > VIDEO_QUALITY_MAX:
        msg = f"Video quality must be between 1 and 10, got {video_quality}"
        raise ValueError(msg)
