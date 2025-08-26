"""Tests for video output and image saving in style_transfer_visualizer."""

from pathlib import Path
from unittest.mock import patch

from style_transfer_visualizer.config import VideoConfig
from style_transfer_visualizer.video import setup_video_writer


def test_setup_video_writer_returns_none_when_disabled() -> None:
    """Test that None is returned when create_video is False."""
    cfg = VideoConfig(fps=30, quality=10, save_every=10, create_video=False)
    result = setup_video_writer(cfg, Path(), "test.mp4")
    assert result is None


def test_writer_called_with_correct_args(tmp_path: Path) -> None:
    """Test that get_writer is called with correct arguments."""
    cfg = VideoConfig(fps=30, quality=10, save_every=10, create_video=True)
    with patch(
        "style_transfer_visualizer.video.imageio.get_writer",
    ) as mock_writer:
        setup_video_writer(cfg, tmp_path, "output.mp4")
        mock_writer.assert_called_once()
