"""Tests for video output and image saving in style_transfer_visualizer.
"""
from pathlib import Path
from unittest.mock import patch

from style_transfer_visualizer.video import setup_video_writer


def test_setup_video_writer_returns_none_when_disabled():
    result = setup_video_writer(
        output_path=Path("."),
        video_name="test.mp4",
        fps=24,
        video_quality=5,
        create_video=False
    )
    assert result is None


def test_writer_called_with_correct_args(tmp_path):
    with patch("style_transfer_visualizer.video.imageio.get_writer"
               ) as mock_writer:
        setup_video_writer(
            output_path=tmp_path,
            video_name="output.mp4",
            fps=30,
            video_quality=10,
            create_video=True
        )
        mock_writer.assert_called_once()