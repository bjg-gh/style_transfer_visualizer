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
        _, kwargs = mock_writer.call_args
        params: list[str] = kwargs.get("ffmpeg_params", [])
        assert isinstance(params, list)

        def has_prefix(prefix: str) -> bool:
            """Check if any value startswith a prefix."""
            return any(isinstance(s, str) and s.startswith(prefix)
                       for s in params)

        # Ensure metadata flags and keys exist.
        # allow either global or stream-scoped variants
        assert has_prefix("title=")
        assert has_prefix("artist=")
        assert has_prefix("comment=")
        assert has_prefix("encoder=")
        assert has_prefix("creation_time=")


def test_writer_uses_custom_title_and_artist_when_provided(
    tmp_path: Path,
) -> None:
    """Test that custom metadata fields from config are forwarded."""
    cfg = VideoConfig(
        fps=24,
        quality=6,
        save_every=1,
        create_video=True,
        metadata_title="Custom Title",
        metadata_artist="Custom Artist",
    )
    with patch(
        "style_transfer_visualizer.video.imageio.get_writer",
    ) as mock_writer:
        setup_video_writer(cfg, tmp_path, "video.mp4")
        _, kwargs = mock_writer.call_args
        params: list[str] = kwargs.get("ffmpeg_params", [])
        assert any(s == "title=Custom Title"
                   for s in params if isinstance(s, str))
        assert any(s == "artist=Custom Artist"
                   for s in params if isinstance(s, str))


def test_writer_non_mp4_has_no_metadata(tmp_path: Path) -> None:
    """Non-MP4 outputs should not include metadata flags."""
    cfg = VideoConfig(fps=24, quality=5, save_every=5, create_video=True)

    with patch("style_transfer_visualizer.video.imageio.get_writer") as mock_w:
        setup_video_writer(cfg, tmp_path, "output.avi")
        mock_w.assert_called_once()
        _, kwargs = mock_w.call_args
        params: list[str] = kwargs.get("ffmpeg_params", [])
        assert params == []  # nothing should be injected
