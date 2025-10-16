"""Tests for video output and intro sequencing."""

from __future__ import annotations

from pathlib import Path
from typing import Self
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

import style_transfer_visualizer.video as stv_video
from style_transfer_visualizer.config import VideoConfig

DEFAULT_INTRO_DURATION = 0.0
DEFAULT_OUTRO_DURATION = 0.0


class DummyWriter:
    """Minimal writer stub capturing appended frames for assertions."""

    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def append_data(self, frame: np.ndarray) -> None:
        """Record a frame for later assertions."""
        self.frames.append(frame)


class _SizedDummyWriter(DummyWriter):
    """Writer stub exposing an ffmpeg-style _size attribute."""

    def __init__(self, size: tuple[int, int]) -> None:
        super().__init__()
        self._size: tuple[int, int] = size


def test_setup_video_writer_returns_none_when_disabled() -> None:
    """Test that None is returned when create_video is False."""
    cfg = VideoConfig(
        fps=30,
        quality=10,
        save_every=10,
        create_video=False,
        intro_duration_seconds=DEFAULT_INTRO_DURATION,
        outro_duration_seconds=DEFAULT_OUTRO_DURATION,
    )
    result = stv_video.setup_video_writer(cfg, Path(), "test.mp4")
    assert result is None


def test_writer_called_with_correct_args(tmp_path: Path) -> None:
    """Test that get_writer is called with correct arguments."""
    cfg = VideoConfig(
        fps=30,
        quality=10,
        save_every=10,
        create_video=True,
        intro_duration_seconds=DEFAULT_INTRO_DURATION,
        outro_duration_seconds=DEFAULT_OUTRO_DURATION,
    )

    with patch(
        "style_transfer_visualizer.video.imageio.get_writer",
    ) as mock_writer:
        stv_video.setup_video_writer(cfg, tmp_path, "output.mp4")
        mock_writer.assert_called_once()
        _, kwargs = mock_writer.call_args
        params: list[str] = kwargs.get("ffmpeg_params", [])
        assert isinstance(params, list)

        def has_prefix(prefix: str) -> bool:
            """Check if any value startswith a prefix."""
            return any(isinstance(s, str) and s.startswith(prefix)
                       for s in params)

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
        intro_duration_seconds=DEFAULT_INTRO_DURATION,
        outro_duration_seconds=DEFAULT_OUTRO_DURATION,
    )
    with patch(
        "style_transfer_visualizer.video.imageio.get_writer",
    ) as mock_writer:
        stv_video.setup_video_writer(cfg, tmp_path, "video.mp4")
        _, kwargs = mock_writer.call_args
        params: list[str] = kwargs.get("ffmpeg_params", [])
        assert any(s == "title=Custom Title"
                   for s in params if isinstance(s, str))
        assert any(s == "artist=Custom Artist"
                   for s in params if isinstance(s, str))


def test_writer_non_mp4_has_no_metadata(tmp_path: Path) -> None:
    """Non-MP4 outputs should not include metadata flags."""
    cfg = VideoConfig(
        fps=24,
        quality=5,
        save_every=5,
        create_video=True,
        intro_duration_seconds=DEFAULT_INTRO_DURATION,
        outro_duration_seconds=DEFAULT_OUTRO_DURATION,
    )

    with patch("style_transfer_visualizer.video.imageio.get_writer") as mock_w:
        stv_video.setup_video_writer(cfg, tmp_path, "output.avi")
        mock_w.assert_called_once()
        _, kwargs = mock_w.call_args
        params: list[str] = kwargs.get("ffmpeg_params", [])
        assert params == []


def test_blend_frames_shape_mismatch_raises() -> None:
    """_blend_frames should validate matching frame shapes."""
    a = np.zeros((1, 1, 3), dtype=np.uint8)
    b = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape for blending"):
        stv_video._blend_frames(a, b, 0.5)  # noqa: SLF001


def test_blend_frames_linear_mix() -> None:
    """Valid inputs should return a uint8 blend."""
    a = np.zeros((1, 1, 3), dtype=np.uint8)
    b = np.full((1, 1, 3), 255, dtype=np.uint8)
    result = stv_video._blend_frames(a, b, 0.25)  # noqa: SLF001
    assert result.dtype == np.uint8
    assert 0 < result[0, 0, 0] < b[0, 0, 0]


def test_append_fade_transition_handles_non_positive_frames() -> None:
    """Frame count <= 0 should append only the end frame."""
    writer = DummyWriter()
    start = np.zeros((1, 1, 3), dtype=np.uint8)
    end = np.ones((1, 1, 3), dtype=np.uint8)
    stv_video._append_fade_transition(writer, start, end, 0)  # noqa: SLF001
    assert writer.frames == [end]


def test_build_intro_frame_raises_for_invalid_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_build_intro_frame should guard against zero-sized inputs."""

    class DummyImage:
        def __init__(self) -> None:
            self.size = (0, 64)

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(stv_video.Image, "open", lambda _path: DummyImage())

    with pytest.raises(ValueError, match="invalid dimensions"):
        stv_video._build_intro_frame(  # noqa: SLF001
            Path("content.png"),
            Path("style.png"),
        )


def test_build_intro_frame_keeps_size_when_gallery_matches(
    monkeypatch: pytest.MonkeyPatch,
    content_image: Path,
    style_image: Path,
) -> None:
    """If gallery already matches content size no resizing should occur."""

    def fake_make_gallery(**_unused: object) -> Image.Image:
        return Image.new("RGB", (64, 64))

    monkeypatch.setattr(
        stv_video,
        "make_gallery_comparison",
        fake_make_gallery,
    )

    intro = stv_video._build_intro_frame(  # noqa: SLF001
        content_image,
        style_image,
    )
    assert intro.shape == (64, 64, 3)


def test_build_intro_frame_resizes_when_gallery_too_large(
    monkeypatch: pytest.MonkeyPatch,
    content_image: Path,
    style_image: Path,
) -> None:
    """If gallery output is larger it is resized to the content size."""

    def fake_make_gallery(
        *,
        target_size: tuple[int, int],
        **_unused: object,
    ) -> Image.Image:
        return Image.new(
            "RGB",
            (target_size[0] * 2, target_size[1] * 2),
        )

    monkeypatch.setattr(
        stv_video,
        "make_gallery_comparison",
        fake_make_gallery,
    )

    intro = stv_video._build_intro_frame(  # noqa: SLF001
        content_image,
        style_image,
    )
    assert intro.shape == (64, 64, 3)


def test_prepare_intro_segment_returns_none_when_intro_disabled(
    content_image: Path,
    style_image: Path,
) -> None:
    """Intro segment should skip when intro is disabled."""
    cfg = VideoConfig(
        fps=5,
        quality=5,
        save_every=1,
        create_video=True,
        intro_enabled=False,
        intro_duration_seconds=DEFAULT_INTRO_DURATION,
        outro_duration_seconds=DEFAULT_OUTRO_DURATION,
    )
    writer = DummyWriter()
    result = stv_video.prepare_intro_segment(
        cfg,
        writer,
        content_image,
        style_image,
    )
    assert result is None
    assert writer.frames == []


def test_prepare_intro_segment_generates_frames(
    content_image: Path,
    style_image: Path,
) -> None:
    """Intro segment renders fade-in, hold, and crossfade frames."""
    cfg = VideoConfig(
        fps=5,
        quality=5,
        save_every=1,
        create_video=True,
        intro_enabled=True,
        intro_duration_seconds=0.2,
        outro_duration_seconds=DEFAULT_OUTRO_DURATION,
    )
    writer = DummyWriter()
    info = stv_video.prepare_intro_segment(
        cfg,
        writer,
        content_image,
        style_image,
    )
    assert info is not None
    intro_frame, crossfade_frames = info
    assert intro_frame.shape == (64, 64, 3)
    expected_crossfade = round(cfg.fps * stv_video.INTRO_CROSSFADE_SECONDS)
    assert crossfade_frames == expected_crossfade
    expected_fade = max(
        1,
        min(
            round(cfg.fps * stv_video.INTRO_FADE_IN_SECONDS),
            stv_video.INTRO_MAX_FADE_FRAMES,
        ),
    )
    expected_hold = max(0, round(cfg.fps * cfg.intro_duration_seconds))
    assert len(writer.frames) == expected_fade + expected_hold
    assert all(frame.shape == intro_frame.shape for frame in writer.frames)


def test_append_crossfade_handles_zero_frame_count() -> None:
    """Zero or negative frame counts should no-op."""
    writer = DummyWriter()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    stv_video.append_crossfade(writer, frame, frame, 0)
    assert writer.frames == []


def test_append_crossfade_limits_maximum_frames() -> None:
    """Crossfade frame count should be capped at the configured max."""
    writer = DummyWriter()
    start = np.zeros((2, 2, 3), dtype=np.uint8)
    end = np.full((2, 2, 3), 255, dtype=np.uint8)
    stv_video.append_crossfade(writer, start, end, 50)
    assert len(writer.frames) == stv_video.INTRO_MAX_CROSSFADE_FRAMES
    assert all(frame.shape == (2, 2, 3) for frame in writer.frames)


def test_resolve_writer_dimensions_respects_writer_size() -> None:
    """Valid writer dimensions should drive resize behaviour."""
    writer = _SizedDummyWriter((32, 48))
    last_frame = np.zeros((64, 64, 3), dtype=np.uint8)

    resized, width, height = stv_video._resolve_writer_dimensions(  # noqa: SLF001
        writer,
        last_frame,
    )

    assert (width, height) == (32, 48)
    assert resized.shape == (48, 32, 3)


def test_resolve_writer_dimensions_ignores_invalid_writer_size() -> None:
    """Non-positive writer dimensions should be ignored."""
    writer = _SizedDummyWriter((0, 48))
    last_frame = np.zeros((40, 50, 3), dtype=np.uint8)

    resized, width, height = stv_video._resolve_writer_dimensions(  # noqa: SLF001
        writer,
        last_frame,
    )

    assert (width, height) == (50, 40)
    assert resized.shape == last_frame.shape
    np.testing.assert_array_equal(resized, last_frame)


def test_append_final_comparison_frame_skips_when_disabled(
    content_image: Path,
    style_image: Path,
) -> None:
    """Final comparison frame should not be appended when disabled."""
    cfg = VideoConfig(
        fps=5,
        quality=5,
        save_every=1,
        create_video=True,
        final_frame_compare=False,
        intro_duration_seconds=DEFAULT_INTRO_DURATION,
        outro_duration_seconds=DEFAULT_OUTRO_DURATION,
    )
    writer = DummyWriter()
    last_frame = np.zeros((64, 64, 3), dtype=np.uint8)

    stv_video.append_final_comparison_frame(
        cfg,
        writer,
        content_image,
        style_image,
        last_frame,
    )

    assert writer.frames == []


def test_append_final_comparison_frame_appends_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    content_image: Path,
    style_image: Path,
) -> None:
    """Final comparison frame should be generated and appended."""
    cfg = VideoConfig(
        fps=5,
        quality=5,
        save_every=1,
        create_video=True,
        final_frame_compare=True,
        intro_duration_seconds=DEFAULT_INTRO_DURATION,
        outro_duration_seconds=0.2,
    )
    writer = DummyWriter()
    last_frame = np.zeros((64, 64, 3), dtype=np.uint8)

    def fake_comparison(**_unused: object) -> Image.Image:
        return Image.new("RGB", (64, 64), color="red")

    monkeypatch.setattr(
        stv_video,
        "make_gallery_comparison",
        fake_comparison,
    )

    stv_video.append_final_comparison_frame(
        cfg,
        writer,
        content_image,
        style_image,
        last_frame,
    )

    expected_timelapse_hold = max(
        stv_video.FINAL_TIMELAPSE_MIN_FRAMES,
        round(cfg.fps * stv_video.FINAL_TIMELAPSE_HOLD_SECONDS),
    )
    expected_crossfade = max(
        1,
        min(
            round(cfg.fps * stv_video.OUTRO_CROSSFADE_SECONDS),
            stv_video.OUTRO_MAX_CROSSFADE_FRAMES,
        ),
    )
    expected_hold = max(
        stv_video.FINAL_COMPARISON_MIN_FRAMES,
        round(cfg.fps * cfg.outro_duration_seconds),
    )
    assert len(writer.frames) == (
        expected_timelapse_hold + expected_crossfade + expected_hold
    )
    assert all(frame.shape == last_frame.shape for frame in writer.frames)
    # Ensure timelapse hold frames precede the crossfade.
    np.testing.assert_array_equal(
        writer.frames[0],
        writer.frames[expected_timelapse_hold - 1],
    )
    # The tail frames should match the final comparison image exactly.
    np.testing.assert_array_equal(
        writer.frames[-1],
        writer.frames[-expected_hold],
    )


def test_append_final_comparison_frame_validates_shape(
    content_image: Path,
    style_image: Path,
) -> None:
    """Non-RGB final frames should raise an error."""
    cfg = VideoConfig(
        fps=5,
        quality=5,
        save_every=1,
        create_video=True,
        final_frame_compare=True,
        intro_duration_seconds=DEFAULT_INTRO_DURATION,
        outro_duration_seconds=DEFAULT_OUTRO_DURATION,
    )
    writer = DummyWriter()
    last_frame = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(
        ValueError,
        match="Last timelapse frame must be an RGB array",
    ):
        stv_video.append_final_comparison_frame(
            cfg,
            writer,
            content_image,
            style_image,
            last_frame,
        )
