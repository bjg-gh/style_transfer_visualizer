"""Tests for video output and intro sequencing."""

from __future__ import annotations

from pathlib import Path
from typing import Self, cast
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

import style_transfer_visualizer.video as stv_video
from style_transfer_visualizer.config import VideoConfig
from style_transfer_visualizer.config_defaults import DEFAULT_VIDEO_QUALITY
from style_transfer_visualizer.type_defs import VideoMode

pytestmark = pytest.mark.visual

DEFAULT_INTRO_DURATION = 0.0
DEFAULT_OUTRO_DURATION = 0.0
EXPECTED_CAPTURED_FRAMES = 2
AUTO_SWITCH_STEPS = 2000
OVERRIDE_STEPS = 5000


class DummyWriter:
    """Minimal writer stub capturing appended frames for assertions."""

    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self._size: tuple[int, int] | None = None

    def append_data(self, frame: np.ndarray) -> None:
        """Record a frame for later assertions."""
        rgb = np.asarray(frame, dtype=np.uint8)
        self._size = (rgb.shape[1], rgb.shape[0])
        self.frames.append(rgb)

    def close(self) -> None:
        """No-op close method to satisfy writer protocol."""
        return


class _SizedDummyWriter(DummyWriter):
    """Writer stub exposing an ffmpeg-style _size attribute."""

    def __init__(self, size: tuple[int, int]) -> None:
        super().__init__()
        self._size: tuple[int, int] | None = size


def _reason_config(*, save_every: int = 1, fps: int = 10) -> VideoConfig:
    """Build a VideoConfig instance for auto-detection tests."""
    return VideoConfig(
        save_every=save_every,
        fps=fps,
        quality=DEFAULT_VIDEO_QUALITY,
        mode="realtime",
        intro_duration_seconds=DEFAULT_INTRO_DURATION,
        outro_duration_seconds=DEFAULT_OUTRO_DURATION,
    )


def test_setup_video_writer_returns_none_when_disabled() -> None:
    """Test that None is returned when create_video is False."""
    cfg = VideoConfig(
        fps=30,
        quality=10,
        save_every=10,
        create_video=False,
        intro_duration_seconds=DEFAULT_INTRO_DURATION,
        outro_duration_seconds=DEFAULT_OUTRO_DURATION,
        mode="realtime",
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
        mode="realtime",
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
        mode="realtime",
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
        mode="realtime",
    )

    with patch("style_transfer_visualizer.video.imageio.get_writer") as mock_w:
        stv_video.setup_video_writer(cfg, tmp_path, "output.avi")
        mock_w.assert_called_once()
        _, kwargs = mock_w.call_args
        params: list[str] = kwargs.get("ffmpeg_params", [])
        assert params == []


def test_ensure_rgb_uint8_converts_dtype() -> None:
    """Helper should coerce non-uint8 frames before writing."""
    max_val = float(np.iinfo(np.uint8).max)
    float_frame = np.full((2, 2, 3), max_val, dtype=np.float32)

    result = stv_video._ensure_rgb_uint8(float_frame)  # noqa: SLF001

    assert result.dtype == np.uint8
    assert int(result[0, 0, 0]) == np.iinfo(np.uint8).max


def test_setup_video_writer_returns_postprocess_writer_when_mode_postprocess(
    tmp_path: Path,
) -> None:
    """Selecting postprocess mode should return the collecting writer."""
    cfg = VideoConfig(
        fps=12,
        quality=7,
        save_every=5,
        create_video=True,
        intro_duration_seconds=DEFAULT_INTRO_DURATION,
        outro_duration_seconds=DEFAULT_OUTRO_DURATION,
        mode="postprocess",
    )
    writer = stv_video.setup_video_writer(cfg, tmp_path, "timelapse.mp4")
    assert isinstance(writer, stv_video.PostprocessVideoWriter)


def test_setup_video_writer_rejects_unknown_mode(tmp_path: Path) -> None:
    """Unsupported modes should raise a clear error."""
    cfg = VideoConfig(
        fps=12,
        quality=7,
        save_every=1,
        create_video=True,
        intro_duration_seconds=DEFAULT_INTRO_DURATION,
        outro_duration_seconds=DEFAULT_OUTRO_DURATION,
        mode="realtime",
    )
    cfg.mode = cast(VideoMode, "bogus")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unsupported video mode"):
        stv_video.setup_video_writer(cfg, tmp_path, "video.mp4")


def test_setup_gif_collector_returns_none_when_disabled(tmp_path: Path) -> None:
    """GIF collector should not be created when disabled in config."""
    cfg = VideoConfig(
        fps=12,
        quality=7,
        save_every=1,
        create_video=True,
        intro_duration_seconds=DEFAULT_INTRO_DURATION,
        outro_duration_seconds=DEFAULT_OUTRO_DURATION,
        mode="realtime",
        create_gif=False,
    )
    collector = stv_video.setup_gif_collector(cfg, tmp_path, "anim.gif")
    assert collector is None


def test_setup_gif_collector_creates_collector(tmp_path: Path) -> None:
    """When enabled, the GIF collector should be returned."""
    cfg = VideoConfig(
        fps=8,
        quality=6,
        save_every=2,
        create_video=False,
        intro_duration_seconds=DEFAULT_INTRO_DURATION,
        outro_duration_seconds=DEFAULT_OUTRO_DURATION,
        mode="realtime",
        create_gif=True,
    )
    collector = stv_video.setup_gif_collector(cfg, tmp_path, "anim.gif")
    try:
        assert isinstance(collector, stv_video.GifFrameCollector)
    finally:
        if collector is not None:
            collector.close()


def test_postprocess_writer_collects_and_encodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Postprocess writer should flush collected frames via helper."""
    cfg = VideoConfig(
        fps=12,
        quality=7,
        save_every=1,
        create_video=True,
        intro_duration_seconds=DEFAULT_INTRO_DURATION,
        outro_duration_seconds=DEFAULT_OUTRO_DURATION,
        mode="postprocess",
    )
    sink = stv_video.setup_video_writer(cfg, tmp_path, "video.mp4")
    assert isinstance(sink, stv_video.PostprocessVideoWriter)

    frame_a = np.zeros((32, 48, 3), dtype=np.uint8)
    frame_b = np.ones((32, 48, 3), dtype=np.uint8) * 127
    sink.append_data(frame_a)
    sink.append_data(frame_b)

    captured: list[np.ndarray] = []

    class FakeWriter(DummyWriter):
        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            _exc_type: object,
            _exc: object,
            _tb: object,
        ) -> bool:
            return False

        def append_data(self, frame: np.ndarray) -> None:
            captured.append(np.asarray(frame, dtype=np.uint8))

    monkeypatch.setattr(
        stv_video,
        "_open_imageio_writer",
        lambda *_args, **_kwargs: FakeWriter(),
    )

    sink.close()

    assert len(captured) == EXPECTED_CAPTURED_FRAMES
    np.testing.assert_array_equal(captured[0], frame_a)
    np.testing.assert_array_equal(captured[1], frame_b)
    assert not list(tmp_path.glob("stv_frames_*"))


def test_postprocess_writer_close_guards_after_close(tmp_path: Path) -> None:
    """Postprocess writer should ignore double close and reject new frames."""
    cfg = VideoConfig(
        fps=12,
        quality=7,
        save_every=1,
        create_video=True,
        intro_duration_seconds=DEFAULT_INTRO_DURATION,
        outro_duration_seconds=DEFAULT_OUTRO_DURATION,
        mode="postprocess",
    )
    writer = stv_video.PostprocessVideoWriter(cfg, tmp_path / "video.mp4")

    writer.close()
    writer.close()  # second call should be a no-op

    with pytest.raises(RuntimeError, match="closed"):
        writer.append_data(np.zeros((4, 4, 3), dtype=np.uint8))


def test_select_video_mode_auto_switches() -> None:
    """Heuristic should move to postprocess for long high-res runs."""
    cfg = VideoConfig(
        save_every=1,
        fps=12,
        quality=DEFAULT_VIDEO_QUALITY,
        intro_duration_seconds=DEFAULT_INTRO_DURATION,
        outro_duration_seconds=DEFAULT_OUTRO_DURATION,
        mode="realtime",
    )
    mode, reason, frames = stv_video.select_video_mode(
        cfg,
        frame_size=(3840, 2160),
        total_steps=AUTO_SWITCH_STEPS,
    )
    assert mode == "postprocess"
    assert reason is not None
    assert frames == AUTO_SWITCH_STEPS


def test_select_video_mode_respects_explicit_override() -> None:
    """Explicitly configured modes should not be overridden."""
    cfg = VideoConfig(
        save_every=1,
        fps=60,
        quality=DEFAULT_VIDEO_QUALITY,
        intro_duration_seconds=DEFAULT_INTRO_DURATION,
        outro_duration_seconds=DEFAULT_OUTRO_DURATION,
        mode="realtime",
        mode_override=True,
    )
    mode, reason, frames = stv_video.select_video_mode(
        cfg,
        frame_size=(4096, 2160),
        total_steps=OVERRIDE_STEPS,
    )
    assert mode == "realtime"
    assert reason is None
    assert frames == OVERRIDE_STEPS


def test_auto_postprocess_reason_handles_zero_save_every() -> None:
    """Heuristic should return early when save_every is invalid."""
    cfg = _reason_config()
    cfg.save_every = 0

    reason, frames = stv_video._auto_postprocess_reason(  # noqa: SLF001
        cfg,
        frame_size=(640, 480),
        total_steps=100,
    )

    assert reason is None
    assert frames == 0


def test_auto_postprocess_reason_handles_zero_frames() -> None:
    """No estimated frames should skip postprocess recommendation."""
    cfg = _reason_config()

    reason, frames = stv_video._auto_postprocess_reason(  # noqa: SLF001
        cfg,
        frame_size=(640, 480),
        total_steps=0,
    )

    assert reason is None
    assert frames == 0


def test_auto_postprocess_reason_handles_invalid_dimensions() -> None:
    """Non-positive dimensions should bypass heuristic."""
    cfg = _reason_config()
    total_steps = 50

    reason, frames = stv_video._auto_postprocess_reason(  # noqa: SLF001
        cfg,
        frame_size=(0, 480),
        total_steps=total_steps,
    )

    assert reason is None
    assert frames == total_steps


def test_auto_postprocess_reason_ultra_high_res() -> None:
    """4K-class runs should be promoted when thresholds are met."""
    cfg = _reason_config()
    total_steps = stv_video._AUTO_ULTRA_RES_FRAME_THRESHOLD  # noqa: SLF001

    reason, frames = stv_video._auto_postprocess_reason(  # noqa: SLF001
        cfg,
        frame_size=(3840, 2160),
        total_steps=total_steps,
    )

    assert reason is not None
    assert "4K-class" in reason
    assert frames == total_steps


def test_auto_postprocess_reason_high_res() -> None:
    """High-resolution but sub-4K runs should trigger the high-res branch."""
    cfg = _reason_config()
    total_steps = stv_video._AUTO_HIGH_RES_FRAME_THRESHOLD  # noqa: SLF001

    reason, frames = stv_video._auto_postprocess_reason(  # noqa: SLF001
        cfg,
        frame_size=(1920, 1080),
        total_steps=total_steps,
    )

    assert reason is not None
    assert "high-res" in reason
    assert frames == total_steps


def test_auto_postprocess_reason_high_fps() -> None:
    """High frame-rate runs can also trigger postprocess mode."""
    cfg = _reason_config(
        fps=stv_video._AUTO_HIGH_FPS_THRESHOLD,  # noqa: SLF001
    )
    total_steps = stv_video._AUTO_HIGH_FPS_FRAME_THRESHOLD  # noqa: SLF001

    reason, frames = stv_video._auto_postprocess_reason(  # noqa: SLF001
        cfg,
        frame_size=(800, 600),
        total_steps=total_steps,
    )

    assert reason is not None
    assert "fps" in reason
    assert frames == total_steps


def test_auto_postprocess_reason_dense_sampling() -> None:
    """Frequent frame dumps should trigger the save-every branch."""
    cfg = _reason_config(save_every=1, fps=10)
    total_steps = stv_video._AUTO_SAVE_EVERY_FRAME_THRESHOLD  # noqa: SLF001

    reason, frames = stv_video._auto_postprocess_reason(  # noqa: SLF001
        cfg,
        frame_size=(320, 240),
        total_steps=total_steps,
    )

    assert reason is not None
    assert "--save-every" in reason
    assert frames == total_steps


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
        mode="realtime",
    )
    writer = DummyWriter()
    result = stv_video.prepare_intro_segment(
        cfg,
        writer,
        (content_image, style_image),
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
        mode="realtime",
    )
    writer = DummyWriter()
    info = stv_video.prepare_intro_segment(
        cfg,
        writer,
        (content_image, style_image),
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


def test_prepare_intro_segment_appends_hold_frames_to_all_sinks(
    content_image: Path,
    style_image: Path,
) -> None:
    """Hold frames should be emitted to both writer and GIF sink when requested."""
    cfg = VideoConfig(
        fps=2,
        quality=5,
        save_every=1,
        create_video=True,
        create_gif=True,
        intro_enabled=True,
        intro_duration_seconds=0.5,  # ensure at least one hold frame
        outro_duration_seconds=DEFAULT_OUTRO_DURATION,
        mode="realtime",
    )
    video_writer = DummyWriter()
    gif_writer = DummyWriter()
    gif_options = stv_video.GifSegmentOptions(
        sink=gif_writer,
        include_intro=True,
    )

    info = stv_video.prepare_intro_segment(
        cfg,
        video_writer,
        (content_image, style_image),
        gif_options=gif_options,
    )

    assert info is not None
    assert video_writer.frames  # hold frames captured
    assert gif_writer.frames  # gif sink also received frames


def test_prepare_intro_segment_emits_gif_frames_only(
    content_image: Path,
    style_image: Path,
) -> None:
    """Intro segment should still render when only GIF output is enabled."""
    cfg = VideoConfig(
        fps=5,
        quality=5,
        save_every=1,
        create_video=False,
        create_gif=True,
        intro_enabled=True,
        intro_duration_seconds=0.0,
        outro_duration_seconds=DEFAULT_OUTRO_DURATION,
        mode="realtime",
    )
    gif_writer = DummyWriter()
    gif_options = stv_video.GifSegmentOptions(
        sink=gif_writer,
        include_intro=True,
    )
    info = stv_video.prepare_intro_segment(
        cfg,
        None,
        (content_image, style_image),
        gif_options=gif_options,
    )
    assert info is not None
    assert gif_writer.frames, "GIF writer should capture intro frames"


def test_prepare_intro_segment_skips_gif_when_not_requested(
    content_image: Path,
    style_image: Path,
) -> None:
    """GIF sink should remain empty when intro inclusion is disabled."""
    cfg = VideoConfig(
        fps=5,
        quality=5,
        save_every=1,
        create_video=False,
        create_gif=True,
        intro_enabled=True,
        intro_duration_seconds=0.0,
        outro_duration_seconds=DEFAULT_OUTRO_DURATION,
        mode="realtime",
    )
    gif_writer = DummyWriter()
    gif_options = stv_video.GifSegmentOptions(
        sink=gif_writer,
        include_intro=False,
    )
    result = stv_video.prepare_intro_segment(
        cfg,
        None,
        (content_image, style_image),
        gif_options=gif_options,
    )
    assert result is None
    assert gif_writer.frames == []


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


def test_append_final_comparison_frame_emits_gif_frames(
    content_image: Path,
    style_image: Path,
) -> None:
    """Final comparison frame should be emitted to GIF sink when enabled."""
    cfg = VideoConfig(
        fps=6,
        quality=5,
        save_every=1,
        create_video=False,
        create_gif=True,
        intro_duration_seconds=DEFAULT_INTRO_DURATION,
        outro_duration_seconds=0.0,
        final_frame_compare=True,
        mode="realtime",
    )
    gif_writer = DummyWriter()
    last_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    gif_options = stv_video.GifSegmentOptions(
        sink=gif_writer,
        include_outro=True,
    )

    stv_video.append_final_comparison_frame(
        cfg,
        None,
        (content_image, style_image),
        last_frame,
        gif_options=gif_options,
    )

    assert gif_writer.frames, "GIF writer should capture outro frames"


def test_append_final_comparison_frame_skips_gif_when_not_requested(
    content_image: Path,
    style_image: Path,
) -> None:
    """GIF sink should remain untouched when outro inclusion is disabled."""
    cfg = VideoConfig(
        fps=6,
        quality=5,
        save_every=1,
        create_video=False,
        create_gif=True,
        intro_duration_seconds=DEFAULT_INTRO_DURATION,
        outro_duration_seconds=0.0,
        final_frame_compare=True,
        mode="realtime",
    )
    gif_writer = DummyWriter()
    last_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    gif_options = stv_video.GifSegmentOptions(
        sink=gif_writer,
        include_outro=False,
    )

    stv_video.append_final_comparison_frame(
        cfg,
        None,
        (content_image, style_image),
        last_frame,
        gif_options=gif_options,
    )

    assert gif_writer.frames == []


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
        mode="realtime",
    )
    writer = DummyWriter()
    last_frame = np.zeros((64, 64, 3), dtype=np.uint8)

    stv_video.append_final_comparison_frame(
        cfg,
        writer,
        (content_image, style_image),
        last_frame,
    )

    assert writer.frames == []


def test_append_final_comparison_frame_returns_when_no_targets(
    content_image: Path,
    style_image: Path,
) -> None:
    """Function should return early when neither writer nor GIF sink is active."""
    cfg = VideoConfig(
        fps=5,
        quality=5,
        save_every=1,
        create_video=False,
        create_gif=False,
        final_frame_compare=True,
        intro_duration_seconds=DEFAULT_INTRO_DURATION,
        outro_duration_seconds=DEFAULT_OUTRO_DURATION,
        mode="realtime",
    )
    last_frame = np.zeros((10, 10, 3), dtype=np.uint8)

    # Should not raise when both sinks are inactive.
    stv_video.append_final_comparison_frame(
        cfg,
        None,
        (content_image, style_image),
        last_frame,
    )


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
        mode="realtime",
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
        (content_image, style_image),
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
        mode="realtime",
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
            (content_image, style_image),
            last_frame,
        )



def test_gif_frame_collector_writes_gif(tmp_path: Path) -> None:
    """Collector should emit a GIF file and clean temp storage."""
    output_path = tmp_path / "anim.gif"
    collector = stv_video.GifFrameCollector(output_path, fps=5)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    collector.append_data(frame)
    collector.append_data(frame)
    collector.close()
    assert output_path.exists()
    assert not collector._temp_dir.exists()  # noqa: SLF001
    with pytest.raises(RuntimeError, match="Cannot append frame"):
        collector.append_data(frame)
    collector.close()
