"""Handles timelapse video writer and output file saving."""

from __future__ import annotations

import shutil
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import imageio
import numpy as np
from PIL import Image

from style_transfer_visualizer.constants import (
    COLOR_GREY,
    ENCODING_BLOCK_SIZE,
    VIDEO_CODEC,
)
from style_transfer_visualizer.image_grid.core import FrameParams
from style_transfer_visualizer.image_grid.layouts import (
    make_gallery_comparison,
)
from style_transfer_visualizer.runtime.version import resolve_project_version

if TYPE_CHECKING:  # pragma: no cover
    from style_transfer_visualizer.config import VideoConfig
    from style_transfer_visualizer.type_defs import VideoMode


def _utc_timestamp() -> str:
    """Return an ISO 8601 UTC timestamp suitable for container tags."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_mp4_metadata_params(
    title: str | None,
    artist: str | None,
) -> list[str]:
    """
    Construct ffmpeg_params for broadly compatible MP4 metadata.

    Tags are written at the container level and on the first video
    stream. Keys are chosen for wide recognition across Windows,
    Linux tools, and Apple QuickTime stack.
    """
    version = resolve_project_version()
    ts = _utc_timestamp()

    eff_title = title or "Style Transfer Visualizer Output"
    eff_artist = artist or "Style Transfer Visualizer"
    comment = f"Created using style_transfer_visualizer v{version}"
    enc = f"style_transfer_visualizer v{version}"

    def add_tags(args: list[str]) -> None:
        args.extend(["-metadata", f"title={eff_title}"])
        args.extend(["-metadata", f"artist={eff_artist}"])
        args.extend(["-metadata", f"comment={comment}"])
        args.extend(["-metadata", f"encoder={enc}"])
        args.extend(["-metadata", f"creation_time={ts}"])

    params: list[str] = [] # ["-movflags", "use_metadata_tags"]

    # container tags
    add_tags(params)

    return params

INTRO_FADE_IN_SECONDS = 1.0
INTRO_CROSSFADE_SECONDS = 0.5
INTRO_MAX_FADE_FRAMES = 48
INTRO_MAX_CROSSFADE_FRAMES = 12
INTRO_MIN_DIM = 128
OUTRO_CROSSFADE_SECONDS = 0.5
OUTRO_MAX_CROSSFADE_FRAMES = 12
OUTRO_MIN_DIM = 512
FINAL_COMPARISON_MIN_FRAMES = 1
FINAL_TIMELAPSE_HOLD_SECONDS = 1.0
FINAL_TIMELAPSE_MIN_FRAMES = 1
_FRAME_NDIMS = 3
_RGB_CHANNELS = 3
_SIZE_TUPLE_LEN = 2
_PNG_SUFFIX = ".png"
_MAX_RGB_VALUE = 255
_MEGAPIXEL = 1_000_000
_AUTO_LONG_RUN_FRAME_THRESHOLD = 2400
_AUTO_HIGH_RES_AREA = 2560 * 1440
_AUTO_HIGH_RES_FRAME_THRESHOLD = 2000
_AUTO_ULTRA_RES_AREA = 3840 * 2160
_AUTO_ULTRA_RES_FRAME_THRESHOLD = 280
_AUTO_HIGH_FPS_THRESHOLD = 48
_AUTO_HIGH_FPS_FRAME_THRESHOLD = 2000
_AUTO_SAVE_EVERY_THRESHOLD = 5
_AUTO_SAVE_EVERY_FRAME_THRESHOLD = 2000


def _ensure_rgb_uint8(
    frame: np.ndarray,
    *,
    message: str | None = None,
) -> np.ndarray:
    """Validate and return an RGB uint8 frame."""
    if frame.ndim != _FRAME_NDIMS or frame.shape[-1] != _RGB_CHANNELS:
        msg = message or "Frames must be RGB arrays with shape (H, W, 3)"
        raise ValueError(msg)
    if frame.dtype != np.uint8:
        frame = np.clip(
            np.rint(frame),
            0,
            _MAX_RGB_VALUE,
        ).astype(np.uint8)
    return np.asarray(frame, dtype=np.uint8)


class VideoFrameSink(Protocol):
    """Minimal protocol for writer-like objects used in the pipeline."""

    _size: tuple[int, int] | None

    def append_data(self, frame: np.ndarray) -> None:
        """Append a single RGB frame to the sink."""

    def close(self) -> None:
        """Flush and release any resources held by the sink."""


@dataclass(slots=True)
class GifSegmentOptions:
    """Options controlling optional GIF segment emission."""

    sink: VideoFrameSink | None = None
    include_intro: bool = False
    include_outro: bool = False


def _blend_frames(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Linearly blend two RGB frames with given alpha."""
    if frame_a.shape != frame_b.shape:
        msg = "Frames must share shape for blending"
        raise ValueError(msg)
    inv_alpha = 1.0 - alpha
    mixed = (
        frame_a.astype(np.float32) * inv_alpha
        + frame_b.astype(np.float32) * alpha
    )
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def _append_fade_transition(
    writer: VideoFrameSink,
    start_frame: np.ndarray,
    end_frame: np.ndarray,
    frame_count: int,
) -> None:
    """Append a fade transition from start_frame to end_frame."""
    if frame_count <= 0:
        writer.append_data(end_frame)
        return
    for idx in range(frame_count):
        alpha = (idx + 1) / frame_count
        writer.append_data(_blend_frames(start_frame, end_frame, alpha))


def _build_intro_frame(content_path: Path, style_path: Path) -> np.ndarray:
    """Construct the comparison intro frame as an RGB array."""
    with ExitStack() as stack:
        content = stack.enter_context(Image.open(content_path))
        style = stack.enter_context(Image.open(style_path))
        base_w, base_h = content.size
        if base_w <= 0 or base_h <= 0:
            msg = "Content image has invalid dimensions"
            raise ValueError(msg)
        scale_w = INTRO_MIN_DIM / base_w if base_w < INTRO_MIN_DIM else 1.0
        scale_h = INTRO_MIN_DIM / base_h if base_h < INTRO_MIN_DIM else 1.0
        scale = max(scale_w, scale_h, 1.0)
        safe_size = (
            max(1, round(base_w * scale)),
            max(1, round(base_h * scale)),
        )
        frame_params = FrameParams(frame_tone="gold", label="on")
        gallery = make_gallery_comparison(
            content=content,
            style=style,
            result=None,
            target_size=safe_size,
            layout="gallery-two-across",
            wall_color=COLOR_GREY,
            frame=frame_params,
        )
        if gallery.size != content.size:
            gallery = gallery.resize(content.size, Image.Resampling.LANCZOS)
    return np.asarray(gallery.convert("RGB"), dtype=np.uint8)


def prepare_intro_segment(
    config: VideoConfig,
    writer: VideoFrameSink | None,
    paths: tuple[Path, Path],
    gif_options: GifSegmentOptions | None = None,
) -> tuple[np.ndarray, int] | None:
    """
    Render intro sequence, returning the last intro frame and crossfade length.

    Frames are emitted to the provided writer and, optionally, to a GIF sink
    when the corresponding option is enabled. Returns ``None`` when no sink
    requires the intro sequence.
    """
    content_path, style_path = paths
    gif_sink = gif_options.sink if gif_options else None
    include_gif_intro = bool(gif_options and gif_options.include_intro)

    use_writer = (
        writer is not None
        and config.create_video
        and config.intro_enabled
    )
    use_gif = (
        gif_sink is not None
        and include_gif_intro
        and config.intro_enabled
    )

    if not use_writer and not use_gif:
        return None

    intro_frame = _build_intro_frame(content_path, style_path)
    fade_frames_raw = round(config.fps * INTRO_FADE_IN_SECONDS)
    fade_frames = max(1, min(fade_frames_raw, INTRO_MAX_FADE_FRAMES))
    hold_frames_raw = round(config.fps * config.intro_duration_seconds)
    hold_frames = max(0, hold_frames_raw)

    black = np.zeros_like(intro_frame)
    writer_sink = writer if use_writer else None
    gif_sink_live = gif_sink if use_gif else None

    if writer_sink is not None:
        _append_fade_transition(writer_sink, black, intro_frame, fade_frames)
    if gif_sink_live is not None:
        _append_fade_transition(gif_sink_live, black, intro_frame, fade_frames)

    for _ in range(hold_frames):
        if writer_sink is not None:
            writer_sink.append_data(intro_frame)
        else:  # pragma: no cover - writer path exercised in tests
            pass
        if gif_sink_live is not None:
            gif_sink_live.append_data(intro_frame)

    crossfade_raw = round(config.fps * INTRO_CROSSFADE_SECONDS)
    crossfade_frames = max(1, min(crossfade_raw, INTRO_MAX_CROSSFADE_FRAMES))
    return intro_frame, crossfade_frames


def append_crossfade(
    writer: VideoFrameSink,
    start_frame: np.ndarray,
    end_frame: np.ndarray,
    frame_count: int,
    *,
    max_frames: int = INTRO_MAX_CROSSFADE_FRAMES,
) -> None:
    """Append a quick crossfade between intro and stylized frame."""
    if frame_count <= 0:
        return
    limited = max(1, min(frame_count, max_frames))
    for idx in range(limited):
        alpha = (idx + 1) / (limited + 1)
        writer.append_data(_blend_frames(start_frame, end_frame, alpha))


def _resolve_writer_dimensions(
    writer: VideoFrameSink,
    last_frame: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Return the resized last frame and writer-aligned dimensions."""
    last_rgb = _ensure_rgb_uint8(
        last_frame,
        message="Last timelapse frame must be an RGB array",
    )
    target_width = last_rgb.shape[1]
    target_height = last_rgb.shape[0]

    writer_size = getattr(writer, "_size", None)
    if isinstance(writer_size, tuple) and len(writer_size) == _SIZE_TUPLE_LEN:
        writer_w, writer_h = writer_size
        if writer_w > 0 and writer_h > 0:
            target_width = int(writer_w)
            target_height = int(writer_h)

    if (target_height, target_width) != last_rgb.shape[:2]:
        resized = Image.fromarray(last_rgb).resize(
            (target_width, target_height),
            Image.Resampling.LANCZOS,
        )
        last_rgb = np.asarray(resized, dtype=np.uint8)

    return last_rgb, target_width, target_height


def _build_outro_frame(
    content_style_paths: tuple[Path, Path],
    result_image: Image.Image,
    frame_params: FrameParams,
    *,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Create the outro comparison image resized for the video writer."""
    render_width = max(target_width, OUTRO_MIN_DIM)
    render_height = max(target_height, OUTRO_MIN_DIM)
    render_size = (render_width, render_height)

    with ExitStack() as stack:
        content_path, style_path = content_style_paths
        content = stack.enter_context(Image.open(content_path))
        style = stack.enter_context(Image.open(style_path))
        comparison = make_gallery_comparison(
            content=content,
            style=style,
            result=result_image,
            target_size=render_size,
            layout="gallery-stacked-left",
            wall_color=COLOR_GREY,
            frame=frame_params,
        )

    comparison = comparison.convert("RGB")
    if comparison.size != (target_width, target_height):
        comparison = comparison.resize(
            (target_width, target_height),
            Image.Resampling.LANCZOS,
        )
    return np.asarray(comparison, dtype=np.uint8)


def append_final_comparison_frame(
    config: VideoConfig,
    writer: VideoFrameSink | None,
    paths: tuple[Path, Path],
    last_frame: np.ndarray,
    gif_options: GifSegmentOptions | None = None,
) -> None:
    """
    Append a final comparison frame showing content, style, and result.

    Uses the gallery wall layout with labels to mirror the standalone
    comparison output. Includes a short crossfade from the most recent
    timelapse frame followed by a configurable hold duration. No-op when
    the feature is disabled.
    """
    gif_sink = gif_options.sink if gif_options else None
    include_gif_outro = bool(gif_options and gif_options.include_outro)

    use_writer = (
        writer is not None
        and config.create_video
        and config.final_frame_compare
    )
    use_gif = (
        gif_sink is not None
        and include_gif_outro
        and config.final_frame_compare
    )

    if not use_writer and not use_gif:
        return

    validated_last = _ensure_rgb_uint8(
        last_frame,
        message="Last timelapse frame must be an RGB array",
    )
    result_image = Image.fromarray(
        np.asarray(validated_last, dtype=np.uint8),
    )
    frame_params = FrameParams(frame_tone="gold", label="on")

    targets: list[tuple[VideoFrameSink, np.ndarray, np.ndarray]] = []

    if writer is not None and use_writer:
        targets.append(
            _prepare_outro_target(
                sink=writer,
                last_frame=validated_last,
                paths=paths,
                result_image=result_image,
                frame_params=frame_params,
            ),
        )
    if gif_sink is not None and use_gif:
        targets.append(
            _prepare_outro_target(
                sink=gif_sink,
                last_frame=validated_last,
                paths=paths,
                result_image=result_image,
                frame_params=frame_params,
            ),
        )

    timelapse_hold_raw = round(config.fps * FINAL_TIMELAPSE_HOLD_SECONDS)
    timelapse_hold_frames = max(FINAL_TIMELAPSE_MIN_FRAMES, timelapse_hold_raw)
    for _ in range(timelapse_hold_frames):
        for sink, last_rgb, _ in targets:
            sink.append_data(last_rgb)

    crossfade_raw = round(config.fps * OUTRO_CROSSFADE_SECONDS)
    crossfade_frames = max(1, min(crossfade_raw, OUTRO_MAX_CROSSFADE_FRAMES))
    for sink, last_rgb, frame_np in targets:
        append_crossfade(
            sink,
            last_rgb,
            frame_np,
            crossfade_frames,
            max_frames=OUTRO_MAX_CROSSFADE_FRAMES,
        )

    outro_seconds = max(0.0, config.outro_duration_seconds)
    hold_frames_raw = round(config.fps * outro_seconds)
    hold_frames = max(FINAL_COMPARISON_MIN_FRAMES, hold_frames_raw)
    for _ in range(hold_frames):
        for sink, _, frame_np in targets:
            sink.append_data(frame_np)


def _prepare_outro_target(
    *,
    sink: VideoFrameSink,
    last_frame: np.ndarray,
    paths: tuple[Path, Path],
    result_image: Image.Image,
    frame_params: FrameParams,
) -> tuple[VideoFrameSink, np.ndarray, np.ndarray]:
    """Return sink paired with aligned timelapse and outro frames."""
    last_rgb, target_width, target_height = _resolve_writer_dimensions(
        sink,
        last_frame,
    )
    frame_np = _build_outro_frame(
        paths,
        result_image,
        frame_params,
        target_width=target_width,
        target_height=target_height,
    )
    return sink, last_rgb, frame_np


class PostprocessVideoWriter:
    """Collect frames on disk and encode them once optimization completes."""

    def __init__(self, config: VideoConfig, output_path: Path) -> None:
        self._config = config
        self._output_path = output_path
        self._temp_dir = Path(
            tempfile.mkdtemp(
                prefix="stv_frames_",
                dir=output_path.parent,
            ),
        )
        self._frames: list[Path] = []
        self._closed = False
        self._size: tuple[int, int] | None = None

    def append_data(self, frame: np.ndarray) -> None:
        """Persist a single frame to the temporary frame directory."""
        if self._closed:
            msg = "Cannot append frame after writer has been closed."
            raise RuntimeError(msg)

        rgb = _ensure_rgb_uint8(frame)
        height, width = rgb.shape[:2]
        self._size = (width, height)

        frame_path = self._temp_dir / (
            f"frame_{len(self._frames):08d}{_PNG_SUFFIX}"
        )
        Image.fromarray(rgb, mode="RGB").save(frame_path, format="PNG")
        self._frames.append(frame_path)

    def close(self) -> None:
        """Encode the accumulated frames and clean up temporary storage."""
        if self._closed:
            return

        self._closed = True
        try:
            if not self._frames:
                return

            with _open_imageio_writer(
                self._config,
                self._output_path,
            ) as writer:
                for frame_path in self._frames:
                    with Image.open(frame_path) as img:
                        writer.append_data(
                            np.asarray(img.convert("RGB"), dtype=np.uint8),
                        )
        finally:
            shutil.rmtree(self._temp_dir, ignore_errors=True)


class GifFrameCollector:
    """Collect frames destined for GIF export and encode them on close."""

    def __init__(self, output_path: Path, fps: int) -> None:
        self._output_path = output_path
        self._fps = max(1, fps)
        self._temp_dir = Path(
            tempfile.mkdtemp(
                prefix="stv_gif_",
                dir=output_path.parent,
            ),
        )
        self._frames: list[Path] = []
        self._closed = False
        self._size: tuple[int, int] | None = None

    def append_data(self, frame: np.ndarray) -> None:
        """Persist a frame for inclusion in the final GIF."""
        if self._closed:
            msg = "Cannot append frame after GIF collector has been closed."
            raise RuntimeError(msg)

        rgb = _ensure_rgb_uint8(frame)
        height, width = rgb.shape[:2]
        self._size = (width, height)

        frame_path = self._temp_dir / (
            f"gif_{len(self._frames):08d}{_PNG_SUFFIX}"
        )
        Image.fromarray(rgb, mode="RGB").save(frame_path, format="PNG")
        self._frames.append(frame_path)

    def close(self) -> None:
        """Encode collected frames into a GIF and delete temporary storage."""
        if self._closed:
            return

        self._closed = True
        try:
            if not self._frames:
                return

            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            duration = 1.0 / float(self._fps)
            writer_ctx = imageio.get_writer(
                self._output_path.as_posix(),
                mode="I",
                duration=duration,
                loop=0,
            )
            with cast("Any", writer_ctx) as writer:
                for frame_path in self._frames:
                    with Image.open(frame_path) as img:
                        writer.append_data(
                            np.asarray(img.convert("RGB"), dtype=np.uint8),
                        )
        finally:
            shutil.rmtree(self._temp_dir, ignore_errors=True)


def _open_imageio_writer(
    config: VideoConfig,
    output_path: Path,
) -> imageio.plugins.ffmpeg.FfmpegFormat.Writer:
    """Create an imageio writer with config-derived metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_params: list[str] = []
    if output_path.suffix.lower() == ".mp4":
        ffmpeg_params = _build_mp4_metadata_params(
            title=config.metadata_title,
            artist=config.metadata_artist,
        )

    return imageio.get_writer(
        output_path.as_posix(),
        fps=config.fps,
        codec=VIDEO_CODEC,
        quality=config.quality,
        mode="I",
        macro_block_size=ENCODING_BLOCK_SIZE,
        ffmpeg_params=ffmpeg_params,
    )


def setup_video_writer(
    config: VideoConfig,
    output_dir: Path,
    video_name: str,
) -> VideoFrameSink | None:
    """
    Create and return a writer-like sink or None when disabled.

    The sink supports both realtime streaming (imageio Writer) and
    postprocess collection that encodes after optimization completes.
    """
    if not config.create_video:
        return None

    output_path = (output_dir / video_name).resolve()

    if config.mode == "postprocess":
        return PostprocessVideoWriter(config, output_path)
    if config.mode != "realtime":
        msg = f"Unsupported video mode: {config.mode}"
        raise ValueError(msg)
    return _open_imageio_writer(config, output_path)


def setup_gif_collector(
    config: VideoConfig,
    output_dir: Path,
    gif_name: str,
) -> VideoFrameSink | None:
    """Return a GIF frame collector when GIF export is enabled."""
    if not config.create_gif:
        return None

    output_path = (output_dir / gif_name).resolve()
    return GifFrameCollector(output_path, config.fps)


def _auto_postprocess_reason(
    config: VideoConfig,
    *,
    frame_size: tuple[int, int],
    total_steps: int,
) -> tuple[str | None, int]:
    """Return the heuristic reason for postprocess mode, if any."""
    if config.save_every <= 0:
        return None, 0

    estimated_frames = total_steps // config.save_every
    if estimated_frames <= 0:
        return None, estimated_frames

    width, height = frame_size
    if width <= 0 or height <= 0:
        return None, estimated_frames

    area = width * height
    megapixels = area / _MEGAPIXEL
    reason: str | None = None

    if estimated_frames >= _AUTO_LONG_RUN_FRAME_THRESHOLD:
        reason = (
            f"estimated {estimated_frames} frames exceeds long-run "
            f"threshold ({_AUTO_LONG_RUN_FRAME_THRESHOLD})"
        )
    elif (
        area >= _AUTO_ULTRA_RES_AREA
        and estimated_frames >= _AUTO_ULTRA_RES_FRAME_THRESHOLD
    ):
        reason = (
            f"4K-class frame ({width}x{height}) with {estimated_frames} frames"
        )
    elif (
        area >= _AUTO_HIGH_RES_AREA
        and estimated_frames >= _AUTO_HIGH_RES_FRAME_THRESHOLD
    ):
        reason = (
            f"high-res {megapixels:.1f}MP frame with {estimated_frames} frames"
        )
    elif (
        config.fps >= _AUTO_HIGH_FPS_THRESHOLD
        and estimated_frames >= _AUTO_HIGH_FPS_FRAME_THRESHOLD
    ):
        reason = (
            f"{config.fps} fps run producing {estimated_frames} frames "
            "while encoding in realtime"
        )
    elif (
        config.save_every <= _AUTO_SAVE_EVERY_THRESHOLD
        and estimated_frames >= _AUTO_SAVE_EVERY_FRAME_THRESHOLD
    ):
        reason = (
            f"--save-every {config.save_every} yields "
            f"{estimated_frames} frames"
        )

    return reason, estimated_frames


def select_video_mode(
    config: VideoConfig,
    *,
    frame_size: tuple[int, int],
    total_steps: int,
) -> tuple[VideoMode, str | None, int]:
    """
    Determine the effective video mode and provide the heuristic reason.

    Returns a tuple of (effective_mode, reason, estimated_frame_count). The
    reason is only populated when the heuristic promotes postprocess mode.
    """
    reason, estimated_frames = _auto_postprocess_reason(
        config,
        frame_size=frame_size,
        total_steps=total_steps,
    )

    if config.mode_override or config.mode == "postprocess":
        return config.mode, None, estimated_frames

    if reason is not None:
        return "postprocess", reason, estimated_frames

    return config.mode, None, estimated_frames
