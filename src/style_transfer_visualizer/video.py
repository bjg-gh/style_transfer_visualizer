"""Handles timelapse video writer and output file saving."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import UTC, datetime
from typing import TYPE_CHECKING

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
    from pathlib import Path

    from style_transfer_visualizer.config import VideoConfig


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
_FRAME_NDIMS = 3
_RGB_CHANNELS = 3
_SIZE_TUPLE_LEN = 2


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
    writer: imageio.plugins.ffmpeg.FfmpegFormat.Writer,
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
    writer: imageio.plugins.ffmpeg.FfmpegFormat.Writer | None,
    content_path: Path,
    style_path: Path,
) -> tuple[np.ndarray, int] | None:
    """Render intro sequence, return final intro frame and crossfade length."""
    if writer is None or not config.create_video or not config.intro_enabled:
        return None

    intro_frame = _build_intro_frame(content_path, style_path)
    fade_frames_raw = round(config.fps * INTRO_FADE_IN_SECONDS)
    fade_frames = max(1, min(fade_frames_raw, INTRO_MAX_FADE_FRAMES))
    hold_frames_raw = round(config.fps * config.intro_duration_seconds)
    hold_frames = max(0, hold_frames_raw)

    black = np.zeros_like(intro_frame)
    _append_fade_transition(writer, black, intro_frame, fade_frames)
    for _ in range(hold_frames):
        writer.append_data(intro_frame)

    crossfade_raw = round(config.fps * INTRO_CROSSFADE_SECONDS)
    crossfade_frames = max(1, min(crossfade_raw, INTRO_MAX_CROSSFADE_FRAMES))
    return intro_frame, crossfade_frames


def append_crossfade(
    writer: imageio.plugins.ffmpeg.FfmpegFormat.Writer,
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
    writer: imageio.plugins.ffmpeg.FfmpegFormat.Writer,
    last_frame: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Return the resized last frame and writer-aligned dimensions."""
    if last_frame.ndim != _FRAME_NDIMS or last_frame.shape[2] != _RGB_CHANNELS:
        msg = "Last timelapse frame must be an RGB array"
        raise ValueError(msg)

    last_rgb = np.asarray(last_frame, dtype=np.uint8)
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
    writer: imageio.plugins.ffmpeg.FfmpegFormat.Writer | None,
    content_path: Path,
    style_path: Path,
    last_frame: np.ndarray,
) -> None:
    """
    Append a final comparison frame showing content, style, and result.

    Uses the gallery wall layout with labels to mirror the standalone
    comparison output. Includes a short crossfade from the most recent
    timelapse frame followed by a configurable hold duration. No-op when
    the feature is disabled.
    """
    if (
        writer is None
        or not config.create_video
        or not config.final_frame_compare
    ):
        return
    last_rgb, target_width, target_height = _resolve_writer_dimensions(
        writer,
        last_frame,
    )
    result_image = Image.fromarray(np.asarray(last_frame, dtype=np.uint8))
    frame_params = FrameParams(frame_tone="gold", label="on")
    frame_np = _build_outro_frame(
        (content_path, style_path),
        result_image,
        frame_params,
        target_width=target_width,
        target_height=target_height,
    )

    crossfade_raw = round(config.fps * OUTRO_CROSSFADE_SECONDS)
    crossfade_frames = max(1, min(crossfade_raw, OUTRO_MAX_CROSSFADE_FRAMES))
    append_crossfade(
        writer,
        last_rgb,
        frame_np,
        crossfade_frames,
        max_frames=OUTRO_MAX_CROSSFADE_FRAMES,
    )

    outro_seconds = max(0.0, config.outro_duration_seconds)
    hold_frames_raw = round(config.fps * outro_seconds)
    hold_frames = max(FINAL_COMPARISON_MIN_FRAMES, hold_frames_raw)
    for _ in range(hold_frames):
        writer.append_data(frame_np)


def setup_video_writer(
    config: VideoConfig,
    output_dir: Path,
    video_name: str,
) -> imageio.plugins.ffmpeg.FfmpegFormat.Writer | None:
    """
    Create and return an imageio writer or None when disabled.

    Requires cfg.create_video, cfg.fps, cfg.quality, cfg.save_every.
    Optionally uses cfg.metadata_title and cfg.metadata_artist for tags.
    """
    if not config.create_video:
        return None

    output_path = (output_dir / video_name).resolve()
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
