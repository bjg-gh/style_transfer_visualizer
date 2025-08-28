"""Handles timelapse video writer and output file saving."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import imageio

from style_transfer_visualizer.constants import (
    ENCODING_BLOCK_SIZE,
    VIDEO_CODEC,
)
from style_transfer_visualizer.utils import resolve_project_version

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
