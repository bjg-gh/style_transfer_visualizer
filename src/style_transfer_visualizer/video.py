"""Handles timelapse video writer and output file saving."""

from pathlib import Path

import imageio

from style_transfer_visualizer.config import VideoConfig
from style_transfer_visualizer.constants import (
    ENCODING_BLOCK_SIZE,
    VIDEO_CODEC,
)


def setup_video_writer(
    config: VideoConfig,
    output_path: Path,
    video_name: str,
) -> imageio.plugins.ffmpeg.FfmpegFormat.Writer | None:
    """
    Initialize a timelapse video writer if requested.

    Args:
        config: Validated video configuration.
        output_path: Directory where the video will be written.
        video_name: Filename for the output video.

    Returns:
        An imageio writer or None if video creation is disabled.

    """
    if not config.create_video:
        return None

    return imageio.get_writer(
        output_path / video_name,
        fps=config.fps,
        codec=VIDEO_CODEC,
        quality=config.quality,
        mode="I",
        macro_block_size=ENCODING_BLOCK_SIZE,
    )
