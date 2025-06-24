"""Handles timelapse video writer and output file saving."""

from pathlib import Path
from typing import Optional

import imageio

from style_transfer_visualizer.constants import(
    VIDEO_CODEC,
    ENCODING_BLOCK_SIZE
)

def setup_video_writer(
    output_path: Path,
    video_name: str,
    fps: int,
    video_quality: int,
    create_video: bool
) -> Optional[imageio.plugins.ffmpeg.FfmpegFormat.Writer]:
    """Initialize video writer if requested."""
    if not create_video:
        return None

    return imageio.get_writer(
        output_path / video_name,
        fps=fps,
        codec=VIDEO_CODEC,
        quality=video_quality,
        mode="I",  # Explicitly set mode for clarity
        macro_block_size=ENCODING_BLOCK_SIZE
    )
