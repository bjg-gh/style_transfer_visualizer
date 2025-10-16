"""Helpers for managing output locations and persisted artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from torchvision import utils as tv_utils

import style_transfer_visualizer.image_io as stv_image_io
from style_transfer_visualizer.logging_utils import logger

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import torch

    from style_transfer_visualizer.type_defs import LossHistory, SaveOptions


def setup_output_directory(
    output_path: str,
    path_factory: Callable[[str], Path] = Path,
) -> Path:
    """
    Create the output directory if needed and return its resolved path.

    Falls back to ``style_transfer_output`` on failure to create the desired
    directory to keep the run from aborting.
    """
    resolved_path = path_factory(output_path)
    try:
        resolved_path.mkdir(parents=True, exist_ok=True)
    except OSError:
        fallback_path = path_factory("style_transfer_output")
        fallback_path.mkdir(parents=True, exist_ok=True)
        return fallback_path
    return resolved_path


def _canonical_stem(path: Path) -> str:
    """Return a filesystem-safe stem (spaces mapped to underscores)."""
    return path.stem.replace(" ", "_")


def stylized_image_path_from_names(
    output_dir: Path,
    content_name: str,
    style_name: str,
) -> Path:
    """Return the canonical stylized image path for content/style names."""
    return output_dir / f"stylized_{content_name}_x_{style_name}.png"


def stylized_image_path_from_paths(
    output_dir: Path,
    content_path: Path,
    style_path: Path,
) -> Path:
    """Return the stylized image path using stems from the provided files."""
    return stylized_image_path_from_names(
        output_dir=output_dir,
        content_name=_canonical_stem(content_path),
        style_name=_canonical_stem(style_path),
    )


def save_outputs(
    input_img: torch.Tensor,
    loss_metrics: LossHistory,
    output_dir: Path,
    elapsed: float,
    opts: SaveOptions,
) -> None:
    """
    Persist final artifacts from a style transfer run.

    Handles output directory creation, image saving, optional plot generation,
    and final logging so call sites remain focused on orchestration logic.
    """
    try:
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created output directory: %s", output_dir)
    except (PermissionError, OSError) as exc:
        logger.error("Failed to create output directory: %s", exc)
        fallback_dir = Path("style_transfer_output")
        fallback_dir.mkdir(exist_ok=True)
        logger.info("Using fallback directory: %s", fallback_dir)
        output_dir = fallback_dir

    final_path = stylized_image_path_from_names(
        output_dir=output_dir,
        content_name=opts.content_name,
        style_name=opts.style_name,
    )
    image_to_save = stv_image_io.prepare_image_for_output(
        input_img,
        normalize=opts.normalize,
    )
    tv_utils.save_image(image_to_save, final_path)

    if opts.video_created and opts.video_name:
        logger.info("Video saved to: %s", output_dir / opts.video_name)

    if opts.plot_losses:
        from style_transfer_visualizer.visualization.metrics import (  # noqa: PLC0415
            plot_loss_curves,
        )

        plot_loss_curves(loss_metrics, output_dir)

    logger.info("Style transfer completed in %.2f seconds", elapsed)
    logger.info("Final stylized image saved to: %s", final_path)
