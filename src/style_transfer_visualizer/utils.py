"""Utility functions: logging, seeding, validation, device setup, output path."""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torchvision.utils import save_image

from style_transfer_visualizer.logging_utils import logger
from style_transfer_visualizer.types import LossMetrics
import style_transfer_visualizer.image_io as stv_image_io


def setup_device(device_name: str) -> torch.device:
    """Set up the device for computation.

    Args:
        device_name: Device to run on ("cuda" or "cpu")

    Returns:
        torch.device: The device to use
    """
    if device_name == "cuda" and not torch.cuda.is_available():
        logger.warning(
            "CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    logger.info("Using device: %s", device)
    return device


def setup_random_seed(seed: int) -> None:
    """Sets random seeds for all libraries.

    Ensures deterministic behavior by setting seeds for PyTorch (both
    CPU and CUDA), Python's random module, and NumPy. This is important
    for scientific reproducibility and debugging.

    Args:
        seed: Random seed value used for all random number generators
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)  # Python's seed
    np.random.seed(seed)


def validate_input_paths(content_path: str, style_path: str) -> None:
    """Validate that the content and style image paths exist."""
    if not Path(content_path).is_file():
        raise FileNotFoundError(f"Content image not found: {content_path}")
    if not Path(style_path).is_file():
        raise FileNotFoundError(f"Style image not found: {style_path}")


def validate_parameters(video_quality: int) -> None:
    """Validates that parameters fall within acceptable ranges.

        Currently only checks video quality, but can be expanded to validate
        other parameters as the function signature suggests.
    """
    if video_quality < 1 or video_quality > 10:
        raise ValueError(
            f"Video quality must be between 1 and 10, got {video_quality}")


def setup_output_directory(output_path: str, path_factory=Path) -> Path:
    """Create and return the output directory path."""
    output_path = path_factory(output_path)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception:
        fallback_path = path_factory("style_transfer_output")
        fallback_path.mkdir(parents=True, exist_ok=True)
        return fallback_path
    return output_path


def plot_loss_curves(metrics: LossMetrics, output_dir: Path) -> None:
    """Save a matplotlib plot of training loss curves."""
    if not metrics:
        logger.warning("No loss metrics dictionary provided.")
        return

    if not any(len(values) > 0 for values in metrics.values()):
        logger.warning("Loss metrics dictionary is empty, nothing to plot.")
        return

    try: # import is here because this functionality is optional
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ImportError:
        logger.warning("matplotlib not found: skipping loss plot.")
        return

    fig = plt.figure(figsize=(10, 6))
    try:
        for k in metrics:
            if metrics[k]:
                plt.plot(metrics[k], label=k)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Loss Curves")
        plt.legend()
        plt.tight_layout()
        loss_plot_path = output_dir / "loss_plot.png"
        plt.savefig(loss_plot_path)
        logger.info("Loss plot saved to: %s", loss_plot_path)
    finally:
        plt.close(fig)  # Ensure figure is closed to prevent memory leaks


def save_outputs(
    input_img: torch.Tensor,
    loss_metrics: LossMetrics,
    output_dir: Path,
    elapsed: float,
    content_name: str,
    style_name: str,
    video_name: Optional[str] = None,
    normalize: bool = True,
    video_created: bool = True,
    plot_losses: bool = True
) -> None:
    """Save final stylized image, optional video, and loss plot."""
    # Ensure output directory exists
    try:
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created output directory: %s", output_dir)
    except (PermissionError, OSError) as e:
        logger.error("Failed to create output directory: %s", e)
        # Create a fallback directory in the current working directory
        fallback_dir = Path("style_transfer_output")
        fallback_dir.mkdir(exist_ok=True)
        logger.info("Using fallback directory: %s", fallback_dir)
        output_dir = fallback_dir

    # Save the final stylized image
    final_path = output_dir / f"stylized_{content_name}_x_{style_name}.png"
    img_to_save = stv_image_io.prepare_image_for_output(input_img, normalize)
    save_image(img_to_save, final_path)

    # Log video information
    if video_created and video_name:
        logger.info("Video saved to: %s", output_dir / video_name)

    # Create and save loss plot
    if plot_losses:
        plot_loss_curves(loss_metrics, output_dir)

    # Log completion information
    logger.info("Style transfer completed in %.2f seconds", elapsed)
    logger.info("Final stylized image saved to: %s", final_path)
