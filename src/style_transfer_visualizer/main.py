"""Top-level orchestration for style transfer logic."""

from pathlib import Path
from typing import cast

import torch

import style_transfer_visualizer.core_model as stv_core_model
import style_transfer_visualizer.image_io as stv_image_io
import style_transfer_visualizer.optimization as stv_optimizer
import style_transfer_visualizer.utils as stv_utils
import style_transfer_visualizer.video as stv_video
from style_transfer_visualizer.config_defaults import (
    DEFAULT_CONTENT_LAYERS,
    DEFAULT_CONTENT_WEIGHT,
    DEFAULT_CREATE_VIDEO,
    DEFAULT_DEVICE,
    DEFAULT_FINAL_ONLY,
    DEFAULT_FPS,
    DEFAULT_INIT_METHOD,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOG_EVERY,
    DEFAULT_NORMALIZE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SAVE_EVERY,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_STYLE_LAYERS,
    DEFAULT_STYLE_WEIGHT,
    DEFAULT_VIDEO_QUALITY,
)
from style_transfer_visualizer.type_defs import InitMethod


def style_transfer(  # noqa: PLR0913
    content_path: str,
    style_path: str,
    *,
    steps: int = DEFAULT_STEPS,
    style_weight: float = DEFAULT_STYLE_WEIGHT,
    content_weight: float = DEFAULT_CONTENT_WEIGHT,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    style_layers: list[int] = DEFAULT_STYLE_LAYERS,
    content_layers: list[int] = DEFAULT_CONTENT_LAYERS,
    device_name: str = DEFAULT_DEVICE,
    init_method: str = DEFAULT_INIT_METHOD,
    normalize: bool = DEFAULT_NORMALIZE,
    seed: int = DEFAULT_SEED,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    final_only: bool = DEFAULT_FINAL_ONLY,
    plot_losses: bool = True,
    log_loss_path: str | None = None,
    log_every: int = DEFAULT_LOG_EVERY,
    create_video: bool = DEFAULT_CREATE_VIDEO,
    save_every: int = DEFAULT_SAVE_EVERY,
    fps: int = DEFAULT_FPS,
    video_quality: int = DEFAULT_VIDEO_QUALITY,
) -> torch.Tensor:
    """
    Run full neural style transfer pipeline on a pair of input images.

    Combines image loading, model setup, optimization, and optional
    video generation to produce a stylized version of the content image.

    Args:
        content_path: Path to the content image.
        style_path: Path to the style image.
        steps: Number of optimization steps.
        style_weight: Weight applied to the style loss.
        content_weight: Weight applied to the content loss.
        learning_rate: Learning rate used by the optimizer.
        style_layers: VGG layer indices to use for style features.
        content_layers: VGG layer indices to use for content features.
        device_name: Device to run on ("cuda" or "cpu").
        init_method: Initialization method for the input image.
        normalize: Whether to apply ImageNet normalization.
        seed: Seed for random number generation.
        output_dir: Directory to save output images and video.
        final_only: If True, only save the final image.
        plot_losses: If True, plot loss curves with matplotlib.
        log_loss_path: If set, path to write a CSV loss log.
        log_every: Log loss every N steps.
        create_video: If True, generate timelapse video of the transfer.
        save_every: Save a frame every N steps during optimization.
        fps: Frames per second for the timelapse video.
        video_quality: Quality of the output video (1-10 scale).

    Returns:
        Final stylized image as a torch tensor.

    Note:
        Input images must be pre-sized by the user. Minimum size: 64px;
        performance may degrade above ~3000px resolution.

    """
    # Validate inputs
    stv_utils.validate_input_paths(content_path, style_path)
    stv_utils.validate_parameters(video_quality)

    # Adjust for final-only mode
    if final_only:
        create_video = False
        save_every = steps + 1

    # Setup environment
    stv_utils.setup_random_seed(seed)
    device = stv_utils.setup_device(device_name)

    # Load and preprocess input images
    content_img = stv_image_io.load_image_to_tensor(content_path, device,
                                       normalize=normalize)
    style_img = stv_image_io.load_image_to_tensor(style_path, device,
                                                  normalize=normalize)

    # Prepare model and optimizer
    model, input_img, optimizer = stv_core_model.prepare_model_and_input(
        content_img, style_img, device, cast(InitMethod, init_method), # noqa: TC006
        learning_rate, style_layers, content_layers,
    )

    # Prepare output paths
    output_path = stv_utils.setup_output_directory(output_dir)
    content_name = Path(content_path).stem
    style_name = Path(style_path).stem
    video_name = f"timelapse_{content_name}_x_{style_name}.mp4"

    # Initialize video writer (if needed)
    video_writer = stv_video.setup_video_writer(
        output_path, video_name, fps, video_quality, create_video=create_video,
    )

    # Run optimization
    input_img, loss_metrics, elapsed = stv_optimizer.run_optimization_loop(
        model, input_img, optimizer,
        steps, save_every, style_weight,
        content_weight, normalize=normalize, video_writer=video_writer,
        log_loss_path=log_loss_path, log_every=log_every,
    )

    # Clean up and save outputs
    if video_writer:
        video_writer.close()

    stv_utils.save_outputs(
        input_img=input_img,
        loss_metrics=loss_metrics,
        output_dir=output_path,
        elapsed=elapsed,
        content_name=content_name,
        style_name=style_name,
        video_name=video_name,
        normalize=normalize,
        video_created=create_video,
        plot_losses=plot_losses,
    )

    return input_img.detach().clamp(0, 1)
