"""Top-level orchestration for style transfer logic."""

from pathlib import Path

import torch

from style_transfer_visualizer.config_defaults import (
    DEFAULT_OUTPUT_DIR, DEFAULT_STEPS, DEFAULT_SAVE_EVERY,
    DEFAULT_STYLE_WEIGHT, DEFAULT_CONTENT_WEIGHT,
    DEFAULT_INIT_METHOD, DEFAULT_SEED, DEFAULT_NORMALIZE, DEFAULT_FPS,
    DEFAULT_VIDEO_QUALITY, DEFAULT_CREATE_VIDEO, DEFAULT_FINAL_ONLY,
    DEFAULT_DEVICE, DEFAULT_OPTIMIZER, DEFAULT_LBFGS_LR
)
import style_transfer_visualizer.core_model as stv_core_model
import style_transfer_visualizer.image_io as stv_image_io
import style_transfer_visualizer.optimization as stv_optimizer
import style_transfer_visualizer.utils as stv_utils
import style_transfer_visualizer.video as stv_video


def style_transfer(
    content_path: str,
    style_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    steps: int = DEFAULT_STEPS,
    save_every: int = DEFAULT_SAVE_EVERY,
    style_weight: float = DEFAULT_STYLE_WEIGHT,
    content_weight: float = DEFAULT_CONTENT_WEIGHT,
    optimizer_name: str = DEFAULT_OPTIMIZER,
    learning_rate: float = DEFAULT_LBFGS_LR,
    fps: int = DEFAULT_FPS,
    device_name: str = DEFAULT_DEVICE,
    init_method: str = DEFAULT_INIT_METHOD,
    normalize: bool = DEFAULT_NORMALIZE,
    create_video: bool = DEFAULT_CREATE_VIDEO,
    final_only: bool = DEFAULT_FINAL_ONLY,
    video_quality: int = DEFAULT_VIDEO_QUALITY,
    seed: int = DEFAULT_SEED
) -> torch.Tensor:
    """
    Orchestrates the full style transfer pipeline.

    Args:
        content_path: Path to the content image
        style_path: Path to the style image
        output_dir: Directory to save outputs
        steps: Number of optimization steps
        save_every: Save frame every N steps
        style_weight: Weight for style loss
        content_weight: Weight for content loss
        optimizer_name: Optimizer name (lbfgs or adam)
        learning_rate: Learning rate for optimizer
        fps: Frames per second for timelapse video
        device_name: Device to run on ("cuda" or "cpu")
        init_method: Method to initialize the input image
        normalize: Whether to use ImageNet normalization
        create_video: Whether to create a timelapse video
        final_only: Whether to only save the final image
        video_quality: Quality setting for output video (1-10)
        seed: Random seed for reproducibility

    Returns:
        The final stylized image tensor

    Note:
        Input images must be pre-sized by the user. Minimum size: 64px;
        processing may be slow above 3000px.
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
        content_img, style_img, device, init_method, optimizer_name,
        learning_rate
    )

    # Prepare output paths
    output_path = stv_utils.setup_output_directory(output_dir)
    content_name = Path(content_path).stem
    style_name = Path(style_path).stem
    video_name = f"timelapse_{content_name}_x_{style_name}.mp4"

    # Initialize video writer (if needed)
    video_writer = stv_video.setup_video_writer(
        output_path, video_name, fps, video_quality, create_video
    )

    # Run optimization
    if isinstance(optimizer, torch.optim.LBFGS):
        input_img, loss_metrics, elapsed = stv_optimizer.run_second_order_optimization_loop(
            model, input_img, optimizer, steps, save_every,
            style_weight, content_weight, normalize, video_writer
        )
    else:
        input_img, loss_metrics, elapsed = stv_optimizer.run_first_order_optimization_loop(
            model, input_img, optimizer, steps, save_every,
            style_weight, content_weight, normalize, video_writer
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
        video_created=create_video
    )

    return input_img.detach().clamp(0, 1)