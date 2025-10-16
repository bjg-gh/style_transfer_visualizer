"""Top-level orchestration for style transfer logic."""

from pathlib import Path

import torch

import style_transfer_visualizer.core_model as stv_core_model
import style_transfer_visualizer.image_io as stv_image_io
import style_transfer_visualizer.optimization as stv_optimizer
import style_transfer_visualizer.runtime as stv_runtime
import style_transfer_visualizer.video as stv_video
from style_transfer_visualizer.config import StyleTransferConfig
from style_transfer_visualizer.type_defs import (
    InputPaths,
    SaveOptions,
)


def style_transfer(
    paths: InputPaths,
    config: StyleTransferConfig,
) -> torch.Tensor:
    """Top level style transfer entry point."""
    # Validate inputs
    stv_runtime.validate_input_paths(paths.content_path, paths.style_path)
    stv_runtime.validate_parameters(config.video.quality)

    # Adjust for final-only mode
    if config.video.final_only:
        config.video.create_video = False
        config.video.save_every = config.optimization.steps + 1

    # Setup environment
    stv_runtime.setup_random_seed(config.optimization.seed)
    device = stv_runtime.setup_device(config.hardware.device)

    # Load and preprocess input images
    content_img = stv_image_io.load_image_to_tensor(
        paths.content_path,
        device,
        normalize=config.optimization.normalize,
    )
    style_img = stv_image_io.load_image_to_tensor(
        paths.style_path,
        device,
        normalize=config.optimization.normalize,
    )

    # Prepare model and optimizer
    model, input_img, optimizer = stv_core_model.prepare_model_and_input(
        content_img,
        style_img,
        device,
        config.optimization,
    )

    # Prepare output paths
    output_path = stv_runtime.setup_output_directory(config.output.output)
    content_path = Path(paths.content_path)
    style_path = Path(paths.style_path)
    content_name = content_path.stem
    style_name = style_path.stem
    video_name = f"timelapse_{content_name}_x_{style_name}.mp4"

    # Initialize video writer (if needed)
    video_writer = stv_video.setup_video_writer(config.video, output_path,
                                                video_name)
    intro_last_frame = None
    intro_crossfade_frames = 0
    if video_writer:
        intro_info = stv_video.prepare_intro_segment(
            config.video,
            video_writer,
            content_path,
            style_path,
        )
        if intro_info is not None:
            intro_last_frame, intro_crossfade_frames = intro_info

    # Run optimization
    runner = stv_optimizer.OptimizationRunner(
        model,
        input_img,
        config,
        optimizer=optimizer,
        video_writer=video_writer,
        intro_last_frame=intro_last_frame,
        intro_crossfade_frames=intro_crossfade_frames,
    )
    input_img, loss_metrics, elapsed = runner.run()

    if video_writer and config.video.final_frame_compare:
        with torch.no_grad():
            final_tensor = stv_image_io.prepare_image_for_output(
                input_img,
                normalize=config.optimization.normalize,
            )
        final_frame = (
            final_tensor.squeeze(0)
            .detach()
            .cpu()
            .mul(255.0)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(1, 2, 0)
            .contiguous()
            .numpy()
        )
        stv_video.append_final_comparison_frame(
            config.video,
            video_writer,
            content_path,
            style_path,
            final_frame,
        )

    # Clean up and save outputs
    if video_writer:
        video_writer.close()

    save_opts = SaveOptions(
        content_name=content_name,
        style_name=style_name,
        video_name=video_name,
        normalize=config.optimization.normalize,
        video_created=config.video.create_video,
        plot_losses=config.output.plot_losses,
    )

    stv_runtime.save_outputs(
        input_img,
        loss_metrics,
        output_path,
        elapsed,
        save_opts,
    )

    return input_img.detach().clamp(0, 1)
