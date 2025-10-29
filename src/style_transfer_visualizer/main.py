"""Top-level orchestration for style transfer logic."""

from pathlib import Path

import torch

import style_transfer_visualizer.core_model as stv_core_model
import style_transfer_visualizer.image_io as stv_image_io
import style_transfer_visualizer.optimization as stv_optimizer
import style_transfer_visualizer.runtime as stv_runtime
import style_transfer_visualizer.video as stv_video
from style_transfer_visualizer.config import StyleTransferConfig, VideoConfig
from style_transfer_visualizer.logging_utils import logger
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
        config.video.create_gif = False
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

    if config.video.create_video:
        height, width = content_img.shape[-2:]
        frame_size = (int(width), int(height))
        effective_mode, reason, frame_estimate = stv_video.select_video_mode(
            config.video,
            frame_size=frame_size,
            total_steps=config.optimization.steps,
        )
        if effective_mode != config.video.mode:
            config.video.mode = effective_mode
        if reason is not None:
            logger.info(
                (
                    "Auto-selected postprocess video mode (%s). "
                    "Estimated frames: %d."
                ),
                reason,
                frame_estimate,
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
    gif_name = f"timelapse_{content_name}_x_{style_name}.gif"

    # Initialize video writer (if needed)
    video_writer = stv_video.setup_video_writer(
        config.video,
        output_path,
        video_name,
    )
    gif_collector = stv_video.setup_gif_collector(
        config.video,
        output_path,
        gif_name,
    )
    gif_segment_options = stv_video.GifSegmentOptions(
        sink=gif_collector,
        include_intro=config.video.gif_include_intro,
        include_outro=config.video.gif_include_outro,
    )
    intro_last_frame = None
    intro_crossfade_frames = 0
    gif_intro_requested = (
        gif_segment_options.sink is not None
        and gif_segment_options.include_intro
    )
    needs_intro = video_writer is not None or gif_intro_requested
    if needs_intro:
        intro_info = stv_video.prepare_intro_segment(
            config.video,
            video_writer,
            (content_path, style_path),
            gif_options=gif_segment_options,
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
        gif_collector=gif_collector,
        intro_last_frame=intro_last_frame,
        intro_crossfade_frames=intro_crossfade_frames,
    )
    input_img, loss_metrics, elapsed = runner.run()

    _maybe_append_final_segments(
        config.video,
        video_writer,
        gif_segment_options,
        content_path,
        style_path,
        input_img,
        normalize=config.optimization.normalize,
    )

    for sink in (video_writer, gif_collector):
        if sink:
            sink.close()

    save_opts = SaveOptions(
        content_name=content_name,
        style_name=style_name,
        video_name=video_name if video_writer else None,
        gif_name=gif_name if gif_collector else None,
        normalize=config.optimization.normalize,
        video_created=video_writer is not None,
        gif_created=gif_collector is not None,
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


def _maybe_append_final_segments(  # noqa: PLR0913
    video_config: VideoConfig,
    video_writer: stv_video.VideoFrameSink | None,
    gif_options: stv_video.GifSegmentOptions | None,
    content_path: Path,
    style_path: Path,
    input_img: torch.Tensor,
    *,
    normalize: bool,
) -> None:
    """
    Append final comparison frames to any active sinks when configured.

    Handles the shared logic for building the final frame tensor, guarding
    against disabled outputs, and dispatching to the video helpers.
    """
    gif_outro_requested = bool(
        gif_options and gif_options.sink and gif_options.include_outro,
    )
    if not video_config.final_frame_compare:
        return
    if video_writer is None and not gif_outro_requested:
        return

    with torch.no_grad():
        final_tensor = stv_image_io.prepare_image_for_output(
            input_img,
            normalize=normalize,
        )

    if final_tensor is None:
        return

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

    append_kwargs: dict[str, stv_video.GifSegmentOptions] = {}
    if gif_options is not None and gif_options.sink is not None:
        append_kwargs["gif_options"] = gif_options

    stv_video.append_final_comparison_frame(
        video_config,
        video_writer,
        (content_path, style_path),
        final_frame,
        **append_kwargs,
    )
