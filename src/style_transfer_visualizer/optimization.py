"""Optimization step and loop logic for style transfer."""
import time
from typing import Optional, Tuple

import imageio
import torch
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm

from style_transfer_visualizer.logging_utils import logger
from style_transfer_visualizer.loss_logger import LossCSVLogger
from style_transfer_visualizer.types import LossMetrics
import style_transfer_visualizer.image_io as stv_image_io
from style_transfer_visualizer.config_defaults import DEFAULT_LOG_EVERY


def optimization_step(
    model: nn.Module,
    input_img: torch.Tensor,
    optimizer: Optimizer,
    style_weight: float,
    content_weight: float,
    loss_metrics: Optional[LossMetrics],
    step: int,
    save_every: int,
    video_writer: Optional[imageio.plugins.ffmpeg.FfmpegFormat.Writer],
    normalize: bool,
    progress_bar: tqdm,
    loss_logger: Optional[LossCSVLogger] = None
) -> float:
    """Perform a single optimization step.

    Args:
        model: The style transfer model
        input_img: The input image tensor
        optimizer: The optimizer
        style_weight: Weight for style loss
        content_weight: Weight for content loss
        loss_metrics: Dictionary to track loss metrics
        step: Current step number
        save_every: Save frame every N steps
        video_writer: Video writer object (if video creation is enabled)
        normalize: Whether to use ImageNet normalization
        progress_bar: Progress bar object
        loss_logger: Logger for writing loss metrics to CSV.

    Returns:
        float: The current loss value
    """
    optimizer.zero_grad()
    style_losses, content_losses = model(input_img)
    style_score = sum(style_losses)
    content_score = sum(content_losses)
    loss = style_weight * style_score + content_weight * content_score
    loss.backward()

    if not torch.isfinite(style_score):
        logger.warning("Non-finite style score at step %d", step)
    if not torch.isfinite(content_score):
        logger.warning("Non-finite content score at step %d", step)
    if not torch.isfinite(loss):
        logger.warning(
            "Non-finite total loss at step %d, using previous loss", step)

    logger.debug("Step %d: Style %s, Content %.4e, Total %.4e",
                 step, [s.item() for s in style_losses], content_score.item(),
                 loss.item())

    if loss_metrics:  # In-memory loss tracking
        loss_metrics["style_loss"].append(style_score.item())
        loss_metrics["content_loss"].append(content_score.item())
        loss_metrics["total_loss"].append(loss.item())

    if loss_logger:
        loss_logger.log(step, style_score.item(), content_score.item(),
                        loss.item())

    if step % save_every == 0:
        with torch.no_grad():
            img = stv_image_io.prepare_image_for_output(input_img, normalize)
            if img is not None and video_writer is not None:
                img_np = (img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                          * 255).astype("uint8")
                video_writer.append_data(img_np)
        progress_bar.set_postfix({
            "style": f"{style_score.item():.4f}",
            "content": f"{content_score.item():.4f}",
            "loss": f"{loss.item():.4f}"
        })

    progress_bar.update(1)
    return loss.item()


def run_optimization_loop(
    model: nn.Module,
    input_img: torch.Tensor,
    optimizer: Optimizer,
    steps: int,
    save_every: int,
    style_weight: float,
    content_weight: float,
    normalize: bool,
    video_writer: Optional[imageio.plugins.ffmpeg.FfmpegFormat.Writer],
    log_loss_path: Optional[str] = None,
    log_every: int = DEFAULT_LOG_EVERY
) -> Tuple[torch.Tensor, LossMetrics, float]:
    """Run the optimization loop for style transfer.

    Args:
        model: The style transfer model
        input_img: The input image tensor
        optimizer: The optimizer
        steps: Number of optimization steps
        save_every: Save frame every N steps
        style_weight: Weight for style loss
        content_weight: Weight for content loss
        normalize: Whether to use ImageNet normalization
        video_writer: Video writer object (if video creation is enabled)
        log_loss_path: Path to CSV file for logging loss metrics.
        log_every: Log losses every N steps.

    Returns:
        (final image tensor, loss metrics dictionary, elapsed time)
    """
    loss_metrics: Optional[LossMetrics] = None
    loss_logger: Optional[LossCSVLogger] = None
    if log_loss_path:  # Log loss in csv
        try:
            loss_logger = LossCSVLogger(log_loss_path, log_every)
            logger.info("Loss CSV logging enabled: %s", log_loss_path)
        except OSError as e:
            logger.error("Failed to initialize CSV logging: %s", e)
            loss_logger = None  # Fall back gracefully

    else:  # Log loss in memory
        loss_metrics = {"style_loss": [], "content_loss": [],
                        "total_loss": []}

        if steps > 2000:
            logger.warning(
                "Long run detected (%d steps). Consider enabling --log-loss "
                "to reduce memory usage.",
                steps
            )

    # Progress tracking
    step = 0
    progress_bar = tqdm(total=steps, desc="Style Transfer")
    start_time = time.time()

    def closure() -> float:
        """Optimization closure for LBFGS"""
        nonlocal step
        loss = optimization_step(
            model, input_img, optimizer, style_weight, content_weight,
            loss_metrics, step, save_every, video_writer, normalize,
            progress_bar, loss_logger
        )
        step += 1
        return loss

    while step < steps:
        optimizer.step(closure)

    progress_bar.close()

    if loss_logger:  # Close logger if used
        loss_logger.close()

    elapsed = time.time() - start_time

    return input_img, loss_metrics or {}, elapsed
