"""Optimization step and loop logic for style transfer."""
import time
from typing import Optional, Tuple

import imageio
import torch
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm

from style_transfer_visualizer.logging_utils import logger
from style_transfer_visualizer.types import LossMetrics
import style_transfer_visualizer.image_io as stv_image_io


def optimization_step(
    model: nn.Module,
    input_img: torch.Tensor,
    optimizer: Optimizer,
    style_weight: float,
    content_weight: float,
    loss_metrics: LossMetrics,
    step: int,
    save_every: int,
    video_writer: Optional[imageio.plugins.ffmpeg.FfmpegFormat.Writer],
    normalize: bool,
    progress_bar: tqdm
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

    loss_metrics["style_loss"].append(style_score.item())
    loss_metrics["content_loss"].append(content_score.item())
    loss_metrics["total_loss"].append(loss.item())

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
    video_writer: Optional[imageio.plugins.ffmpeg.FfmpegFormat.Writer]
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

    Returns:
        (final image tensor, loss metrics dictionary, elapsed time)
    """
    loss_metrics = {"style_loss": [], "content_loss": [], "total_loss": []}

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
            progress_bar
        )
        step += 1
        return loss

    while step < steps:
        optimizer.step(closure)

    progress_bar.close()
    elapsed = time.time() - start_time

    return input_img, loss_metrics, elapsed
