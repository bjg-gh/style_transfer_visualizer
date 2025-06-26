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


def compute_total_loss(
    model: nn.Module,
    input_img: torch.Tensor,
    style_weight: float,
    content_weight: float
) -> Tuple[torch.Tensor, float, float]:
    style_losses, content_losses = model(input_img)
    style_score = sum(style_losses)
    content_score = sum(content_losses)
    loss = style_weight * style_score + content_weight * content_score
    return loss, style_score.item(), content_score.item()


def log_and_track(
    step: int,
    loss: float,
    style_score: float,
    content_score: float,
    loss_metrics: LossMetrics,
    progress_bar
) -> None:
    logger.debug("Step %d: Style %.4e, Content %.4e, Total %.4e",
                 step, style_score, content_score, loss)
    loss_metrics["style_loss"].append(style_score)
    loss_metrics["content_loss"].append(content_score)
    loss_metrics["total_loss"].append(loss)
    progress_bar.set_postfix({
        "style": f"{style_score:.4f}",
        "content": f"{content_score:.4f}",
        "loss": f"{loss:.4f}"
    })
    progress_bar.update(1)


def maybe_save_frame(input_img, step, save_every, video_writer, normalize):
    if step % save_every == 0 and video_writer:
        with torch.no_grad():
            img = stv_image_io.prepare_image_for_output(input_img, normalize)
            img_np = (img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            video_writer.append_data(img_np)


def run_second_order_optimization_loop(
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
    loss_metrics = {"style_loss": [], "content_loss": [], "total_loss": []}
    progress_bar = tqdm(total=steps, desc="Style Transfer")
    step = 0
    start_time = time.time()

    def closure():
        nonlocal step
        optimizer.zero_grad()
        loss, style_score, content_score = compute_total_loss(model, input_img, style_weight, content_weight)
        loss.backward()
        log_and_track(step, loss.item(), style_score, content_score, loss_metrics, progress_bar)
        maybe_save_frame(input_img, step, save_every, video_writer, normalize)
        step += 1
        return loss.item()

    while step < steps:
        optimizer.step(closure)

    progress_bar.close()
    elapsed = time.time() - start_time
    return input_img, loss_metrics, elapsed  # second-order optimizer


def run_first_order_optimization_loop(
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
    loss_metrics = {"style_loss": [], "content_loss": [], "total_loss": []}
    progress_bar = tqdm(total=steps, desc="Style Transfer")
    start_time = time.time()

    for step in range(steps):
        optimizer.zero_grad()
        loss, style_score, content_score = compute_total_loss(model, input_img, style_weight, content_weight)
        loss.backward()
        optimizer.step()
        log_and_track(step, loss.item(), style_score, content_score, loss_metrics, progress_bar)
        maybe_save_frame(input_img, step, save_every, video_writer, normalize)

    progress_bar.close()
    elapsed = time.time() - start_time
    return input_img, loss_metrics, elapsed
