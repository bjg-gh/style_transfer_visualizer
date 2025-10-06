"""Optimization step and loop logic for style transfer."""
import time
from dataclasses import dataclass

import imageio
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm

import style_transfer_visualizer.image_io as stv_image_io
import style_transfer_visualizer.video as stv_video
from style_transfer_visualizer.config import StyleTransferConfig
from style_transfer_visualizer.constants import CSV_LOGGING_RECOMMENDED_STEPS
from style_transfer_visualizer.logging_utils import logger
from style_transfer_visualizer.loss_logger import LossCSVLogger
from style_transfer_visualizer.type_defs import LossHistory


@dataclass(slots=True)
class StepContext:
    """Runtime objects shared across optimization steps."""

    config: StyleTransferConfig
    progress_bar: tqdm
    video_writer: imageio.plugins.ffmpeg.FfmpegFormat.Writer | None
    loss_metrics: LossHistory | None
    loss_logger: LossCSVLogger | None = None
    last_loss: float | None = None
    intro_last_frame: np.ndarray | None = None
    intro_crossfade_frames: int = 0
    intro_transition_done: bool = False


def optimization_step(
    model: nn.Module,
    input_img: torch.Tensor,
    optimizer: Optimizer,
    step_idx: int,
    ctx: StepContext,
) -> float:
    """One optimization step and optional logging and video frame write."""
    cfg = ctx.config
    style_weight = cfg.optimization.style_w
    content_weight = cfg.optimization.content_w
    save_every = cfg.video.save_every
    normalize = cfg.optimization.normalize
    progress_bar = ctx.progress_bar
    loss_logger = ctx.loss_logger
    loss_metrics = ctx.loss_metrics
    video_writer = ctx.video_writer

    optimizer.zero_grad()
    style_losses, content_losses = model(input_img)

    device = input_img.device
    dtype = input_img.dtype
    zero = torch.zeros((), device=device, dtype=dtype)
    style_score = (
        torch.stack(style_losses).sum()
        if style_losses
        else zero
    )
    content_score = (
        torch.stack(content_losses).sum()
        if content_losses
        else zero
    )

    loss = style_weight * style_score + content_weight * content_score
    loss.backward()

    if not torch.isfinite(style_score):
        logger.warning("Non-finite style score at step %d", step_idx)
    if not torch.isfinite(content_score):
        logger.warning("Non-finite content score at step %d", step_idx)
    if not torch.isfinite(loss):
        logger.warning(
            "Non-finite total loss at step %d, using previous loss",
            step_idx,
        )

    logger.debug(
        "Step %d: Style %s, Content %.4e, Total %.4e",
        step_idx,
        [s.item() for s in style_losses],
        content_score.item(),
        loss.item(),
    )

    if loss_metrics is not None:
        loss_metrics["style_loss"].append(style_score.item())
        loss_metrics["content_loss"].append(content_score.item())
        loss_metrics["total_loss"].append(loss.item())

    if loss_logger is not None:
        loss_logger.log(
            step_idx,
            style_score.item(),
            content_score.item(),
            loss.item(),
        )

    if save_every and step_idx % save_every == 0:
        with torch.no_grad():
            img = stv_image_io.prepare_image_for_output(
                input_img,
                normalize=normalize,
            )
            if video_writer is not None and img is not None:
                img_np = (
                    img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
                ).astype("uint8")
                if (
                    ctx.intro_last_frame is not None
                    and not ctx.intro_transition_done
                ):
                    stv_video.append_crossfade(
                        video_writer,
                        ctx.intro_last_frame,
                        img_np,
                        ctx.intro_crossfade_frames,
                    )
                    ctx.intro_transition_done = True
                    ctx.intro_last_frame = None
                video_writer.append_data(img_np)
        progress_bar.set_postfix({
            "style": f"{style_score.item():.4f}",
            "content": f"{content_score.item():.4f}",
            "loss": f"{loss.item():.4f}",
        })

    loss_value = loss.item()
    ctx.last_loss = loss_value
    progress_bar.update(1)
    return loss_value


def run_optimization_loop(  # noqa: PLR0913
    model: nn.Module,
    input_img: torch.Tensor,
    optimizer: Optimizer,
    config: StyleTransferConfig,
    video_writer: imageio.plugins.ffmpeg.FfmpegFormat.Writer | None,
    *,
    intro_last_frame: np.ndarray | None = None,
    intro_crossfade_frames: int = 0,
) -> tuple[torch.Tensor, LossHistory, float]:
    """Run the optimization loop for style transfer."""
    steps = config.optimization.steps
    log_loss_path = config.output.log_loss
    log_every = config.output.log_every

    loss_metrics: LossHistory | None = None
    loss_logger: LossCSVLogger | None = None
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

        if steps > CSV_LOGGING_RECOMMENDED_STEPS:
            logger.warning(
                "Long run detected (%d steps). Consider enabling --log-loss "
                "to reduce memory usage.",
                steps,
            )

    # Progress tracking
    step = 0
    progress_bar = tqdm(total=steps, desc="Style Transfer")
    start_time = time.time()
    ctx = StepContext(
        config,
        progress_bar,
        video_writer,
        loss_metrics,
        loss_logger,
        intro_last_frame=intro_last_frame,
        intro_crossfade_frames=intro_crossfade_frames,
    )

    def closure() -> float:
        """Optimization closure for LBFGS."""
        nonlocal step
        if step >= steps:
            last = ctx.last_loss
            return 0.0 if last is None else last
        step_idx = step + 1
        loss = optimization_step(model, input_img, optimizer, step_idx, ctx)
        step += 1
        return loss

    while step < steps:
        optimizer.step(closure)

    progress_bar.close()

    if loss_logger:  # Close logger if used
        loss_logger.close()

    elapsed = time.time() - start_time

    return input_img, loss_metrics or {}, elapsed
