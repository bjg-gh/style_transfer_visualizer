"""Optimization orchestration for style transfer."""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping  # noqa: TC003
from dataclasses import dataclass
from typing import Protocol

import numpy as np  # noqa: TC002
import torch
from torch import nn
from torch.optim import Optimizer  # noqa: TC002
from tqdm import tqdm

import style_transfer_visualizer.image_io as stv_image_io
import style_transfer_visualizer.video as stv_video
from style_transfer_visualizer.config import (
    StyleTransferConfig,  # noqa: TC001
)
from style_transfer_visualizer.constants import CSV_LOGGING_RECOMMENDED_STEPS
from style_transfer_visualizer.logging_utils import logger
from style_transfer_visualizer.loss_logger import LossCSVLogger
from style_transfer_visualizer.type_defs import LossHistory  # noqa: TC001


class ProgressReporter(Protocol):
    """Protocol capturing the subset of tqdm's interface we rely on."""

    def update(self, n: float | None = 1) -> bool | None:
        """Advance the progress display by ``n`` units."""

    def set_postfix(
        self,
        ordered_dict: Mapping[str, object] | None = None,
        refresh: bool | None = True,  # noqa: FBT001,FBT002
        **kwargs: object,
    ) -> None:
        """Update the supplementary values shown beside the progress bar."""

    def close(self) -> None:
        """Release any resources associated with the display."""


@dataclass(slots=True)
class StepMetrics:
    """Data recorded at the end of each optimization step."""

    step: int
    style_loss: float
    content_loss: float
    total_loss: float


@dataclass(slots=True)
class OptimizationCallbacks:
    """Optional hooks invoked around optimization events."""

    on_step_start: Callable[[int], None] | None = None
    on_step_end: Callable[[StepMetrics], None] | None = None
    on_video_frame: Callable[[np.ndarray, int], None] | None = None
    on_logging_error: Callable[[Exception], None] | None = None


class OptimizationRunner:
    """
    Encapsulate the optimization loop, logging, and progress reporting.

    Instances coordinate execution of the forward/backward pass,
    optional CSV logging, timelapse frame generation, and any custom
    callbacks interested in step progress.
    """

    def __init__(  # noqa: PLR0913
        self,
        model: nn.Module,
        input_img: torch.Tensor,
        config: StyleTransferConfig,
        *,
        optimizer: Optimizer | None = None,
        optimizer_factory: Callable[[torch.Tensor], Optimizer] | None = None,
        progress_bar: ProgressReporter | None = None,
        callbacks: OptimizationCallbacks | None = None,
        video_writer: stv_video.VideoFrameSink | None = None,
        gif_collector: stv_video.VideoFrameSink | None = None,
        intro_last_frame: np.ndarray | None = None,
        intro_crossfade_frames: int = 0,
    ) -> None:
        if optimizer is not None and optimizer_factory is not None:
            msg = "Provide either optimizer or optimizer_factory, not both."
            raise ValueError(msg)

        self.model = model
        self.input_img = input_img
        self.config = config

        self.optimizer = (
            optimizer
            if optimizer is not None
            else self._build_optimizer(optimizer_factory)
        )

        self._progress_bar: ProgressReporter | None = progress_bar
        self._owns_progress_bar = False

        self.callbacks = callbacks or OptimizationCallbacks()

        self.video_writer = video_writer
        self.gif_collector = gif_collector
        self.intro_last_frame = intro_last_frame
        self.intro_crossfade_frames = intro_crossfade_frames
        self.intro_transition_done = intro_last_frame is None

        self.loss_logger: LossCSVLogger | None = None
        self.loss_metrics: LossHistory | None = None
        self._configure_logging()

        self._step_index = 0
        self.last_loss: float | None = None
        self._active_step_idx: int | None = None
        self._pending_step_metrics: StepMetrics | None = None
        self._closure_calls = 0

    @property
    def progress_bar(self) -> ProgressReporter:
        """Return the active progress reporter."""
        if self._progress_bar is None:
            msg = "Progress bar not initialized. Call run() before use."
            raise RuntimeError(msg)
        return self._progress_bar

    @property
    def total_steps(self) -> int:
        """Total optimization steps configured for this run."""
        return self.config.optimization.steps

    def run(self) -> tuple[torch.Tensor, LossHistory, float]:
        """Execute the optimization loop and return metrics."""
        self._ensure_progress_bar()

        start_time = time.time()

        try:
            while self._step_index < self.total_steps:
                step_idx = self._step_index + 1
                self._emit_step_start(step_idx)
                self._active_step_idx = step_idx
                self._pending_step_metrics = None
                try:
                    self.optimizer.step(self._closure)  # type: ignore[arg-type]
                finally:
                    self._active_step_idx = None

                metrics = self._pending_step_metrics
                if metrics is None:
                    msg = (
                        "Optimizer closure did not record metrics "
                        f"for step {step_idx}"
                    )
                    raise RuntimeError(msg)

                self._finalize_step(metrics)
                self._pending_step_metrics = None
        finally:
            self._cleanup()

        elapsed = time.time() - start_time
        self._log_optimization_summary()
        return self.input_img, self.loss_metrics or {}, elapsed

    def _build_optimizer(
        self,
        optimizer_factory: Callable[[torch.Tensor], Optimizer] | None,
    ) -> Optimizer:
        """Create an optimizer when one is not supplied."""
        if optimizer_factory is not None:
            return optimizer_factory(self.input_img)
        opt_cfg = self.config.optimization
        return torch.optim.LBFGS(
            [self.input_img],
            lr=opt_cfg.lr,
            max_iter=opt_cfg.lbfgs_max_iter,
            max_eval=opt_cfg.lbfgs_max_eval,
        )

    def _configure_logging(self) -> None:
        """Configure loss tracking (CSV logger or in-memory metrics)."""
        log_loss_path = self.config.output.log_loss
        log_every = self.config.output.log_every
        steps = self.total_steps

        if log_loss_path:
            try:
                self.loss_logger = LossCSVLogger(log_loss_path, log_every)
                logger.info("Loss CSV logging enabled: %s", log_loss_path)
            except OSError as exc:
                logger.error("Failed to initialize CSV logging: %s", exc)
                if self.callbacks.on_logging_error is not None:
                    self.callbacks.on_logging_error(exc)
                self.loss_logger = None
                self.loss_metrics = None
        else:
            self.loss_metrics = {
                "style_loss": [],
                "content_loss": [],
                "total_loss": [],
            }
            self.loss_logger = None

            if steps > CSV_LOGGING_RECOMMENDED_STEPS:
                logger.warning(
                    (
                        "Long run detected (%d steps). Consider enabling "
                        "--log-loss to reduce memory usage."
                    ),
                    steps,
                )

    def _ensure_progress_bar(self) -> None:
        """Initialise the progress bar if one was not provided."""
        if self._progress_bar is None:
            self._progress_bar = tqdm(
                total=self.total_steps,
                desc="Style Transfer",
            )
            self._owns_progress_bar = True

    def _closure(self) -> float:
        """Optimizer closure used by LBFGS and compatible optimizers."""
        self._closure_calls += 1

        if self._step_index >= self.total_steps:
            return 0.0 if self.last_loss is None else self.last_loss

        step_idx = self._active_step_idx or (self._step_index + 1)
        metrics = self._run_single_step(step_idx)
        self._pending_step_metrics = metrics
        return metrics.total_loss

    def _run_single_step(self, step_idx: int) -> StepMetrics:
        """Execute a single forward/backward update and record metrics."""
        cfg = self.config
        optimizer = self.optimizer

        optimizer.zero_grad()
        style_losses, content_losses = self.model(self.input_img)
        style_components = [s.item() for s in style_losses]

        device = self.input_img.device
        dtype = self.input_img.dtype
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

        loss = (
            cfg.optimization.style_w * style_score
            + cfg.optimization.content_w * content_score
        )
        loss.backward()

        self._check_finite(
            style_score,
            content_score,
            loss,
            step_idx,
            style_components,
        )

        loss_value = loss.item()
        return StepMetrics(
            step=step_idx,
            style_loss=style_score.item(),
            content_loss=content_score.item(),
            total_loss=loss_value,
        )

    def _finalize_step(self, metrics: StepMetrics) -> None:
        """Record metrics and emit hooks after a successful optimizer step."""
        self._step_index = metrics.step
        self.last_loss = metrics.total_loss

        self._record_losses(metrics)
        self._maybe_write_video_frame(metrics)
        self.progress_bar.update(1)
        self._emit_step_end(metrics)

    def _log_optimization_summary(self) -> None:
        """Log how many closure evaluations were consumed per accepted step."""
        if self._step_index <= 0:
            return
        avg_closures = self._closure_calls / self._step_index
        logger.info(
            (
                "Optimization finished with %d accepted steps and %d closure "
                "evaluations (%.2f closures/step)."
            ),
            self._step_index,
            self._closure_calls,
            avg_closures,
        )

    def _check_finite(
        self,
        style_score: torch.Tensor,
        content_score: torch.Tensor,
        total_loss: torch.Tensor,
        step_idx: int,
        style_components: list[float],
    ) -> None:
        """Warn if any recorded loss is non-finite."""
        if not torch.isfinite(style_score):
            logger.warning("Non-finite style score at step %d", step_idx)
        if not torch.isfinite(content_score):
            logger.warning("Non-finite content score at step %d", step_idx)
        if not torch.isfinite(total_loss):
            logger.warning(
                "Non-finite total loss at step %d, using previous loss",
                step_idx,
            )

        logger.debug(
            "Step %d: Style %s, Content %.4e, Total %.4e",
            step_idx,
            style_components,
            content_score.item(),
            total_loss.item(),
        )

    def _record_losses(self, metrics: StepMetrics) -> None:
        """Persist loss metrics to CSV or in-memory buffers."""
        if self.loss_metrics is not None:
            self.loss_metrics["style_loss"].append(metrics.style_loss)
            self.loss_metrics["content_loss"].append(metrics.content_loss)
            self.loss_metrics["total_loss"].append(metrics.total_loss)

        if self.loss_logger is not None:
            self.loss_logger.log(
                metrics.step,
                metrics.style_loss,
                metrics.content_loss,
                metrics.total_loss,
            )

    def _maybe_write_video_frame(self, metrics: StepMetrics) -> None:
        """Write a timelapse frame when configured to do so."""
        save_every = self.config.video.save_every
        video_writer = self.video_writer
        gif_collector = self.gif_collector
        step_idx = metrics.step

        if (
            not save_every
            or step_idx % save_every != 0
            or (video_writer is None and gif_collector is None)
        ):
            return

        with torch.no_grad():
            image_tensor = stv_image_io.prepare_image_for_output(
                self.input_img,
                normalize=self.config.optimization.normalize,
            )
            if image_tensor is None:
                return

            img_np = (
                image_tensor.squeeze(0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
                * 255
            ).astype("uint8")

        if (
            self.intro_last_frame is not None
            and not self.intro_transition_done
        ):
            if (
                video_writer is not None
                and self.config.video.intro_enabled
            ):
                stv_video.append_crossfade(
                    video_writer,
                    self.intro_last_frame,
                    img_np,
                    self.intro_crossfade_frames,
                )
            if (
                gif_collector is not None
                and self.config.video.gif_include_intro
            ):
                stv_video.append_crossfade(
                    gif_collector,
                    self.intro_last_frame,
                    img_np,
                    self.intro_crossfade_frames,
                )
            self.intro_transition_done = True
            self.intro_last_frame = None

        if video_writer is not None:
            video_writer.append_data(img_np)
        if gif_collector is not None:
            gif_collector.append_data(img_np)
        self.progress_bar.set_postfix({
            "style": f"{metrics.style_loss:.4f}",
            "content": f"{metrics.content_loss:.4f}",
            "loss": f"{metrics.total_loss:.4f}",
        })

        if self.callbacks.on_video_frame is not None:
            self.callbacks.on_video_frame(img_np, step_idx)

    def _emit_step_start(self, step_idx: int) -> None:
        """Fire the on_step_start callback if registered."""
        if self.callbacks.on_step_start is not None:
            self.callbacks.on_step_start(step_idx)

    def _emit_step_end(self, metrics: StepMetrics) -> None:
        """Fire the on_step_end callback if registered."""
        if self.callbacks.on_step_end is not None:
            self.callbacks.on_step_end(metrics)

    def _cleanup(self) -> None:
        """Release any resources acquired during the run."""
        if self.loss_logger is not None:
            self.loss_logger.close()

        if self._owns_progress_bar and self._progress_bar is not None:
            self._progress_bar.close()
