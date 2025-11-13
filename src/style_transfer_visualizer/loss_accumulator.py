"""Device-side helpers for batching loss logging."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover
    from style_transfer_visualizer.type_defs import LossHistory


DEFAULT_HISTORY_CAPACITY = 2048


@dataclass(slots=True)
class LoggedLoss:
    """Scalar loss values flushed from the device."""

    step: int
    style_loss: float
    content_loss: float
    total_loss: float


class LossAccumulator:
    """
    Aggregate loss tensors on the device and batch host synchronizations.

    Tracks per-step loss values using a circular buffer to keep memory usage
    bounded while only materializing Python floats at the configured logging
    cadence (``log_every``). Consumers can still request the latest synced
    values for progress reporting or CSV logging without paying the cost on
    every optimizer closure invocation.
    """

    def __init__(
        self,
        *,
        log_every: int,
        history_capacity: int | None,
        track_history: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self._log_every = max(1, log_every)
        self._history_capacity = max(
            1,
            history_capacity or DEFAULT_HISTORY_CAPACITY,
        )
        self._track_history = track_history
        self._device = device
        self._buffer_dtype = (
            torch.float32 if dtype != torch.float16 else torch.float16
        )

        self._style_history: torch.Tensor | None = None
        self._content_history: torch.Tensor | None = None
        self._total_history: torch.Tensor | None = None
        self._history_index = 0
        self._history_count = 0
        self._history_total_records = 0
        self._history_truncated = False

        if track_history:
            self._style_history = torch.empty(
                self._history_capacity,
                dtype=self._buffer_dtype,
                device=self._device,
            )
            self._content_history = torch.empty_like(self._style_history)
            self._total_history = torch.empty_like(self._style_history)

        self._pending_step: int | None = None
        self._pending_style: torch.Tensor | None = None
        self._pending_content: torch.Tensor | None = None
        self._pending_total: torch.Tensor | None = None
        self._last_logged: LoggedLoss | None = None

    @property
    def capacity(self) -> int:
        """Return the maximum number of history entries stored in-memory."""
        return self._history_capacity

    @property
    def tracks_history(self) -> bool:
        """Return True when in-memory metrics are being tracked."""
        return self._track_history

    @property
    def history_truncated(self) -> bool:
        """Return True if the circular buffer has overwritten older entries."""
        return self._history_truncated

    def accumulate(
        self,
        step_idx: int,
        style_loss: torch.Tensor,
        content_loss: torch.Tensor,
        total_loss: torch.Tensor,
        *,
        force: bool = False,
    ) -> LoggedLoss | None:
        """
        Record per-step losses.

        Return scalars only at the logging cadence.
        """
        style_detached = style_loss.detach()
        content_detached = content_loss.detach()
        total_detached = total_loss.detach()

        self._pending_step = step_idx
        self._pending_style = style_detached
        self._pending_content = content_detached
        self._pending_total = total_detached

        if self._track_history:
            self._record_history(style_detached, content_detached,
                                 total_detached)

        should_sync = force or (step_idx % self._log_every == 0)
        if should_sync:
            return self._sync_pending()
        return None

    def latest(self) -> LoggedLoss | None:
        """Return the most recent synced loss scalars."""
        return self._last_logged

    def export_history(self) -> LossHistory:
        """Return bounded loss history as lists suitable for plotting."""
        if not self._track_history or self._history_count == 0:
            return {"style_loss": [], "content_loss": [], "total_loss": []}

        style = self._gather_history(self._style_history)
        content = self._gather_history(self._content_history)
        total = self._gather_history(self._total_history)

        return {
            "style_loss": style.cpu().tolist(),
            "content_loss": content.cpu().tolist(),
            "total_loss": total.cpu().tolist(),
        }

    def _record_history(
        self,
        style_loss: torch.Tensor,
        content_loss: torch.Tensor,
        total_loss: torch.Tensor,
    ) -> None:
        if (
            self._style_history is None
            or self._content_history is None
            or self._total_history is None
        ):
            msg = "History buffers are uninitialized."
            raise RuntimeError(msg)

        idx = self._history_index
        self._style_history[idx] = self._prepare_history_value(style_loss)
        self._content_history[idx] = self._prepare_history_value(content_loss)
        self._total_history[idx] = self._prepare_history_value(total_loss)

        self._history_index = (idx + 1) % self._history_capacity
        self._history_count = min(self._history_count + 1,
                                  self._history_capacity)
        self._history_total_records += 1
        if self._history_total_records > self._history_capacity:
            self._history_truncated = True

    def _gather_history(self, buffer: torch.Tensor | None) -> torch.Tensor:
        if buffer is None:
            return torch.empty(0, device=self._device,
                               dtype=self._buffer_dtype)

        count = self._history_count
        if count == 0:
            return torch.empty(0, device=buffer.device, dtype=buffer.dtype)

        capacity = self._history_capacity
        start = (self._history_index - count) % capacity

        if start + count <= capacity:
            return buffer.narrow(0, start, count)

        first = buffer.narrow(0, start, capacity - start)
        second = buffer.narrow(0, 0, count - first.size(0))
        return torch.cat((first, second))

    def _prepare_history_value(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(dtype=self._buffer_dtype, device=self._device)

    def _sync_pending(self) -> LoggedLoss | None:
        if (
            self._pending_step is None
            or self._pending_style is None
            or self._pending_content is None
            or self._pending_total is None
        ):
            return None

        logged = LoggedLoss(
            step=self._pending_step,
            style_loss=self._to_float(self._pending_style),
            content_loss=self._to_float(self._pending_content),
            total_loss=self._to_float(self._pending_total),
        )
        self._last_logged = logged
        return logged

    def _to_float(self, tensor: torch.Tensor) -> float:
        return float(tensor.item())
