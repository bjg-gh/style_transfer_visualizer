"""Plotting helpers for monitoring optimization metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from style_transfer_visualizer.logging_utils import logger

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    from style_transfer_visualizer.type_defs import LossHistory


def plot_loss_curves(metrics: LossHistory, output_dir: Path) -> None:
    """
    Persist a matplotlib plot of loss curve metrics, if available.

    Skips work when metrics are missing, empty, or matplotlib cannot be
    imported in the current environment.
    """
    if not metrics:
        logger.warning("No loss metrics dictionary provided.")
        return

    if not any(len(values) > 0 for values in metrics.values()):
        logger.warning("Loss metrics dictionary is empty, nothing to plot.")
        return

    try:  # deferred import keeps matplotlib optional
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        logger.warning("matplotlib not found: skipping loss plot.")
        return

    figure = plt.figure(figsize=(10, 6))
    try:
        for series_name, series_values in metrics.items():
            if series_values:
                plt.plot(series_values, label=series_name)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Loss Curves")
        plt.legend()
        plt.tight_layout()
        loss_plot_path = output_dir / "loss_plot.png"
        plt.savefig(loss_plot_path)
        logger.info("Loss plot saved to: %s", loss_plot_path)
    finally:
        plt.close(figure)
