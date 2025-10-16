"""Tests for visualization.metrics helpers."""

from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path

import pytest

from style_transfer_visualizer.visualization import metrics as viz_metrics


def test_plot_loss_curves_no_metrics(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    with caplog.at_level("WARNING"):
        viz_metrics.plot_loss_curves({}, tmp_path)
    assert "No loss metrics dictionary provided." in caplog.text


def test_plot_loss_curves_empty_sequences(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    with caplog.at_level("WARNING"):
        viz_metrics.plot_loss_curves({"loss": []}, tmp_path)
    assert "Loss metrics dictionary is empty" in caplog.text


def test_plot_loss_curves_missing_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    original_import = builtins.__import__

    def fake_import(
        name: str,
        globals_: dict | None = None,
        locals_: dict | None = None,
        fromlist: tuple | None = None,
        level: int = 0,
    ):
        if name == "matplotlib.pyplot":
            raise ImportError("mock matplotlib missing")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with caplog.at_level("WARNING"):
        viz_metrics.plot_loss_curves({"loss": [1.0]}, tmp_path)

    assert "matplotlib not found" in caplog.text


def test_plot_loss_curves_saves_png(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    saved: dict[str, object] = {}

    class DummyFigure:
        def close(self) -> None:
            saved["closed"] = True

    class DummyPyplot:
        def __init__(self) -> None:
            self._fig = DummyFigure()
            saved["plots"] = []

        def figure(self, *args: object, **kwargs: object) -> DummyFigure:
            return self._fig

        def plot(self, values: list[float], *, label: str | None = None) -> None:
            saved["plots"].append((tuple(values), label))

        @staticmethod
        def xlabel(label: str) -> None:
            saved["xlabel"] = label

        @staticmethod
        def ylabel(label: str) -> None:
            saved["ylabel"] = label

        @staticmethod
        def title(label: str) -> None:
            saved["title"] = label

        @staticmethod
        def legend() -> None:
            saved["legend"] = True

        @staticmethod
        def tight_layout() -> None:
            saved["tight_layout"] = True

        @staticmethod
        def savefig(target: Path) -> None:
            saved["target"] = Path(target)

        @staticmethod
        def close(fig: DummyFigure) -> None:
            fig.close()

    dummy_pyplot = DummyPyplot()
    dummy_matplotlib = types.ModuleType("matplotlib")
    dummy_matplotlib.pyplot = dummy_pyplot
    dummy_matplotlib.__path__ = []  # mark as package
    monkeypatch.setitem(sys.modules, "matplotlib", dummy_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", dummy_pyplot)

    viz_metrics.plot_loss_curves(
        {"style_loss": [1.0, 0.5], "content_loss": []},
        tmp_path,
    )

    expected = tmp_path / "loss_plot.png"
    assert saved["target"] == expected
    assert saved["plots"] == [((1.0, 0.5), "style_loss")]
    assert saved["xlabel"] == "Step"
    assert saved["ylabel"] == "Loss"
    assert saved["title"] == "Loss Curves"
