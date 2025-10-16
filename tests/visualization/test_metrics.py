"""Tests for visualization.metrics helpers."""

from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path
from typing import Any, Callable, Sequence

import pytest

from style_transfer_visualizer.visualization import metrics as viz_metrics


class _DummyFigure:
    def __init__(self, saved: dict[str, Any]) -> None:
        self._saved = saved

    def close(self) -> None:
        self._saved["closed"] = True


class _FakePyplot:
    def __init__(self, saved: dict[str, Any]) -> None:
        self._saved = saved
        self._figure = _DummyFigure(saved)
        self.plots: list[tuple[tuple[float, ...], str | None]] = []
        saved["plots"] = self.plots

    def figure(self, *args: object, **kwargs: object) -> _DummyFigure:
        return self._figure

    def plot(
        self,
        values: Sequence[float],
        *,
        label: str | None = None,
    ) -> None:
        self.plots.append((tuple(values), label))

    def xlabel(self, label: str) -> None:
        self._saved["xlabel"] = label

    def ylabel(self, label: str) -> None:
        self._saved["ylabel"] = label

    def title(self, label: str) -> None:
        self._saved["title"] = label

    @staticmethod
    def legend() -> None:
        """Record that a legend was requested."""

    def tight_layout(self) -> None:
        self._saved["tight_layout"] = True

    def savefig(self, target: Path | str) -> None:
        self._saved["target"] = Path(target)

    def close(self, figure: _DummyFigure) -> None:
        figure.close()


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
    original_import: Callable[..., object] = builtins.__import__

    def fake_import(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> object:
        if name == "matplotlib.pyplot":
            raise ImportError("mock matplotlib missing")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with caplog.at_level("WARNING"):
        viz_metrics.plot_loss_curves({"loss": [1.0]}, tmp_path)

    assert "matplotlib not found" in caplog.text


def _install_dummy_pyplot(
    monkeypatch: pytest.MonkeyPatch,
    saved: dict[str, Any],
) -> list[tuple[tuple[float, ...], str | None]]:
    dummy_pyplot = _FakePyplot(saved)
    dummy_matplotlib = types.ModuleType("matplotlib")
    dummy_matplotlib.__dict__["pyplot"] = dummy_pyplot
    dummy_matplotlib.__path__ = []  # type: ignore[assignment]
    monkeypatch.setitem(sys.modules, "matplotlib", dummy_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", dummy_pyplot)
    return dummy_pyplot.plots


def test_plot_loss_curves_saves_png(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    saved: dict[str, Any] = {}
    plots = _install_dummy_pyplot(monkeypatch, saved)

    viz_metrics.plot_loss_curves(
        {"style_loss": [1.0, 0.5], "content_loss": []},
        tmp_path,
    )

    expected = tmp_path / "loss_plot.png"
    assert saved["target"] == expected
    assert plots == [((1.0, 0.5), "style_loss")]
    assert saved["xlabel"] == "Step"
    assert saved["ylabel"] == "Loss"
    assert saved["title"] == "Loss Curves"
