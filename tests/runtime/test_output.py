"""Tests for runtime.output helpers."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from pathlib import Path, Path as RealPath
from typing import TYPE_CHECKING, cast

import torch

from style_transfer_visualizer.runtime import output as runtime_output
from style_transfer_visualizer.type_defs import SaveOptions

if TYPE_CHECKING:
    import pytest


def test_setup_output_directory_creates_path(tmp_path: Path) -> None:
    target = tmp_path / "new_dir"
    result = runtime_output.setup_output_directory(str(target))
    assert result == target
    assert target.exists()


def test_setup_output_directory_fallback(tmp_path: Path) -> None:
    class FailingPath(RealPath):
        def mkdir(  # type: ignore[override]
            self,
            mode: int = 0o777,
            parents: bool = False,  # noqa: FBT001, FBT002
            exist_ok: bool = False,  # noqa: FBT001, FBT002
        ) -> None:
            if "restricted" in str(self):
                raise PermissionError("Mock failure")
            return super().mkdir(mode=mode, parents=parents, exist_ok=exist_ok)

    result = runtime_output.setup_output_directory(
        "restricted",
        path_factory=cast(Callable[[str], Path], FailingPath),
    )
    assert result.name == "style_transfer_output"
    assert result.exists()


def test_stylized_path_helpers(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    content = tmp_path / "my content.png"
    style = tmp_path / "style pic.jpg"
    content.write_bytes(b"")
    style.write_bytes(b"")

    path = runtime_output.stylized_image_path_from_paths(
        out_dir,
        content,
        style,
    )
    assert path == out_dir / "stylized_my_content_x_style_pic.png"


def test_save_outputs_happy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    image_called: dict[str, Path] = {}
    plot_called: dict[str, Path] = {}

    monkeypatch.setattr(
        "style_transfer_visualizer.image_io.prepare_image_for_output",
        lambda tensor, normalize: tensor,
    )
    monkeypatch.setattr(
        "torchvision.utils.save_image",
        lambda tensor, target_path: image_called.setdefault("path", Path(target_path)),
    )
    monkeypatch.setattr(
        "style_transfer_visualizer.visualization.metrics.plot_loss_curves",
        lambda metrics, output_dir: plot_called.setdefault("path", output_dir),
    )

    tensor = torch.rand(1, 3, 8, 8)
    opts = SaveOptions(
        content_name="content",
        style_name="style",
        video_name="video.mp4",
        normalize=True,
        video_created=True,
        plot_losses=True,
    )

    runtime_output.save_outputs(
        tensor,
        {"total_loss": [1.0]},
        tmp_path,
        elapsed=1.23,
        opts=opts,
    )

    expected_image = tmp_path / "stylized_content_x_style.png"
    assert image_called["path"] == expected_image
    assert plot_called["path"] == tmp_path
    assert "Final stylized image saved to" in caplog.text


def test_save_outputs_logs_gif_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(
        "style_transfer_visualizer.image_io.prepare_image_for_output",
        lambda tensor, normalize: tensor,
    )
    monkeypatch.setattr(
        "torchvision.utils.save_image",
        lambda tensor, target_path: target_path,
    )
    monkeypatch.setattr(
        "style_transfer_visualizer.visualization.metrics.plot_loss_curves",
        lambda *_a, **_k: None,
    )

    gif_path = tmp_path / "anim.gif"
    gif_path.write_bytes(b"GIF89a")

    opts = SaveOptions(
        content_name="content",
        style_name="style",
        video_name=None,
        gif_name="anim.gif",
        normalize=False,
        video_created=False,
        gif_created=True,
        plot_losses=False,
    )

    runtime_output.save_outputs(
        torch.rand(1, 3, 2, 2),
        {},
        tmp_path,
        elapsed=0.2,
        opts=opts,
    )

    assert "GIF saved to" in caplog.text


def test_save_outputs_fallback_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Path] = {}
    monkeypatch.setattr(
        "style_transfer_visualizer.image_io.prepare_image_for_output",
        lambda tensor, normalize: tensor,
    )
    monkeypatch.setattr(
        "torchvision.utils.save_image",
        lambda tensor, target_path: calls.setdefault("path", Path(target_path)),
    )
    monkeypatch.setattr(
        "style_transfer_visualizer.visualization.metrics.plot_loss_curves",
        lambda *_a, **_k: None,
    )

    failing_path = tmp_path / "restricted"
    original_mkdir = Path.mkdir

    def fake_mkdir(
        self: Path,
        mode: int = 0o777,
        parents: bool = False,  # noqa: FBT001, FBT002
        exist_ok: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        if self == failing_path:
            raise PermissionError("boom")
        return original_mkdir(
            self,
            mode=mode,
            parents=parents,
            exist_ok=exist_ok,
        )

    monkeypatch.setattr(Path, "mkdir", fake_mkdir)

    opts = SaveOptions(
        content_name="content",
        style_name="style",
        video_name=None,
        normalize=False,
        video_created=False,
        plot_losses=False,
    )

    runtime_output.save_outputs(
        torch.rand(1, 3, 4, 4),
        {},
        failing_path,
        elapsed=0.5,
        opts=opts,
    )

    fallback_path = Path("style_transfer_output") / "stylized_content_x_style.png"
    assert calls["path"] == fallback_path
    if fallback_path.parent.exists():
        shutil.rmtree(fallback_path.parent)


def test_save_outputs_creates_missing_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, Path] = {}
    monkeypatch.setattr(
        "style_transfer_visualizer.image_io.prepare_image_for_output",
        lambda tensor, normalize: tensor,
    )
    monkeypatch.setattr(
        "torchvision.utils.save_image",
        lambda tensor, target_path: created.setdefault("path", Path(target_path)),
    )
    monkeypatch.setattr(
        "style_transfer_visualizer.visualization.metrics.plot_loss_curves",
        lambda *_a, **_k: None,
    )

    output_dir = tmp_path / "brand_new"
    assert not output_dir.exists()

    opts = SaveOptions(
        content_name="content",
        style_name="style",
        video_name=None,
        normalize=False,
        video_created=False,
        plot_losses=False,
    )

    runtime_output.save_outputs(
        torch.rand(1, 3, 2, 2),
        {},
        output_dir,
        elapsed=0.1,
        opts=opts,
    )

    assert output_dir.exists()
    assert created["path"] == output_dir / "stylized_content_x_style.png"


def test_save_outputs_skips_missing_gif_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No GIF log entry should be emitted when the GIF file is absent."""
    caplog.set_level(logging.INFO)

    monkeypatch.setattr(
        "style_transfer_visualizer.image_io.prepare_image_for_output",
        lambda tensor, normalize: tensor,
    )
    monkeypatch.setattr(
        "torchvision.utils.save_image",
        lambda tensor, target_path: target_path,
    )
    monkeypatch.setattr(
        "style_transfer_visualizer.visualization.metrics.plot_loss_curves",
        lambda *_a, **_k: None,
    )

    opts = SaveOptions(
        content_name="content",
        style_name="style",
        gif_name="missing.gif",
        normalize=False,
        video_created=False,
        gif_created=True,
        plot_losses=False,
    )

    # Ensure the GIF file truly does not exist.
    gif_path = tmp_path / "missing.gif"
    if gif_path.exists():
        gif_path.unlink()

    runtime_output.save_outputs(
        torch.rand(1, 3, 2, 2),
        {},
        tmp_path,
        elapsed=0.1,
        opts=opts,
    )

    assert not any("GIF saved to" in message for message in caplog.messages)
