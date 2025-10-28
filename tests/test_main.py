"""
Integration and functional tests for the style_transfer() pipeline.

Covers:
- Minimal end-to-end runs with mocked components
- Video creation toggles and parameter passing
- Final-only mode behavior
- Output validation and directory handling
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np
import pytest
import torch

import style_transfer_visualizer.core_model as stv_core_model
import style_transfer_visualizer.image_io as stv_image_io
import style_transfer_visualizer.main as stv_main
import style_transfer_visualizer.optimization as stv_optimization
import style_transfer_visualizer.runtime as stv_runtime
import style_transfer_visualizer.video as stv_video
from style_transfer_visualizer.config import StyleTransferConfig, VideoConfig
from style_transfer_visualizer.type_defs import InputPaths, SaveOptions

if TYPE_CHECKING:
    from collections.abc import Callable

    from _pytest.monkeypatch import MonkeyPatch


@dataclass(slots=True)
class VideoTestAssets:
    """Bundle shared fixtures for video-related integration tests."""

    make_subdir: Callable[[str], Path]
    content_image: str
    style_image: str
    make_config: Callable[..., StyleTransferConfig]
    make_input_paths: Callable[..., InputPaths]

    def new_output(self, name: str) -> Path:
        """Create a fresh output directory under the test root."""
        return self.make_subdir(name)

    def inputs(
        self,
        *,
        content: str | Path | None = None,
        style: str | Path | None = None,
    ) -> InputPaths:
        """Build InputPaths with optional overrides."""
        return self.make_input_paths(
            content=content if content is not None else self.content_image,
            style=style if style is not None else self.style_image,
        )

    def config(
        self,
        *,
        optimization: dict[str, Any] | None = None,
        video: dict[str, Any] | None = None,
        output: dict[str, Any] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> StyleTransferConfig:
        """Create a StyleTransferConfig with provided overrides."""
        return self.make_config(
            optimization=optimization,
            video=video,
            output=output,
            extras=extras,
        )


@pytest.fixture
def video_assets(
    make_output_subdir: Callable[[str], Path],
    content_image: Path,
    style_image: Path,
    make_style_transfer_config: Callable[..., StyleTransferConfig],
    make_input_paths: Callable[..., InputPaths],
) -> VideoTestAssets:
    """Aggregate reusable helpers for video-centric tests."""
    return VideoTestAssets(
        make_subdir=make_output_subdir,
        content_image=str(content_image),
        style_image=str(style_image),
        make_config=make_style_transfer_config,
        make_input_paths=make_input_paths,
    )


def test_style_transfer_minimal(
    monkeypatch: MonkeyPatch,
    make_style_transfer_config: Callable[..., StyleTransferConfig],
    make_input_paths: Callable[..., InputPaths],
) -> None:
    """Smoke test for style_transfer() with mocked internals."""
    dummy_tensor = torch.rand(1, 3, 256, 256)

    # IO and model prep
    monkeypatch.setattr(
        stv_image_io,
        "load_image_to_tensor",
        lambda *_args, **_kwargs: dummy_tensor,
    )
    monkeypatch.setattr(
        stv_core_model,
        "prepare_model_and_input",
        lambda *_args, **_kwargs: ("model", dummy_tensor.clone(), "optimizer"),
    )

    # Optimization loop returns a stable output and loss dict
    patch_runner(
        monkeypatch,
        run_result=(dummy_tensor.clone(), {"loss": [1.0]}, 3.14),
    )

    # Filesystem helpers
    monkeypatch.setattr(stv_runtime, "save_outputs", lambda *_a, **_k: None)
    monkeypatch.setattr(
        stv_runtime,
        "setup_output_directory",
        lambda _x: Path("mock_output"),
    )
    monkeypatch.setattr(
        stv_runtime,
        "validate_input_paths",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        stv_video,
        "setup_video_writer",
        lambda *_a, **_k: None,
    )

    # Build inputs via new API
    paths = make_input_paths(content="dummy.jpg", style="dummy2.jpg")
    cfg = make_style_transfer_config(
        optimization={
            "steps": 10,
            "normalize": True,
            "seed": 42,
        },
        output={"output": "output", "plot_losses": True},
        video={
            "create_video": False,
            "save_every": 5,
            "fps": 1,
            "quality": 5,
        },
    )

    result = stv_main.style_transfer(paths, cfg)

    assert isinstance(result, torch.Tensor)
    assert result.shape == dummy_tensor.shape


def test_style_transfer_auto_selects_postprocess(
    monkeypatch: MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    make_style_transfer_config: Callable[..., StyleTransferConfig],
    make_input_paths: Callable[..., InputPaths],
) -> None:
    """Video mode should switch and log when heuristic recommends it."""
    dummy_tensor = torch.rand(1, 3, 64, 96)

    monkeypatch.setattr(
        stv_image_io,
        "load_image_to_tensor",
        lambda *_a, **_kw: dummy_tensor,
    )
    monkeypatch.setattr(
        stv_core_model,
        "prepare_model_and_input",
        lambda *_a, **_kw: ("model", dummy_tensor.clone(), "optimizer"),
    )
    patch_runner(
        monkeypatch,
        run_result=(dummy_tensor.clone(), {"total_loss": []}, 0.1),
    )

    class DummyWriter:
        def __init__(self) -> None:
            self._size: tuple[int, int] | None = None

        def append_data(self, frame: np.ndarray) -> None:
            self._size = (frame.shape[1], frame.shape[0])

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        stv_video,
        "setup_video_writer",
        lambda *_a, **_kw: DummyWriter(),
    )
    monkeypatch.setattr(stv_video, "prepare_intro_segment", lambda *_a, **_kw: None)
    monkeypatch.setattr(stv_video, "append_final_comparison_frame", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        stv_video,
        "select_video_mode",
        lambda *_a, **_kw: ("postprocess", "heavy test run", 42),
    )

    monkeypatch.setattr(stv_runtime, "validate_input_paths", lambda *_a, **_kw: None)
    monkeypatch.setattr(stv_runtime, "validate_parameters", lambda *_a, **_kw: None)
    monkeypatch.setattr(stv_runtime, "setup_random_seed", lambda *_a, **_kw: None)
    monkeypatch.setattr(stv_runtime, "setup_device", lambda *_a, **_kw: "cpu")
    monkeypatch.setattr(
        stv_runtime,
        "setup_output_directory",
        lambda *_a, **_kw: Path("mock_output"),
    )
    monkeypatch.setattr(stv_runtime, "save_outputs", lambda *_a, **_kw: None)

    cfg = make_style_transfer_config(video={"final_frame_compare": False})

    paths = make_input_paths(content="foo.png", style="bar.png")

    with caplog.at_level("INFO"):
        stv_main.style_transfer(paths, cfg)

    assert cfg.video.mode == "postprocess"
    assert any(
        "Auto-selected postprocess video mode" in record.message
        for record in caplog.records
    )


def test_style_transfer_no_plot(
    monkeypatch: MonkeyPatch,
    make_style_transfer_config: Callable[..., StyleTransferConfig],
    make_input_paths: Callable[..., InputPaths],
) -> None:
    """Test style_transfer() with plotting disabled."""
    dummy_tensor = torch.rand(1, 3, 256, 256)

    monkeypatch.setattr(
        stv_image_io,
        "load_image_to_tensor",
        lambda *_a, **_kw: dummy_tensor,
    )
    monkeypatch.setattr(
        stv_core_model,
        "prepare_model_and_input",
        lambda *_a, **_kw: ("model", dummy_tensor.clone(), "optimizer"),
    )
    patch_runner(
        monkeypatch,
        run_result=(dummy_tensor.clone(), {"loss": [1.0]}, 3.14),
    )
    monkeypatch.setattr(stv_runtime, "save_outputs", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        stv_runtime,
        "setup_output_directory",
        lambda _: Path("mock_output").resolve().mkdir(
            parents=True, exist_ok=True,
        )
        or Path("mock_output"),
    )
    monkeypatch.setattr(stv_runtime, "validate_input_paths",
                        lambda *_a, **_kw: None)
    monkeypatch.setattr(stv_video, "setup_video_writer",
                        lambda *_a, **_k: None)

    paths = make_input_paths(content="dummy.jpg", style="dummy2.jpg")
    cfg = make_style_transfer_config(
        optimization={"steps": 1},
        video={"create_video": False},
        output={"output": "output", "plot_losses": False},
    )

    result = stv_main.style_transfer(paths, cfg)
    assert isinstance(result, torch.Tensor)


def expected_video_path(
    out_dir: str | Path,
    content_img: str,
    style_img: str,
) -> Path:
    """Construct expected video file path from image names."""
    c = Path(content_img).stem
    s = Path(style_img).stem
    return Path(out_dir) / f"timelapse_{c}_x_{s}.mp4"


@pytest.mark.parametrize("create_video_flag", [True, False])
def test_video_creation_flag(
    video_assets: VideoTestAssets,
    create_video_flag: bool,  # noqa: FBT001
    monkeypatch: MonkeyPatch,
) -> None:
    """Test whether video is created when flag is toggled."""
    output_dir = video_assets.new_output("video_test")
    apply_mock(monkeypatch)

    paths = video_assets.inputs()
    cfg = video_assets.config(
        optimization={"steps": 1},
        video={
            "save_every": 1,
            "quality": 1,
            "create_video": create_video_flag,
        },
        output={"output": output_dir},
    )

    result = stv_main.style_transfer(paths, cfg)

    assert isinstance(result, torch.Tensor)
    video = expected_video_path(
        output_dir,
        video_assets.content_image,
        video_assets.style_image,
    )
    if create_video_flag:
        assert video.exists()
    else:
        assert not video.exists()


@pytest.mark.parametrize(("fps", "quality"), [(10, 1), (24, 5), (30, 10)])
def test_video_params_passed(
    video_assets: VideoTestAssets,
    fps: int,
    quality: int,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test video FPS and quality are passed to writer."""
    output_dir = video_assets.new_output("video_param_test")
    apply_mock(monkeypatch)

    calls: list[tuple[int, int, Path, str]] = []
    orig_writer = stv_video.setup_video_writer

    def wrapped(
        config: VideoConfig,
        output_path: Path,
        video_name: str,
    ) -> object:
        # Record parameters to verify fps and quality propagation.
        calls.append((config.fps, config.quality, output_path, video_name))
        return orig_writer(config, output_path, video_name)

    monkeypatch.setattr(stv_video, "setup_video_writer", wrapped)

    paths = video_assets.inputs()
    cfg = video_assets.config(
        optimization={"steps": 1},
        video={
            "create_video": True,
            "save_every": 1,
            "fps": fps,
            "quality": quality,
        },
        output={"output": output_dir},
    )

    stv_main.style_transfer(paths, cfg)

    assert calls
    fps_seen, quality_seen, _out_path, _name_seen = calls[0]
    assert fps_seen == fps
    assert quality_seen == quality


def test_style_transfer_passes_intro_info(
    video_assets: VideoTestAssets,
    monkeypatch: MonkeyPatch,
) -> None:
    """Intro metadata flows from prepare_intro_segment into optimization."""
    output_dir = video_assets.new_output("intro_info")
    dummy_tensor = torch.rand(1, 3, 64, 64)

    monkeypatch.setattr(
        stv_runtime,
        "validate_input_paths",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        stv_runtime,
        "validate_parameters",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        stv_runtime,
        "setup_random_seed",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        stv_runtime,
        "setup_device",
        lambda *_a, **_k: torch.device("cpu"),
    )
    monkeypatch.setattr(
        stv_runtime,
        "setup_output_directory",
        lambda _p: Path(output_dir),
    )
    monkeypatch.setattr(
        stv_runtime,
        "save_outputs",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        stv_image_io,
        "load_image_to_tensor",
        lambda *_a, **_k: dummy_tensor,
    )
    monkeypatch.setattr(
        stv_core_model,
        "prepare_model_and_input",
        lambda *_a, **_k: ("model", dummy_tensor.clone(), "optimizer"),
    )

    captured: dict[str, object] = {}

    def init_hook(
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> tuple[torch.Tensor, dict[str, list[float]], float]:
        input_img = cast(torch.Tensor, args[1])
        captured["writer"] = kwargs.get("video_writer")
        captured["intro_last_frame"] = kwargs.get("intro_last_frame")
        captured["intro_crossfade_frames"] = kwargs.get(
            "intro_crossfade_frames",
        )
        return input_img, {"loss": []}, 0.0

    patch_runner(monkeypatch, init_hook=init_hook)

    class StubWriter:
        def __init__(self) -> None:
            self.frames: list[np.ndarray] = []
            self.closed = False

        def append_data(self, frame: np.ndarray) -> None:
            self.frames.append(frame)

        def close(self) -> None:
            self.closed = True

    writer_instance = StubWriter()
    monkeypatch.setattr(
        stv_video,
        "setup_video_writer",
        lambda *_a, **_k: writer_instance,
    )
    monkeypatch.setattr(
        stv_video,
        "prepare_intro_segment",
        lambda *_a, **_k: (np.zeros((64, 64, 3), dtype=np.uint8), 5),
    )

    paths = video_assets.inputs()
    cfg = video_assets.config(
        optimization={"steps": 1},
        video={"create_video": True, "save_every": 1},
        output={"output": output_dir},
    )
    expected_crossfade = round(
        cfg.video.fps * stv_video.INTRO_CROSSFADE_SECONDS,
    )

    result = stv_main.style_transfer(paths, cfg)

    assert isinstance(result, torch.Tensor)
    assert isinstance(captured["writer"], StubWriter)
    assert captured["intro_crossfade_frames"] == expected_crossfade
    intro_frame = captured["intro_last_frame"]
    assert isinstance(intro_frame, np.ndarray)
    assert intro_frame.shape == (64, 64, 3)
    assert writer_instance.closed is True


def test_style_transfer_handles_missing_intro_segment(
    video_assets: VideoTestAssets,
    monkeypatch: MonkeyPatch,
) -> None:
    """When intro is skipped we pass default metadata into optimization."""
    output_dir = video_assets.new_output("intro_none")
    dummy_tensor = torch.rand(1, 3, 64, 64)

    monkeypatch.setattr(
        stv_runtime,
        "validate_input_paths",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        stv_runtime,
        "validate_parameters",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        stv_runtime,
        "setup_random_seed",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        stv_runtime,
        "setup_device",
        lambda *_a, **_k: torch.device("cpu"),
    )
    monkeypatch.setattr(
        stv_runtime,
        "setup_output_directory",
        lambda _p: Path(output_dir),
    )
    monkeypatch.setattr(
        stv_runtime,
        "save_outputs",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        stv_image_io,
        "load_image_to_tensor",
        lambda *_a, **_k: dummy_tensor,
    )
    monkeypatch.setattr(
        stv_core_model,
        "prepare_model_and_input",
        lambda *_a, **_k: ("model", dummy_tensor.clone(), "optimizer"),
    )

    captured: dict[str, object] = {}

    def init_hook(
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> tuple[torch.Tensor, dict[str, list[float]], float]:
        input_img = cast(torch.Tensor, args[1])
        captured["writer"] = kwargs.get("video_writer")
        captured["intro_last_frame"] = kwargs.get("intro_last_frame")
        captured["intro_crossfade_frames"] = kwargs.get(
            "intro_crossfade_frames",
        )
        return input_img, {"loss": []}, 0.0

    patch_runner(monkeypatch, init_hook=init_hook)

    class StubWriter:
        def __init__(self) -> None:
            self.frames: list[np.ndarray] = []
            self.closed = False

        def append_data(self, frame: np.ndarray) -> None:
            self.frames.append(frame)

        def close(self) -> None:
            self.closed = True

    writer_instance = StubWriter()
    monkeypatch.setattr(
        stv_video,
        "setup_video_writer",
        lambda *_a, **_k: writer_instance,
    )
    monkeypatch.setattr(
        stv_video,
        "prepare_intro_segment",
        lambda *_a, **_k: None,
    )

    paths = video_assets.inputs()
    cfg = video_assets.config(
        optimization={"steps": 1},
        video={"create_video": True, "save_every": 1},
        output={"output": output_dir},
    )
    expected_crossfade = 0

    result = stv_main.style_transfer(paths, cfg)

    assert isinstance(result, torch.Tensor)
    assert isinstance(captured["writer"], StubWriter)
    assert captured["intro_last_frame"] is None
    assert captured["intro_crossfade_frames"] == expected_crossfade
    assert writer_instance.closed is True


def test_style_transfer_appends_final_comparison_frame(
    video_assets: VideoTestAssets,
    monkeypatch: MonkeyPatch,
) -> None:
    """Final comparison frame should be appended when enabled."""
    output_dir = video_assets.new_output("final_compare_frame")
    dummy_tensor = torch.rand(1, 3, 64, 64)

    monkeypatch.setattr(
        stv_runtime,
        "validate_input_paths",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        stv_runtime,
        "validate_parameters",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        stv_runtime,
        "setup_random_seed",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        stv_runtime,
        "setup_device",
        lambda *_a, **_k: torch.device("cpu"),
    )
    monkeypatch.setattr(
        stv_runtime,
        "setup_output_directory",
        lambda _p: Path(output_dir),
    )
    monkeypatch.setattr(
        stv_runtime,
        "save_outputs",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        stv_image_io,
        "load_image_to_tensor",
        lambda *_a, **_k: dummy_tensor,
    )
    monkeypatch.setattr(
        stv_core_model,
        "prepare_model_and_input",
        lambda *_a, **_k: ("model", dummy_tensor.clone(), "optimizer"),
    )

    patch_runner(
        monkeypatch,
        run_result=(dummy_tensor.clone(), {"loss": []}, 0.0),
    )

    class StubWriter:
        def __init__(self) -> None:
            self.appended: list[np.ndarray] = []
            self.closed = False

        def append_data(self, frame: np.ndarray) -> None:
            self.appended.append(frame)

        def close(self) -> None:
            self.closed = True

    writer_instance = StubWriter()
    monkeypatch.setattr(
        stv_video,
        "setup_video_writer",
        lambda *_a, **_k: writer_instance,
    )
    monkeypatch.setattr(
        stv_video,
        "prepare_intro_segment",
        lambda *_a, **_k: None,
    )

    captured: dict[str, object] = {}

    def fake_append(
        cfg: VideoConfig,
        writer: StubWriter,
        content_path: str | Path,
        style_path: str | Path,
        last_frame: np.ndarray,
    ) -> None:
        captured["cfg"] = cfg
        captured["writer"] = writer
        captured["content"] = Path(content_path)
        captured["style"] = Path(style_path)
        captured["shape"] = last_frame.shape

    monkeypatch.setattr(
        stv_video,
        "append_final_comparison_frame",
        fake_append,
    )

    paths = video_assets.inputs()
    cfg = video_assets.config(
        optimization={"steps": 1},
        video={
            "create_video": True,
            "save_every": 1,
            "final_frame_compare": True,
        },
        output={"output": output_dir},
    )

    result = stv_main.style_transfer(paths, cfg)

    assert isinstance(result, torch.Tensor)
    assert captured["cfg"] is cfg.video
    assert captured["writer"] is writer_instance
    assert captured["content"] == Path(video_assets.content_image)
    assert captured["style"] == Path(video_assets.style_image)
    assert captured["shape"] == (64, 64, 3)
    assert writer_instance.closed is True


def test_final_only_disables_video(
    video_assets: VideoTestAssets,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test final_only mode disables video even if requested."""
    monkeypatch.setattr(
        stv_runtime,
        "validate_input_paths",
        lambda *_args, **_kwargs: None,
    )
    output_dir = video_assets.new_output("final_only_test")
    apply_mock(monkeypatch)

    paths = video_assets.inputs()
    cfg = video_assets.config(
        optimization={"steps": 1},
        video={
            "final_only": True,
            "create_video": True,
            "save_every": 1,
        },
        output={"output": output_dir},
    )

    result = stv_main.style_transfer(paths, cfg)

    assert isinstance(result, torch.Tensor)
    video = expected_video_path(
        output_dir,
        video_assets.content_image,
        video_assets.style_image,
    )
    assert not video.exists()
    assert Path(
        output_dir,
        f"stylized_{Path(video_assets.content_image).stem}_x_{Path(video_assets.style_image).stem}.png",
    ).exists()


def apply_mock(monkeypatch: MonkeyPatch) -> None:
    """Patch optimization and output functions with test mocks."""
    def init_hook(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[torch.Tensor, dict[str, list[float]], float]:
        img = torch.rand(1, 3, 100, 100)
        losses = {
            "style_loss": [10.0],
            "content_loss": [5.0],
            "total_loss": [15.0],
        }
        return img, losses, 0.1

    patch_runner(monkeypatch, init_hook=lambda args, kwargs: init_hook())

    def mock_save(
        _input_img: torch.Tensor,
        _loss_metrics: dict[str, list[float]],
        output_dir: Path,
        _elapsed: float,
        save_opts: SaveOptions,
    ) -> None:
        """
        Simulate saving output image and optional video file.

        Updated to match new save_outputs signature using SaveOptions.
        """
        image_path = output_dir / (
            f"stylized_{save_opts.content_name}_x_{save_opts.style_name}.png"
        )
        image_path.write_text("mock image")  # Avoid bytes warning
        if save_opts.video_created and save_opts.video_name:
            video_path = output_dir / save_opts.video_name
            video_path.write_text("mock video")  # Avoid bytes warning

    monkeypatch.setattr(stv_runtime, "save_outputs", mock_save)
def patch_runner(
    monkeypatch: MonkeyPatch,
    *,
    run_result: tuple[torch.Tensor, dict[str, list[float]], float] | None = None,
    init_hook: Callable[
        [tuple[object, ...], dict[str, object]],
        tuple[torch.Tensor, dict[str, list[float]], float] | None,
    ] | None = None,
) -> None:
    """
    Patch OptimizationRunner with a stub returning a predictable result.
    """

    class FakeRunner:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._result = run_result
            if init_hook is not None:
                hook_result = init_hook(args, kwargs)
                if hook_result is not None:
                    self._result = hook_result

        def run(self) -> tuple[torch.Tensor, dict[str, list[float]], float]:
            if self._result is None:
                msg = "No run result configured for FakeRunner."
                raise RuntimeError(msg)
            return self._result

    monkeypatch.setattr(stv_optimization, "OptimizationRunner", FakeRunner)
