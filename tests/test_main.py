"""
Integration and functional tests for the style_transfer() pipeline.

Covers:
- Minimal end-to-end runs with mocked components
- Video creation toggles and parameter passing
- Final-only mode behavior
- Output validation and directory handling
"""
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from _pytest.monkeypatch import MonkeyPatch

import style_transfer_visualizer.core_model as stv_core_model
import style_transfer_visualizer.image_io as stv_image_io
import style_transfer_visualizer.main as stv_main
import style_transfer_visualizer.optimization as stv_optimization
import style_transfer_visualizer.utils as stv_utils
import style_transfer_visualizer.video as stv_video


def test_style_transfer_minimal(monkeypatch: MonkeyPatch) -> None:
    """Smoke test for style_transfer() with mocked internals."""
    dummy_tensor = torch.rand(1, 3, 256, 256)

    monkeypatch.setattr(
        stv_image_io, "load_image_to_tensor",
        lambda *_args, **_kwargs: dummy_tensor,
    )
    monkeypatch.setattr(
        stv_core_model, "prepare_model_and_input",
        lambda *_args, **_kwargs: ("model", dummy_tensor.clone(), "optimizer"),
    )
    monkeypatch.setattr(
        stv_optimization, "run_optimization_loop",
        lambda *_args, **_kwargs: (dummy_tensor.clone(),
                                   {"loss": [1.0]}, 3.14),
    )
    monkeypatch.setattr(
        stv_utils, "save_outputs",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        stv_utils, "setup_output_directory",
        lambda _x: Path("mock_output"),
    )
    monkeypatch.setattr(
        stv_utils, "validate_input_paths",
        lambda *_args, **_kwargs: None,
    )

    result = stv_main.style_transfer(
        content_path="dummy.jpg",
        style_path="dummy2.jpg",
        steps=10,
        style_weight=1.0,
        content_weight=1.0,
        learning_rate=0.1,
        device_name="cpu",
        init_method="random",
        normalize=True,
        seed=42,
        output_dir="output",
        final_only=False,
        plot_losses=True,
        create_video=False,
        save_every=5,
        fps=1,
        video_quality=5,
    )

    assert isinstance(result, torch.Tensor)
    assert result.shape == dummy_tensor.shape


def test_style_transfer_no_plot(monkeypatch: MonkeyPatch) -> None:
    """Test style_transfer() with plotting disabled."""
    dummy_tensor = torch.rand(1, 3, 256, 256)
    monkeypatch.setattr(
        stv_image_io, "load_image_to_tensor",
        lambda *_a, **_kw: dummy_tensor,
    )
    monkeypatch.setattr(
        stv_core_model, "prepare_model_and_input",
        lambda *_a, **_kw: ("model", dummy_tensor.clone(), "optimizer"),
    )
    monkeypatch.setattr(
        stv_optimization, "run_optimization_loop",
        lambda *_a, **_kw: (dummy_tensor.clone(), {"loss": [1.0]}, 3.14),
    )
    monkeypatch.setattr(
        stv_utils, "save_outputs",
        lambda *_a, **_kw: None,
    )
    monkeypatch.setattr(
        stv_utils, "setup_output_directory",
        lambda _: Path("mock_output").resolve().mkdir(
            parents=True, exist_ok=True) or Path("mock_output"),
    )
    monkeypatch.setattr(
        stv_utils, "validate_input_paths",
        lambda *_a, **_kw: None,
    )

    result = stv_main.style_transfer(
        content_path="dummy.jpg",
        style_path="dummy2.jpg",
        plot_losses=False,
    )
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
    test_dir: str,
    content_image: str,
    style_image: str,
    create_video_flag: bool,  # noqa: FBT001
    monkeypatch: MonkeyPatch,
) -> None:
    """Test whether video is created when flag is toggled."""
    output_dir = setup_test_directory(test_dir, "video_test")
    apply_mock(monkeypatch)

    result = stv_main.style_transfer(
        content_path=content_image,
        style_path=style_image,
        steps=1,
        device_name="cpu",
        output_dir=str(output_dir),
        create_video=create_video_flag,
        save_every=1,
        video_quality=1,
    )

    assert isinstance(result, torch.Tensor)
    video = expected_video_path(output_dir, content_image, style_image)
    if create_video_flag:
        assert video.exists()
    else:
        assert not video.exists()


@pytest.mark.parametrize(("fps", "quality"), [(10, 1), (24, 5), (30, 10)])
def test_video_params_passed(  # noqa: PLR0913
    test_dir: str,
    content_image: str,
    style_image: str,
    fps: int,
    quality: int,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test video FPS and quality are passed to writer."""
    output_dir = setup_test_directory(test_dir, "video_param_test")
    apply_mock(monkeypatch)

    calls: list[tuple[tuple[Path, str, int, int], dict[str, bool]]] = []
    orig_writer = stv_video.setup_video_writer

    def wrapped(
        output_path: Path,
        video_name: str,
        fps_wrapped: int,
        video_quality: int,
        *,
        create_video: bool,
    ) -> object:
        calls.append(((output_path, video_name, fps_wrapped, video_quality),
                      {"create_video": create_video}))
        return orig_writer(
            output_path,
            video_name,
            fps,
            video_quality,
            create_video=create_video,
        )

    monkeypatch.setattr(stv_video, "setup_video_writer", wrapped)

    stv_main.style_transfer(
        content_path=content_image,
        style_path=style_image,
        steps=1,
        device_name="cpu",
        output_dir=str(output_dir),
        create_video=True,
        save_every=1,
        fps=fps,
        video_quality=quality,
    )

    assert calls
    args, _ = calls[0]
    _, _, actual_fps, actual_quality = args

    assert actual_fps == fps
    assert actual_quality == quality


def test_final_only_disables_video(
    test_dir: str,
    content_image: str,
    style_image: str,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test final_only mode disables video creation even if requested."""
    monkeypatch.setattr(stv_utils, "validate_input_paths",
                        lambda *_args, **_kwargs: None)
    output_dir = setup_test_directory(test_dir, "final_only_test")
    apply_mock(monkeypatch)

    result = stv_main.style_transfer(
        content_path=content_image,
        style_path=style_image,
        steps=1,
        device_name="cpu",
        output_dir=str(output_dir),
        final_only=True,
        create_video=True,
        save_every=1,
    )

    assert isinstance(result, torch.Tensor)
    video = expected_video_path(output_dir, content_image, style_image)
    assert not video.exists()
    assert Path(
        output_dir,
        f"{Path(content_image).stem}_stylized_{Path(style_image).stem}.png",
    ).exists()


def create_mock_optimization() -> Callable[
    ..., tuple[torch.Tensor, dict[str, list[float]], float],
]:
    """
    Return a mock optimization loop result.

    Used to simulate the output of run_optimization_loop().
    """
    def mock_run(
        *_: object,
        **__: object,
    ) -> tuple[torch.Tensor, dict[str, list[float]], float]:
        img = torch.rand(1, 3, 100, 100)
        losses = {
            "style_loss": [10.0],
            "content_loss": [5.0],
            "total_loss": [15.0],
        }
        return img, losses, 0.1

    return mock_run


def apply_mock(monkeypatch: MonkeyPatch) -> None:
    """Patch optimization and output functions with test mocks."""
    monkeypatch.setattr(
        stv_optimization,
        "run_optimization_loop",
        create_mock_optimization(),
    )

    # noinspection PyUnusedLocal
    def mock_save(  # noqa: PLR0913
        *,
        input_img: "torch.Tensor",  # noqa: ARG001
        loss_metrics: dict[str, list[float]],  # noqa: ARG001
        output_dir: Path,
        elapsed: float,  # noqa: ARG001
        content_name: str,
        style_name: str,
        video_name: str,
        normalize: bool,  # noqa: ARG001
        video_created: bool,
        plot_losses: bool,  # noqa: ARG001
    ) -> None:
        """Simulate saving output image and optional video file."""
        image_path = output_dir / f"{content_name}_stylized_{style_name}.png"
        image_path.write_text("mock image")  # Avoid bytes warning

        if video_created and video_name:
            video_path = output_dir / video_name
            video_path.write_text("mock video")  # Avoid bytes warning

    monkeypatch.setattr(stv_utils, "save_outputs", mock_save)


def setup_test_directory(base_dir: str, sub_name: str) -> str:
    """Create a clean test output directory under a base path."""
    out_dir = Path(base_dir) / sub_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    return str(out_dir)
