import os
import shutil
from pathlib import Path
from typing import Any, Optional

import pytest
import torch

from style_transfer_visualizer.types import LossMetrics
import style_transfer_visualizer.main as stv_main
import style_transfer_visualizer.video as stv_video
import style_transfer_visualizer.core_model as stv_core_model
import style_transfer_visualizer.optimization as stv_optimization
import style_transfer_visualizer.utils as stv_utils
import style_transfer_visualizer.image_io as stv_image_io


def test_style_transfer_minimal(monkeypatch):
    """Smoke test for style_transfer() with mocked internals."""
    dummy_tensor = torch.rand(1, 3, 256, 256)

    monkeypatch.setattr(
        stv_image_io, "load_image_to_tensor",
        lambda *a, **kw: dummy_tensor
    )
    monkeypatch.setattr(
        stv_core_model, "prepare_model_and_input",
        lambda *a, **kw: ("model", dummy_tensor.clone(), "optimizer")
    )
    monkeypatch.setattr(
        stv_optimization, "run_optimization_loop",
        lambda *a, **kw: (dummy_tensor.clone(), {"loss": [1.0]}, 3.14)
    )
    monkeypatch.setattr(
        stv_utils, "save_outputs",
        lambda *a, **kw: None
    )
    monkeypatch.setattr(
        stv_utils, "setup_output_directory",
        lambda x: Path("mock_output")
    )
    monkeypatch.setattr(
        stv_utils, "validate_input_paths",
        lambda *a, **kw: None
    )

    result = stv_main.style_transfer(
        content_path="dummy.jpg",
        style_path="dummy2.jpg",
        output_dir="output",
        steps=10,
        save_every=5,
        style_weight=1.0,
        content_weight=1.0,
        learning_rate=0.1,
        fps=1,
        device_name="cpu",
        init_method="random",
        normalize=True,
        create_video=False,
        final_only=False,
        video_quality=5,
        seed=42,
    )

    assert isinstance(result, torch.Tensor)
    assert result.shape == dummy_tensor.shape


def expected_video_path(
    out_dir: str | Path,
    content_img: str,
    style_img: str
) -> Path:
    """Construct expected video file path from image names."""
    c = Path(content_img).stem
    s = Path(style_img).stem
    return Path(out_dir) / f"timelapse_{c}_x_{s}.mp4"


@pytest.mark.parametrize("create_video_flag", [True, False])
def test_video_creation_flag(test_dir, content_image, style_image,
                              create_video_flag, monkeypatch):
    """Test whether video is created when flag is toggled."""
    output_dir = setup_test_directory(test_dir, "video_test")
    apply_mock(monkeypatch)

    result = stv_main.style_transfer(
        content_path=content_image,
        style_path=style_image,
        output_dir=output_dir,
        steps=1,
        save_every=1,
        create_video=create_video_flag,
        device_name="cpu",
        video_quality=1
    )

    assert isinstance(result, torch.Tensor)
    video = expected_video_path(output_dir, content_image, style_image)
    if create_video_flag:
        assert video.exists()
    else:
        assert not video.exists()


@pytest.mark.parametrize("fps,quality", [(10, 1), (24, 5), (30, 10)])
def test_video_params_passed(test_dir, content_image, style_image,
                              fps, quality, monkeypatch):
    """Test video FPS and quality are passed to writer."""
    output_dir = setup_test_directory(test_dir, "video_param_test")
    apply_mock(monkeypatch)

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    orig_writer = stv_video.setup_video_writer

    def wrapped(*wrapped_args: Any, **wrapped_kwargs: Any) -> Any:
        calls.append((wrapped_args, wrapped_kwargs))
        return orig_writer(*wrapped_args, **wrapped_kwargs)

    monkeypatch.setattr(stv_video, "setup_video_writer", wrapped)

    stv_main.style_transfer(
        content_path=content_image,
        style_path=style_image,
        output_dir=output_dir,
        steps=1,
        save_every=1,
        create_video=True,
        device_name="cpu",
        fps=fps,
        video_quality=quality
    )

    assert calls
    args, _ = calls[0]
    _, _, actual_fps, actual_quality, _ = args

    assert actual_fps == fps
    assert actual_quality == quality


def test_final_only_disables_video(test_dir, content_image, style_image,
                                   monkeypatch):
    """Test final_only mode disables video creation even if requested."""
    monkeypatch.setattr(stv_utils, "validate_input_paths",
                        lambda *a, **kw: None)
    output_dir = setup_test_directory(test_dir, "final_only_test")
    apply_mock(monkeypatch)

    result = stv_main.style_transfer(
        content_path=content_image,
        style_path=style_image,
        output_dir=output_dir,
        steps=1,
        save_every=1,
        create_video=True,
        final_only=True,
        device_name="cpu"
    )

    assert isinstance(result, torch.Tensor)
    video = expected_video_path(output_dir, content_image, style_image)
    assert not video.exists()
    assert Path(
        output_dir,
        f"{Path(content_image).stem}_stylized_{Path(style_image).stem}.png"
    ).exists()


def create_mock_optimization():
    """Return a mock optimization loop result."""
    def mock_run(
        *_: Any,
        **__: Any
    ) -> tuple[torch.Tensor, dict[str, list[float]], float]:
        img = torch.rand(1, 3, 100, 100)
        losses = {
            "style_loss": [10.0],
            "content_loss": [5.0],
            "total_loss": [15.0]
        }
        return img, losses, 0.1
    return mock_run


def apply_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch optimization and output functions with test mocks."""
    monkeypatch.setattr(
        stv_optimization, "run_optimization_loop", create_mock_optimization()
    )

    def mock_save(
        input_img: torch.Tensor,
        loss_metrics: LossMetrics,
        output_dir: Path,
        elapsed: float,
        content_name: str,
        style_name: str,
        video_name: Optional[str] = None,
        normalize: bool = True,
        video_created: bool = True
    ) -> None:
        if video_created:
            video_path = Path(output_dir) / video_name
            video_path.write_bytes(b"mock video")
        Path(output_dir,
             f"{content_name}_stylized_{style_name}.png"
             ).write_bytes(b"mock image")

    monkeypatch.setattr(stv_utils, "save_outputs", mock_save)


def setup_test_directory(base_dir: str, sub_name: str) -> str:
    """Create a clean test output directory under a base path."""
    out_dir = os.path.join(base_dir, sub_name)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    return out_dir
