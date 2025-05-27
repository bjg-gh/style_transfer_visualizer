"""Tests for video output and image saving in style_transfer_visualizer.

Covers:
- Timelapse video creation flag behavior
- Parameter propagation to video writer
- Final image saving logic
- Output normalization and NaN/infinity handling
"""

import os
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
import style_transfer_visualizer as stv


def setup_test_directory(base_dir: str, sub_name: str) -> str:
    """Create a clean test output directory under a base path."""
    out_dir = os.path.join(base_dir, sub_name)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    return out_dir


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
        stv, "run_optimization_loop", create_mock_optimization()
    )

    def mock_save(_img: torch.Tensor, _losses: dict[str, list[float]],
                  output_path: Path, _elapsed: float,
                  c_name: str, s_name: str, v_name: str,
                  _normalize: bool, create_vid: bool) -> None:
        if create_vid:
            video_path = Path(output_path) / v_name
            video_path.write_bytes(b"mock video")
        Path(output_path, f"{c_name}_stylized_{s_name}.png").write_bytes(
            b"mock image"
        )

    monkeypatch.setattr(stv, "save_outputs", mock_save)


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

    result = stv.style_transfer(
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
    orig_writer = stv.setup_video_writer

    def wrapped(*wrapped_args: Any, **wrapped_kwargs: Any) -> Any:
        calls.append((wrapped_args, wrapped_kwargs))
        return orig_writer(*wrapped_args, **wrapped_kwargs)

    monkeypatch.setattr(stv, "setup_video_writer", wrapped)

    stv.style_transfer(
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
    output_dir = setup_test_directory(test_dir, "final_only_test")
    apply_mock(monkeypatch)

    result = stv.style_transfer(
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


class TestImageOutputPreparation:
    def test_prepare_image_with_normalization(self):
        """Test normalization and clamping behavior."""
        tensor = torch.tensor([
            [[-3.0, -2.0, -1.0], [0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        ]).view(1, 1, 3, 3).repeat(1, 3, 1, 1)

        out = stv.prepare_image_for_output(tensor, normalize=True)
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert not torch.allclose(out, tensor.clamp(0, 1))
        assert out.shape == tensor.shape

    def test_prepare_image_without_normalization(self):
        """Test clamping behavior without normalization."""
        tensor = torch.tensor([[-0.5, 0.2, 0.7], [0.0, 1.0, 1.5]])
        tensor = tensor.view(1, 1, 2, 3).repeat(1, 3, 1, 1)
        out = stv.prepare_image_for_output(tensor, normalize=False)
        assert torch.allclose(out, tensor.clamp(0, 1))
        assert out.shape == tensor.shape

    def test_extreme_values(self):
        """Test clamping for large magnitude values."""
        tensor = torch.tensor([[-100.0, -50.0, 0.0], [1.0, 50.0, 100.0]])
        tensor = tensor.view(1, 1, 2, 3).repeat(1, 3, 1, 1)
        for norm in [True, False]:
            out = stv.prepare_image_for_output(tensor, normalize=norm)
            assert out.min() >= 0.0 and out.max() <= 1.0
            assert out.shape == tensor.shape

    @pytest.mark.parametrize("device_name", ["cpu", "cuda"])
    def test_device_preserved(self, device_name: str):
        """Test output remains on the original device."""
        if device_name == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device_name)
        tensor = torch.rand(1, 3, 10, 10).to(device)
        for norm in [True, False]:
            out = stv.prepare_image_for_output(tensor, normalize=norm)
            assert out.device.type == device.type

    def test_batch_handling(self):
        """Test image output handles batch size > 1."""
        batch = torch.rand(2, 3, 10, 10)
        out = stv.prepare_image_for_output(batch, normalize=True)
        assert out.shape == batch.shape
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_nan_handling(self):
        """Test NaN and infinity values are sanitized."""
        tensor = torch.tensor([[[[float("nan"),
                                   float("inf"),
                                   -float("inf")]]]]).repeat(1, 3, 1, 1)
        out = stv.prepare_image_for_output(tensor, normalize=False)
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert out.shape == tensor.shape


class TestSaveOutputs:
    def test_creates_final_image(self, output_dir: Path):
        """Test final image is saved correctly."""
        input_img = torch.rand(1, 3, 64, 64)
        loss_metrics = {
            "style_loss": [1.0],
            "content_loss": [0.5],
            "total_loss": [1.5]
        }

        stv.save_outputs(
            input_img=input_img,
            loss_metrics=loss_metrics,
            output_dir=output_dir,
            elapsed=12.3,
            content_name="dog",
            style_name="mosaic",
            video_name="timelapse_dog_x_mosaic.mp4",
            normalize=False,
            video_created=True
        )

        final_path = output_dir / "stylized_dog_x_mosaic.png"
        assert final_path.exists()
        assert final_path.stat().st_size > 0

    def test_logs_video_path(
        self,
        output_dir: Path,
        caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test video path logging message is recorded."""
        input_img = torch.rand(1, 3, 64, 64)
        loss_metrics = {
            "style_loss": [1.0],
            "content_loss": [0.5],
            "total_loss": [1.5]
        }

        caplog.set_level("INFO")
        stv.save_outputs(
            input_img=input_img,
            loss_metrics=loss_metrics,
            output_dir=output_dir,
            elapsed=5.0,
            content_name="cat",
            style_name="wave",
            video_name="timelapse_cat_x_wave.mp4",
            normalize=False,
            video_created=True
        )

        assert "Video saved to:" in caplog.text

    def test_logs_creation(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture
    ):
        """Force logger.info for directory creation to execute."""
        caplog.set_level("INFO")

        # Force .exists() to return False so the mkdir logic runs
        class CustomPath(Path):
            def exists(self, **_kwargs) -> bool:
                return False

        custom_path = CustomPath(tmp_path)

        input_img = torch.rand(1, 3, 64, 64)
        loss_metrics = {"style_loss": [1.0], "content_loss": [0.5],
                        "total_loss": [1.5]}

        stv.save_outputs(
            input_img=input_img,
            loss_metrics=loss_metrics,
            output_dir=custom_path,
            elapsed=1.23,
            content_name="test",
            style_name="coverage",
            video_name="timelapse_test_x_coverage.mp4",
            normalize=False,
            video_created=True
        )

        assert "Created output directory" in caplog.text

    def test_handles_fallback(self, caplog: pytest.LogCaptureFixture):
        """Simulate failure to create output directory and use fallback."""
        class BrokenPath(Path):
            def mkdir(self, *args: Any, **kwargs: Any) -> None:
                raise PermissionError("Mocked mkdir failure")

        fallback_path = Path.cwd() / "style_transfer_output"
        if fallback_path.exists():
            for file in fallback_path.iterdir():
                file.unlink()
            fallback_path.rmdir()

        input_img = torch.rand(1, 3, 64, 64)
        loss_metrics = {
            "style_loss": [1.0],
            "content_loss": [0.5],
            "total_loss": [1.5]
        }

        stv.save_outputs(
            input_img=input_img,
            loss_metrics=loss_metrics,
            output_dir=BrokenPath("/bad/path/"),
            elapsed=1.0,
            content_name="test",
            style_name="fallback",
            video_name=None,
            normalize=False,
            video_created=False
        )

        fallback_img = fallback_path / "stylized_test_x_fallback.png"
        assert fallback_img.exists()
        assert "Mocked mkdir failure" in caplog.text

        # Cleanup
        if fallback_path.exists():
            shutil.rmtree(fallback_path)
