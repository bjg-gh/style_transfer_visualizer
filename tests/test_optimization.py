from typing import Any, Generator, Tuple

import pytest
import torch
from torch import Tensor
from torch.optim import Optimizer

import style_transfer_visualizer.optimization as stv_optimization
import style_transfer_visualizer.image_io as stv_image_io
import style_transfer_visualizer.core_model as stv_core_model


@pytest.fixture
def setup_model_and_images(
    style_layers: list[int],
    content_layers: list[int],
    style_image: str,
    content_image: str,
    test_device: torch.device
) -> Generator[
    Tuple[torch.nn.Module, Tensor, Tensor, Tensor], None, None
]:
    """Set up model and image tensors for integration testing."""
    style_img = stv_image_io.apply_transforms(
        stv_image_io.load_image(style_image),
        normalize=True,
        device=test_device
    )
    content_img = stv_image_io.apply_transforms(
        stv_image_io.load_image(content_image),
        normalize=True,
        device=test_device
    )
    model = stv_core_model.StyleContentModel(style_layers,
                                  content_layers).to(test_device)
    model.set_targets(style_img, content_img)
    input_img = content_img.clone().requires_grad_(True)
    yield model, style_img, content_img, input_img


def test_prepare_image_for_output_denormalized():
    tensor = torch.rand(1, 3, 64, 64)
    out = stv_image_io.prepare_image_for_output(tensor, normalize=False)
    assert out.shape == tensor.shape
    assert torch.all(out >= 0) and torch.all(out <= 1)


class TestOptimization:
    @pytest.mark.parametrize("opt_class", [torch.optim.Adam,
                                           torch.optim.LBFGS])
    def test_manual_closure_executes(self,
                                     setup_model_and_images,
                                     opt_class: type[Optimizer]):
        """Test that optimizer closure executes correctly."""
        model, _, _, input_img = setup_model_and_images
        opt = opt_class([input_img])

        def closure() -> Tensor:
            opt.zero_grad()
            s, c = model(input_img)
            loss = sum(s) + sum(c)
            loss.backward()
            return loss.item()

        # noinspection PyTypeChecker
        result = opt.step(closure)
        assert isinstance(result, float)
        assert result > 0

    def test_optimization_step_basic(self,
                                     setup_model_and_images,
                                     mocker: Any):
        """Test basic functionality of a single optimization step."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img], lr=0.01)
        progress = mocker.MagicMock()
        metrics = {
            "style_loss": [],
            "content_loss": [],
            "total_loss": []
        }
        loss = stv_optimization.optimization_step(
            model, input_img, optimizer, 1e5, 1.0, metrics, 0, 10, None, True,
            progress
        )
        assert isinstance(loss, float)
        assert len(metrics["total_loss"]) == 1
        progress.update.assert_called_once()

    def test_optimization_step_save_frame(self,
                                           setup_model_and_images,
                                           mocker: Any):
        """Test frame is saved if step is divisible by save_every."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])
        progress = mocker.MagicMock()
        video = mocker.MagicMock()
        metrics = {
            "style_loss": [],
            "content_loss": [],
            "total_loss": []
        }
        stv_optimization.optimization_step(
            model, input_img, optimizer, 1e5, 1.0, metrics, 10, 10, video,
            True, progress
        )
        assert video.append_data.called

    def test_non_finite_warning(self,
                                setup_model_and_images,
                                mocker: Any,
                                caplog: Any):
        """Test that non-finite loss triggers warning log."""
        import logging
        caplog.set_level(logging.WARNING)
        _, _, _, input_img = setup_model_and_images

        inf_tensor = torch.ones(1, requires_grad=True) * float("inf")
        nan_tensor = torch.ones(1, requires_grad=True) * float("nan")
        model = mocker.MagicMock()
        model.return_value = ([inf_tensor], [nan_tensor])

        opt = torch.optim.Adam([input_img])
        progress = mocker.MagicMock()
        metrics = {"style_loss": [], "content_loss": [], "total_loss": []}
        stv_optimization.optimization_step(
            model, input_img, opt, 1e5, 1.0, metrics, 0, 10, None, True,
            progress
        )
        assert "Non-finite" in caplog.text

    def test_prepare_image_normalize_respected(self,
                                               setup_model_and_images,
                                               mocker: Any):
        """Test normalize=False is respected during output prep."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])
        progress = mocker.MagicMock()
        writer = mocker.MagicMock()
        patch = mocker.patch.object(stv_image_io, "prepare_image_for_output")
        patch.return_value = input_img.detach()
        stv_optimization.optimization_step(
            model, input_img, optimizer, 1e5, 1.0,
            {"style_loss": [], "content_loss": [], "total_loss": []},
            10, 10, writer, False, progress
        )
        patch.assert_called_once()
        _, norm_arg = patch.call_args[0]
        assert norm_arg is False

    def test_optimization_step_skips_frame(self,
                                           setup_model_and_images,
                                           mocker: Any):
        """Test frame is not written when prepare_image returns None."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])
        mocker.patch.object(stv_image_io, "prepare_image_for_output",
                            return_value=None)
        video_writer = mocker.MagicMock()
        progress_bar = mocker.MagicMock()
        metrics = {"style_loss": [], "content_loss": [], "total_loss": []}
        stv_optimization.optimization_step(
            model, input_img, optimizer, 1e5, 1.0, metrics, 10, 10,
            video_writer, True, progress_bar
        )
        video_writer.append_data.assert_not_called()

    def test_run_optimization_loop_execution(self,
                                             setup_model_and_images):
        """Test full optimization loop runs and returns valid output."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.LBFGS([input_img])
        result_img, metrics, elapsed = stv_optimization.run_optimization_loop(
            model, input_img, optimizer,
            steps=2, save_every=1, style_weight=1.0,
            content_weight=1.0, normalize=True, video_writer=None
        )
        assert isinstance(result_img, torch.Tensor)
        assert isinstance(metrics, dict)
        assert "style_loss" in metrics
        assert isinstance(elapsed, float)
        assert elapsed >= 0
