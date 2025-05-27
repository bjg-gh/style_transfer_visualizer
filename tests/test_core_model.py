"""Tests for core model logic in style_transfer_visualizer.

Covers:
- Input initialization strategies
- Gram matrix properties
- StyleContentModel forward loss computation
- Optimization step and loop
"""

from typing import Any, cast, Generator, Tuple

import pytest
import torch
from torch import Tensor
from torch.optim import Optimizer
import style_transfer_visualizer as stv


@pytest.fixture
def setup_model_and_images(
    style_layers: list[int],
    content_layers: list[int],
    style_image: str,
    content_image: str,
    device: torch.device
) -> Generator[
    Tuple[torch.nn.Module, Tensor, Tensor, Tensor], None, None
]:
    """Set up model and image tensors for integration testing."""
    style_img = stv.apply_transforms(
        stv.load_image(style_image), normalize=True, device=device
    )
    content_img = stv.apply_transforms(
        stv.load_image(content_image), normalize=True, device=device
    )
    model = stv.StyleContentModel(style_layers, content_layers).to(device)
    model.set_targets(style_img, content_img)
    input_img = content_img.clone().requires_grad_(True)
    yield model, style_img, content_img, input_img


class TestInitializeInput:
    def test_content_method(self, sample_tensor: Tensor):
        """Test content-based initialization returns correct values."""
        result = stv.initialize_input(sample_tensor, "content")
        assert torch.allclose(result, sample_tensor)
        assert result.requires_grad
        assert not sample_tensor.requires_grad

    def test_random_method(self, sample_tensor: Tensor):
        """Test random initialization returns non-content values."""
        result = stv.initialize_input(sample_tensor, "random")
        assert result.shape == sample_tensor.shape
        assert not torch.allclose(result, sample_tensor)
        assert result.requires_grad

    def test_white_method(self, sample_tensor: Tensor):
        """Test white initialization returns ones tensor."""
        result = stv.initialize_input(sample_tensor, "white")
        assert torch.allclose(result, torch.ones_like(sample_tensor))
        assert result.requires_grad

    def test_invalid_method(self, sample_tensor: Tensor):
        """Test unknown init method raises ValueError."""
        with pytest.raises(ValueError):
            stv.initialize_input(sample_tensor, cast(Any, "invalid"))

    def test_invalid_input_type(self):
        """Test non-tensor input raises TypeError."""
        with pytest.raises(TypeError):
            stv.initialize_input(cast(Any, "not_a_tensor"), "content")

    @pytest.mark.parametrize("device_name", ["cpu", "cuda"])
    def test_device_preservation(self,
                                  sample_tensor: Tensor,
                                  device_name: str):
        """Test output remains on the same device as input tensor."""
        if device_name == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device_name)
        tensor = sample_tensor.to(device)
        for method in ["content", "random", "white"]:
            result = stv.initialize_input(
                tensor, cast(stv.InitMethod, method)
            )
            assert result.device.type == device.type


def test_gram_matrix_properties(sample_tensor: Tensor):
    """Test symmetry and shape of computed Gram matrix."""
    _, c, _, _ = sample_tensor.shape
    gram = stv.gram_matrix(sample_tensor)
    assert isinstance(gram, torch.Tensor)
    assert gram.shape == (c, c)
    assert torch.allclose(gram, gram.t())
    eigvals = torch.linalg.eigvals(gram).real  # type: ignore[attr-defined]
    assert torch.all(eigvals >= -1e-6)


class TestStyleContentModel:
    def test_forward_loss_accumulation(self,
                                       style_layers: list[int],
                                       content_layers: list[int]):
        """Test forward pass accumulates correct number of losses."""
        model = stv.StyleContentModel(style_layers, content_layers)
        input_tensor = torch.randn(1, 3, 128, 128)
        model.set_targets(input_tensor, input_tensor)
        s_losses, c_losses = model(input_tensor)
        assert len(s_losses) == len(style_layers)
        assert len(c_losses) == len(content_layers)
        for loss in s_losses + c_losses:
            assert loss.dim() == 0


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
        loss = stv.optimization_step(
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
        stv.optimization_step(
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
        stv.optimization_step(
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
        patch = mocker.patch(
            "style_transfer_visualizer.prepare_image_for_output"
        )
        patch.return_value = input_img.detach()
        stv.optimization_step(
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
        mocker.patch(
            "style_transfer_visualizer.prepare_image_for_output",
            return_value=None
        )
        video_writer = mocker.MagicMock()
        progress_bar = mocker.MagicMock()
        metrics = {"style_loss": [], "content_loss": [], "total_loss": []}
        stv.optimization_step(
            model, input_img, optimizer, 1e5, 1.0, metrics, 10, 10,
            video_writer, True, progress_bar
        )
        video_writer.append_data.assert_not_called()

    def test_run_optimization_loop_execution(self,
                                             setup_model_and_images):
        """Test full optimization loop runs and returns valid output."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.LBFGS([input_img])
        result_img, metrics, elapsed = stv.run_optimization_loop(
            model, input_img, optimizer,
            steps=2, save_every=1, style_weight=1.0,
            content_weight=1.0, normalize=True, video_writer=None
        )
        assert isinstance(result_img, torch.Tensor)
        assert isinstance(metrics, dict)
        assert "style_loss" in metrics
        assert isinstance(elapsed, float)
        assert elapsed >= 0
