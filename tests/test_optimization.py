"""
Tests for optimization logic in style_transfer_visualizer.

Covers:
- Single-step optimization execution
- Full optimization loop behavior
- Logging and metric collection (in-memory and CSV)
- Frame saving and loss stability
"""
import logging

import pytest
import torch
from _pytest.logging import LogCaptureFixture
from pytest_mock import MockerFixture
from torch import Tensor
from torch.optim import Optimizer

import style_transfer_visualizer.core_model as stv_core_model
import style_transfer_visualizer.image_io as stv_image_io
import style_transfer_visualizer.optimization as stv_optimization
from style_transfer_visualizer.config import StyleTransferConfig


@pytest.fixture
def setup_model_and_images(
    style_layers: list[int],
    content_layers: list[int],
    style_image: str,
    content_image: str,
    test_device: torch.device,
) -> tuple[torch.nn.Module, Tensor, Tensor, Tensor]:
    """Set up model and image tensors for integration testing."""
    style_img = stv_image_io.apply_transforms(
        stv_image_io.load_image(style_image), device=test_device,
        normalize=True,
    )
    content_img = stv_image_io.apply_transforms(
        stv_image_io.load_image(content_image), device=test_device,
        normalize=True,
    )
    model = stv_core_model.StyleContentModel(style_layers,
                                             content_layers).to(test_device)
    model.set_targets(style_img, content_img)
    input_img = content_img.clone().requires_grad_(True)  # noqa: FBT003
    return model, style_img, content_img, input_img


def test_prepare_image_for_output_denormalized() -> None:
    """Test output tensor is clamped and unchanged when normalize=False."""
    tensor = torch.rand(1, 3, 64, 64)
    out = stv_image_io.prepare_image_for_output(tensor, normalize=False)
    assert out.shape == tensor.shape
    assert torch.all(out >= 0)
    assert torch.all(out <= 1)


class TestOptimization:
    """Tests optimization steps, logging, and loop execution."""

    @pytest.mark.parametrize("opt_class",
                             [torch.optim.Adam, torch.optim.LBFGS])
    def test_manual_closure_executes(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        opt_class: type[Optimizer],
    ) -> None:
        """Test that optimizer closure executes correctly."""
        model, _, _, input_img = setup_model_and_images
        # noinspection PyArgumentList
        opt = opt_class(params=[input_img])  # pyright: ignore[reportCallIssue]

        def closure() -> float:
            opt.zero_grad()
            s, c = model(input_img)
            loss = sum(s, torch.tensor(0.0)) + sum(c, torch.tensor(0.0))
            loss.backward()
            return float(loss.item())

        result = opt.step(closure)
        assert isinstance(result, float)
        assert result > 0

    def test_optimization_step_basic(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        mocker: MockerFixture,
    ) -> None:
        """Test basic functionality of a single optimization step."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img], lr=0.01)

        # Build config used by StepContext
        config = StyleTransferConfig.model_validate({
            "optimization": {"style_w": 1e5, "content_w": 1.0,
                             "normalize": True},
            "video": {"save_every": 10},
        })
        progress = mocker.MagicMock()
        ctx = stv_optimization.StepContext(config=config,
                                           progress_bar=progress,
                                           video_writer=None, loss_metrics={
                                               "style_loss": [],
                                               "content_loss": [],
                                               "total_loss": [],
                                           }, loss_logger=None)
        loss = stv_optimization.optimization_step(
            model, input_img, optimizer, step=0, ctx=ctx,
        )
        assert isinstance(loss, float)
        assert len(ctx.loss_metrics["total_loss"]) == 1  # type: ignore[index]
        progress.update.assert_called_once()

    def test_optimization_step_save_frame(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        mocker: MockerFixture,
    ) -> None:
        """Test frame is saved if step is divisible by save_every."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])
        video = mocker.MagicMock()
        progress = mocker.MagicMock()

        config = StyleTransferConfig.model_validate({
            "optimization": {"style_w": 1e5, "content_w": 1.0,
                             "normalize": True},
            "video": {"save_every": 10},
        })
        ctx = stv_optimization.StepContext(config, progress, video, {
            "style_loss": [], "content_loss": [], "total_loss": [],
        })

        stv_optimization.optimization_step(
            model, input_img, optimizer, step=10, ctx=ctx,
        )
        assert video.append_data.called

    def test_non_finite_warning(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        mocker: MockerFixture,
        caplog: LogCaptureFixture,
    ) -> None:
        """Test that non-finite loss triggers warning log."""
        caplog.set_level(logging.WARNING)
        _, _, _, input_img = setup_model_and_images

        inf_tensor = torch.ones(1, requires_grad=True) * float("inf")
        nan_tensor = torch.ones(1, requires_grad=True) * float("nan")
        model = mocker.MagicMock()
        model.return_value = ([inf_tensor], [nan_tensor])

        opt = torch.optim.Adam([input_img])
        progress = mocker.MagicMock()
        config = StyleTransferConfig.model_validate({
            "optimization": {"style_w": 1e5, "content_w": 1.0,
                             "normalize": True},
            "video": {"save_every": 10},
        })
        ctx = stv_optimization.StepContext(config, progress, None, {
            "style_loss": [], "content_loss": [], "total_loss": [],
        })

        stv_optimization.optimization_step(
            model, input_img, opt, step=0, ctx=ctx,
        )

        assert "Non-finite" in caplog.text

    def test_prepare_image_normalize_respected(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        mocker: MockerFixture,
    ) -> None:
        """Test normalize=False is respected during output prep."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])
        progress = mocker.MagicMock()
        writer = mocker.MagicMock()

        patch = mocker.patch.object(stv_image_io, "prepare_image_for_output")
        patch.return_value = input_img.detach()

        config = StyleTransferConfig.model_validate({
            "optimization": {"style_w": 1e5, "content_w": 1.0,
                             "normalize": False},
            "video": {"save_every": 10},
        })
        ctx = stv_optimization.StepContext(config, progress, writer, {
            "style_loss": [], "content_loss": [], "total_loss": [],
        })

        stv_optimization.optimization_step(
            model, input_img, optimizer, step=10, ctx=ctx,
        )

        patch.assert_called_once()
        assert patch.call_args.kwargs["normalize"] is False

    def test_optimization_step_skips_frame(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        mocker: MockerFixture,
    ) -> None:
        """Test frame is not written when prepare_image returns None."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])
        mocker.patch.object(stv_image_io, "prepare_image_for_output",
                            return_value=None)
        video_writer = mocker.MagicMock()
        progress_bar = mocker.MagicMock()

        config = StyleTransferConfig.model_validate({
            "optimization": {"style_w": 1e5, "content_w": 1.0,
                             "normalize": True},
            "video": {"save_every": 10},
        })
        ctx = stv_optimization.StepContext(config, progress_bar, video_writer,
                                           { "style_loss": [],
                                             "content_loss": [],
                                             "total_loss": [],
                                             },
                                           )

        stv_optimization.optimization_step(
            model, input_img, optimizer, step=10, ctx=ctx,
        )

        video_writer.append_data.assert_not_called()

    def test_run_optimization_loop_execution(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
    ) -> None:
        """Test full optimization loop runs and returns valid output."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.LBFGS([input_img])

        config = StyleTransferConfig.model_validate({
            "optimization": {"steps": 2, "style_w": 1.0, "content_w": 1.0,
                             "normalize": True},
            "video": {"save_every": 1},
        })

        result_img, metrics, elapsed = stv_optimization.run_optimization_loop(
            model, input_img, optimizer, config=config, video_writer=None,
        )
        assert isinstance(result_img, torch.Tensor)
        assert isinstance(metrics, dict)
        assert "style_loss" in metrics
        assert isinstance(elapsed, float)
        assert elapsed >= 0

    def test_run_optimization_loop_uses_csv_logger(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        mocker: MockerFixture,
    ) -> None:
        """LossCSVLogger is initialized and used when log_loss is set."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])

        # Patch LossCSVLogger to mock actual file writing
        mock_logger = mocker.patch(
            "style_transfer_visualizer.optimization.LossCSVLogger",
            autospec=True,
        )

        config = StyleTransferConfig.model_validate({
            "optimization": {"steps": 2, "style_w": 1.0, "content_w": 1.0,
                             "normalize": True},
            "video": {"save_every": 1},
            "output": {"log_loss": "losses.csv", "log_every": 1},
        })

        _img, metrics, _elapsed = stv_optimization.run_optimization_loop(
            model, input_img, optimizer, config=config, video_writer=None,
        )

        # Assert CSV logger lifecycle
        mock_logger.assert_called_once_with("losses.csv", 1)
        assert mock_logger.return_value.log.called
        assert mock_logger.return_value.close.called
        # No in-memory metrics
        assert metrics == {}

    def test_run_optimization_loop_in_memory_metrics(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
    ) -> None:
        """Test metrics are stored in memory when CSV logging is off."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])

        config = StyleTransferConfig.model_validate({
            "optimization": {"steps": 2, "style_w": 1.0, "content_w": 1.0,
                             "normalize": True},
            "video": {"save_every": 1},
        })

        _img, metrics, _elapsed = stv_optimization.run_optimization_loop(
            model, input_img, optimizer, config=config, video_writer=None,
        )

        assert isinstance(metrics, dict)
        assert "style_loss" in metrics
        assert len(metrics["style_loss"]) > 0

    def test_csv_logger_initialization_failure_logs_error(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        mocker: MockerFixture,
        caplog: LogCaptureFixture,
    ) -> None:
        """Test that OSError is logged and no in-memory fallback occurs."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])

        # Patch LossCSVLogger to raise OSError
        mocker.patch("style_transfer_visualizer.optimization.LossCSVLogger",
                     side_effect=OSError("Mocked failure"))

        caplog.set_level("ERROR")

        config = StyleTransferConfig.model_validate({
            "optimization": {"steps": 1, "style_w": 1.0, "content_w": 1.0,
                             "normalize": True},
            "video": {"save_every": 1},
            "output": {"log_loss": "losses.csv", "log_every": 1},
        })

        # Run optimization loop with CSV logging (will fail to init)
        _img, metrics, _elapsed = stv_optimization.run_optimization_loop(
            model, input_img, optimizer, config=config, video_writer=None,
        )

        # Should NOT fallback to in-memory logging in current implementation
        assert metrics == {}
        assert "Failed to initialize CSV logging" in caplog.text

    def test_long_run_warning_when_no_csv_logging(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        caplog: LogCaptureFixture,
    ) -> None:
        """Test warning is logged for long runs without CSV logging."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])

        caplog.set_level("WARNING")

        config = StyleTransferConfig.model_validate({
            "optimization": {"steps": 2500, "style_w": 1.0, "content_w": 1.0,
                             "normalize": True},
            "video": {"save_every": 500},
        })

        stv_optimization.run_optimization_loop(
            model, input_img, optimizer, config=config, video_writer=None,
        )

        assert "Long run detected" in caplog.text
