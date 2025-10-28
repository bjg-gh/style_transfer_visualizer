"""
Tests for optimization logic in style_transfer_visualizer.

Covers:
- Single-step optimization execution
- Full optimization loop behavior
- Logging and metric collection (in-memory and CSV)
- Frame saving, intro crossfade, and callback handling
"""

import numpy as np
import pytest
import torch
from _pytest.logging import LogCaptureFixture
from pytest_mock import MockerFixture
from torch import Tensor
from torch.optim import Optimizer

import style_transfer_visualizer.core_model as stv_core_model
import style_transfer_visualizer.image_io as stv_image_io
import style_transfer_visualizer.optimization as stv_optimization
import style_transfer_visualizer.video as stv_video
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
        stv_image_io.load_image(style_image),
        device=test_device,
        normalize=True,
    )
    content_img = stv_image_io.apply_transforms(
        stv_image_io.load_image(content_image),
        device=test_device,
        normalize=True,
    )
    model = stv_core_model.StyleContentModel(
        style_layers,
        content_layers,
    ).to(test_device)
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

    @pytest.mark.parametrize(
        "opt_class",
        [torch.optim.Adam, torch.optim.LBFGS],
    )
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

    def test_runner_basic_step(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        mocker: MockerFixture,
    ) -> None:
        """Test basic functionality of a single optimization step."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img], lr=0.01)

        config = StyleTransferConfig.model_validate({
            "optimization": {
                "steps": 1,
                "style_w": 1e5,
                "content_w": 1.0,
                "normalize": True,
            },
            "video": {"save_every": 10},
        })
        progress = mocker.MagicMock()
        progress.set_postfix = mocker.MagicMock()

        runner = stv_optimization.OptimizationRunner(
            model,
            input_img,
            config,
            optimizer=optimizer,
            progress_bar=progress,
        )

        _img, metrics, _elapsed = runner.run()

        assert isinstance(metrics, dict)
        assert len(metrics["total_loss"]) == 1
        progress.update.assert_called_once_with(1)

    def test_runner_saves_frame(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        mocker: MockerFixture,
    ) -> None:
        """Test frame is saved if step is divisible by save_every."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])
        video = mocker.MagicMock()
        progress = mocker.MagicMock()
        progress.set_postfix = mocker.MagicMock()

        config = StyleTransferConfig.model_validate({
            "optimization": {
                "steps": 1,
                "style_w": 1e5,
                "content_w": 1.0,
                "normalize": True,
            },
            "video": {"save_every": 1},
        })

        mocker.patch.object(
            stv_image_io,
            "prepare_image_for_output",
            return_value=torch.rand_like(input_img),
        )

        runner = stv_optimization.OptimizationRunner(
            model,
            input_img,
            config,
            optimizer=optimizer,
            progress_bar=progress,
            video_writer=video,
        )

        runner.run()

        video.append_data.assert_called_once()
        progress.set_postfix.assert_called_once()

    def test_runner_execution_returns_metrics(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
    ) -> None:
        """Test the optimization runner returns expected tuple."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])

        config = StyleTransferConfig.model_validate({
            "optimization": {
                "steps": 2,
                "style_w": 1.0,
                "content_w": 1.0,
                "normalize": True,
            },
            "video": {"save_every": 5},
        })

        runner = stv_optimization.OptimizationRunner(
            model,
            input_img,
            config,
            optimizer=optimizer,
        )

        result_img, metrics, elapsed = runner.run()

        assert torch.is_tensor(result_img)
        assert isinstance(metrics, dict)
        assert len(metrics["total_loss"]) == config.optimization.steps
        assert elapsed >= 0

    def test_runner_uses_csv_logger(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        mocker: MockerFixture,
    ) -> None:
        """Test CSV logger is used when configured."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])

        csv_logger = mocker.MagicMock()
        mocker.patch(
            "style_transfer_visualizer.optimization.LossCSVLogger",
            return_value=csv_logger,
        )

        config = StyleTransferConfig.model_validate({
            "optimization": {
                "steps": 2,
                "style_w": 1.0,
                "content_w": 1.0,
                "normalize": True,
            },
            "video": {"save_every": 1},
            "output": {"log_loss": "loss.csv", "log_every": 1},
        })

        runner = stv_optimization.OptimizationRunner(
            model,
            input_img,
            config,
            optimizer=optimizer,
        )

        _img, metrics, _elapsed = runner.run()

        assert metrics == {}  # CSV logging bypasses in-memory metrics
        assert csv_logger.log.call_count >= config.optimization.steps
        csv_logger.close.assert_called_once()

    def test_runner_in_memory_metrics(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
    ) -> None:
        """Test metrics are collected when CSV logging is disabled."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])

        config = StyleTransferConfig.model_validate({
            "optimization": {
                "steps": 3,
                "style_w": 1.0,
                "content_w": 1.0,
                "normalize": True,
            },
            "video": {"save_every": 1},
        })

        runner = stv_optimization.OptimizationRunner(
            model,
            input_img,
            config,
            optimizer=optimizer,
        )

        _img, metrics, _elapsed = runner.run()

        assert isinstance(metrics, dict)
        assert "style_loss" in metrics
        assert len(metrics["style_loss"]) == config.optimization.steps

    def test_runner_triggers_intro_crossfade(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        mocker: MockerFixture,
    ) -> None:
        """Ensure intro crossfade is invoked before the first saved frame."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])

        config = StyleTransferConfig.model_validate({
            "optimization": {
                "steps": 1,
                "style_w": 1.0,
                "content_w": 1.0,
                "normalize": True,
            },
            "video": {"save_every": 1},
        })

        class MemoryWriter:
            def __init__(self) -> None:
                self.frames: list[np.ndarray] = []
                self._size: tuple[int, int] | None = None

            def append_data(self, frame: np.ndarray) -> None:
                rgb = np.asarray(frame, dtype=np.uint8)
                self._size = (rgb.shape[1], rgb.shape[0])
                self.frames.append(rgb)

            def close(self) -> None:
                return None

        writer = MemoryWriter()
        calls: dict[str, object] = {}

        def fake_crossfade(
            writer_arg: MemoryWriter,
            _start_frame: np.ndarray,
            end_frame: np.ndarray,
            frame_count: int,
        ) -> None:
            calls["writer"] = writer_arg
            calls["frame_count"] = frame_count
            writer_arg.append_data(end_frame)

        mocker.patch.object(
            stv_video,
            "append_crossfade",
            side_effect=fake_crossfade,
        )

        intro_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        crossfade_frames = 4

        runner = stv_optimization.OptimizationRunner(
            model,
            input_img,
            config,
            optimizer=optimizer,
            video_writer=writer,
            intro_last_frame=intro_frame,
            intro_crossfade_frames=crossfade_frames,
        )

        runner.run()

        assert calls["writer"] is writer
        assert calls["frame_count"] == crossfade_frames
        expected_min_frames = 2
        assert len(writer.frames) >= expected_min_frames

    def test_runner_csv_logger_failure_logs_error(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        mocker: MockerFixture,
        caplog: LogCaptureFixture,
    ) -> None:
        """Test that OSError is logged and no in-memory fallback occurs."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])

        mocker.patch(
            "style_transfer_visualizer.optimization.LossCSVLogger",
            side_effect=OSError("Mocked failure"),
        )

        caplog.set_level("ERROR")

        config = StyleTransferConfig.model_validate({
            "optimization": {
                "steps": 1,
                "style_w": 1.0,
                "content_w": 1.0,
                "normalize": True,
            },
            "video": {"save_every": 1},
            "output": {"log_loss": "losses.csv", "log_every": 1},
        })

        runner = stv_optimization.OptimizationRunner(
            model,
            input_img,
            config,
            optimizer=optimizer,
        )

        _img, metrics, _elapsed = runner.run()

        assert metrics == {}
        assert "Failed to initialize CSV logging" in caplog.text

    def test_runner_long_run_warning_when_no_csv_logging(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        caplog: LogCaptureFixture,
    ) -> None:
        """Test warning is logged for long runs without CSV logging."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])

        caplog.set_level("WARNING")

        config = StyleTransferConfig.model_validate({
            "optimization": {
                "steps": 2500,
                "style_w": 1.0,
                "content_w": 1.0,
                "normalize": True,
            },
            "video": {"save_every": 500},
        })

        stv_optimization.OptimizationRunner(
            model,
            input_img,
            config,
            optimizer=optimizer,
        )

        assert "Long run detected" in caplog.text

    def test_runner_callbacks_are_invoked(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
    ) -> None:
        """Ensure configured callbacks fire during optimization."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])

        config = StyleTransferConfig.model_validate({
            "optimization": {
                "steps": 1,
                "style_w": 1.0,
                "content_w": 1.0,
                "normalize": True,
            },
            "video": {"save_every": 10},
        })

        started: list[int] = []
        ended: list[float] = []

        callbacks = stv_optimization.OptimizationCallbacks(
            on_step_start=lambda step: started.append(step),
            on_step_end=lambda metrics: ended.append(metrics.total_loss),
        )

        runner = stv_optimization.OptimizationRunner(
            model,
            input_img,
            config,
            optimizer=optimizer,
            callbacks=callbacks,
        )

        runner.run()

        assert started == [1]
        assert len(ended) == 1

    def test_runner_rejects_conflicting_optimizer_args(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
    ) -> None:
        """Supplying optimizer and factory together is not allowed."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])
        config = StyleTransferConfig.model_validate({
            "optimization": {
                "steps": 1,
                "style_w": 1.0,
                "content_w": 1.0,
                "normalize": True,
            },
        })

        def factory(tensor: torch.Tensor) -> Optimizer:
            return torch.optim.Adam([tensor])

        with pytest.raises(
            ValueError,
            match="Provide either optimizer or optimizer_factory",
        ):
            stv_optimization.OptimizationRunner(
                model,
                input_img,
                config,
                optimizer=optimizer,
                optimizer_factory=factory,
            )

    def test_progress_bar_property_requires_initialisation(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
    ) -> None:
        """Accessing progress_bar before run() raises."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])
        config = StyleTransferConfig.model_validate({
            "optimization": {
                "steps": 1,
                "style_w": 1.0,
                "content_w": 1.0,
                "normalize": True,
            },
        })

        runner = stv_optimization.OptimizationRunner(
            model,
            input_img,
            config,
            optimizer=optimizer,
        )

        with pytest.raises(RuntimeError):
            _ = runner.progress_bar

    def test_custom_optimizer_factory_is_used(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
    ) -> None:
        """Delegates optimizer construction to the supplied factory."""
        model, _, _, input_img = setup_model_and_images
        config = StyleTransferConfig.model_validate({
            "optimization": {
                "steps": 1,
                "style_w": 1.0,
                "content_w": 1.0,
                "lr": 0.5,
                "normalize": True,
            },
        })

        created: list[Optimizer] = []

        def factory(tensor: torch.Tensor) -> Optimizer:
            opt = torch.optim.SGD([tensor], lr=0.5)
            created.append(opt)
            return opt

        runner = stv_optimization.OptimizationRunner(
            model,
            input_img,
            config,
            optimizer_factory=factory,
        )

        assert isinstance(runner.optimizer, torch.optim.SGD)
        assert runner.optimizer is created[0]

    def test_default_optimizer_is_lbfgs(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
    ) -> None:
        """LBFGS is the default optimizer when none supplied."""
        model, _, _, input_img = setup_model_and_images
        config = StyleTransferConfig.model_validate({
            "optimization": {
                "steps": 1,
                "style_w": 1.0,
                "content_w": 1.0,
                "normalize": True,
            },
        })

        runner = stv_optimization.OptimizationRunner(
            model,
            input_img,
            config,
        )

        assert isinstance(runner.optimizer, torch.optim.LBFGS)

    def test_logging_error_callback_invoked(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        mocker: MockerFixture,
    ) -> None:
        """Callbacks receive logging initialisation errors."""
        model, _, _, input_img = setup_model_and_images

        mocker.patch(
            "style_transfer_visualizer.optimization.LossCSVLogger",
            side_effect=OSError("boom"),
        )

        config = StyleTransferConfig.model_validate({
            "optimization": {
                "steps": 1,
                "style_w": 1.0,
                "content_w": 1.0,
                "normalize": True,
            },
            "output": {"log_loss": "losses.csv", "log_every": 1},
        })

        captured: list[Exception] = []
        callbacks = stv_optimization.OptimizationCallbacks(
            on_logging_error=captured.append,
        )

        stv_optimization.OptimizationRunner(
            model,
            input_img,
            config,
            callbacks=callbacks,
        )

        assert captured
        assert isinstance(captured[0], OSError)

    def test_closure_returns_last_loss_once_complete(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
    ) -> None:
        """Closure exits early when all steps finished."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])
        config = StyleTransferConfig.model_validate({
            "optimization": {
                "steps": 1,
                "style_w": 1.0,
                "content_w": 1.0,
                "normalize": True,
            },
        })

        runner = stv_optimization.OptimizationRunner(
            model,
            input_img,
            config,
            optimizer=optimizer,
        )
        runner._step_index = runner.total_steps  # noqa: SLF001
        expected_loss = 3.21
        runner.last_loss = expected_loss
        assert runner._closure() == expected_loss  # noqa: SLF001

        runner.last_loss = None
        assert runner._closure() == 0.0  # noqa: SLF001

    def test_check_finite_logs_warnings(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        caplog: LogCaptureFixture,
    ) -> None:
        """Non-finite metrics trigger warnings."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])
        config = StyleTransferConfig.model_validate({
            "optimization": {
                "steps": 1,
                "style_w": 1.0,
                "content_w": 1.0,
                "normalize": True,
            },
        })

        runner = stv_optimization.OptimizationRunner(
            model,
            input_img,
            config,
            optimizer=optimizer,
        )

        caplog.set_level("WARNING")
        nan_tensor = torch.tensor(float("nan"))
        runner._check_finite(  # noqa: SLF001
            nan_tensor,
            nan_tensor,
            nan_tensor,
            step_idx=5,
            style_components=[float("nan")],
        )

        assert "Non-finite style score" in caplog.text
        assert "Non-finite content score" in caplog.text
        assert "Non-finite total loss" in caplog.text

    def test_maybe_write_video_frame_handles_missing_tensor(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        mocker: MockerFixture,
    ) -> None:
        """No frame is emitted when image preparation fails."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])
        progress = mocker.MagicMock()
        progress.set_postfix = mocker.MagicMock()
        config = StyleTransferConfig.model_validate({
            "optimization": {
                "steps": 1,
                "style_w": 1.0,
                "content_w": 1.0,
                "normalize": True,
            },
            "video": {"save_every": 1},
        })

        mocker.patch.object(
            stv_image_io,
            "prepare_image_for_output",
            return_value=None,
        )
        writer = mocker.MagicMock()

        runner = stv_optimization.OptimizationRunner(
            model,
            input_img,
            config,
            optimizer=optimizer,
            progress_bar=progress,
            video_writer=writer,
        )

        value = torch.tensor(1.0)
        runner._maybe_write_video_frame(1, value, value, value)  # noqa: SLF001

        writer.append_data.assert_not_called()
        progress.set_postfix.assert_not_called()

    def test_maybe_write_video_frame_invokes_callback(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        mocker: MockerFixture,
    ) -> None:
        """on_video_frame hook fires when a frame is saved."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])
        progress = mocker.MagicMock()
        progress.set_postfix = mocker.MagicMock()

        config = StyleTransferConfig.model_validate({
            "optimization": {
                "steps": 1,
                "style_w": 1.0,
                "content_w": 1.0,
                "normalize": True,
            },
            "video": {"save_every": 1},
        })

        mocker.patch.object(
            stv_image_io,
            "prepare_image_for_output",
            return_value=torch.rand_like(input_img),
        )

        frames: list[int] = []

        callbacks = stv_optimization.OptimizationCallbacks(
            on_video_frame=lambda _frame, step: frames.append(step),
        )

        class MemoryWriter:
            def __init__(self) -> None:
                self.frames: list[np.ndarray] = []
                self._size: tuple[int, int] | None = None

            def append_data(self, frame: np.ndarray) -> None:
                rgb = np.asarray(frame, dtype=np.uint8)
                self._size = (rgb.shape[1], rgb.shape[0])
                self.frames.append(rgb)

            def close(self) -> None:
                return None

        writer = MemoryWriter()

        runner = stv_optimization.OptimizationRunner(
            model,
            input_img,
            config,
            optimizer=optimizer,
            progress_bar=progress,
            video_writer=writer,
            callbacks=callbacks,
        )

        value = torch.tensor(1.0)
        runner._maybe_write_video_frame(1, value, value, value)  # noqa: SLF001

        assert frames == [1]
        assert len(writer.frames) == 1
        progress.set_postfix.assert_called_once()
