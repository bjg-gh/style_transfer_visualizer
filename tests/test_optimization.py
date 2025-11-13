"""
Tests for optimization logic in style_transfer_visualizer.

Covers:
- Single-step optimization execution
- Full optimization loop behavior
- Logging and metric collection (in-memory and CSV)
- Frame saving, intro crossfade, and callback handling
"""

from collections.abc import Callable, Iterable
from typing import Any

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

pytestmark = pytest.mark.slow


class MultiProbeSGD(torch.optim.SGD):
    """Optimizer that invokes the closure multiple times per accepted step."""

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float = 0.1,
        probes: int = 3,
    ) -> None:
        super().__init__(params, lr=lr)
        self.probes = probes
        self.closure_calls = 0

    def step(
        self,
        closure: Callable[[], float] | None = None,
    ) -> float:  # type: ignore[override]
        if closure is None:
            result = super().step()
            return 0.0 if result is None else float(result)
        loss = 0.0
        for _ in range(self.probes):
            loss = closure()
            self.closure_calls += 1
        super().step()
        return loss


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


@pytest.fixture
def make_runner_config(
    make_style_transfer_config: Callable[..., StyleTransferConfig],
) -> Callable[..., StyleTransferConfig]:
    """
    Provide a helper for constructing StyleTransferConfig instances used by the
    optimization runner tests.
    """

    def _build(
        *,
        optimization: dict[str, Any] | None = None,
        video: dict[str, Any] | None = None,
        output: dict[str, Any] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> StyleTransferConfig:
        opt_section: dict[str, Any] = {
            "steps": 1,
            "style_w": 1.0,
            "content_w": 1.0,
            "normalize": True,
        }
        if optimization:
            opt_section.update(optimization)
        video_section = {"save_every": 10}
        if video:
            video_section.update(video)
        return make_style_transfer_config(
            optimization=opt_section,
            video=video_section,
            output=output,
            extras=extras,
        )

    return _build


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
        make_runner_config: Callable[..., StyleTransferConfig],
    ) -> None:
        """Test basic functionality of a single optimization step."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img], lr=0.01)

        config = make_runner_config(
            optimization={"steps": 1, "style_w": 1e5},
        )
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
        make_runner_config: Callable[..., StyleTransferConfig],
    ) -> None:
        """Test frame is saved if step is divisible by save_every."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])
        video = mocker.MagicMock()
        progress = mocker.MagicMock()
        progress.set_postfix = mocker.MagicMock()

        config = make_runner_config(
            optimization={"steps": 1, "style_w": 1e5},
            video={"save_every": 1},
        )

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
        make_runner_config: Callable[..., StyleTransferConfig],
    ) -> None:
        """Test the optimization runner returns expected tuple."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])

        config = make_runner_config(
            optimization={"steps": 2},
            video={"save_every": 5},
        )

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
        make_runner_config: Callable[..., StyleTransferConfig],
    ) -> None:
        """Test CSV logger is used when configured."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])

        csv_logger = mocker.MagicMock()
        mocker.patch(
            "style_transfer_visualizer.optimization.LossCSVLogger",
            return_value=csv_logger,
        )

        config = make_runner_config(
            optimization={"steps": 2},
            video={"save_every": 1},
            output={"log_loss": "loss.csv", "log_every": 1},
        )

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
        make_runner_config: Callable[..., StyleTransferConfig],
    ) -> None:
        """Test metrics are collected when CSV logging is disabled."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])

        config = make_runner_config(
            optimization={"steps": 3},
            video={"save_every": 1},
        )

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
        make_runner_config: Callable[..., StyleTransferConfig],
    ) -> None:
        """Ensure intro crossfade is invoked before the first saved frame."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])

        config = make_runner_config(
            video={"save_every": 1},
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

    def test_runner_emits_gif_frames_when_enabled(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        make_runner_config: Callable[..., StyleTransferConfig],
    ) -> None:
        """GIF collector should receive frames when configured."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])

        class MemoryCollector:
            def __init__(self) -> None:
                self.frames: list[np.ndarray] = []
                self._size: tuple[int, int] | None = None

            def append_data(self, frame: np.ndarray) -> None:
                rgb = np.asarray(frame, dtype=np.uint8)
                self._size = (rgb.shape[1], rgb.shape[0])
                self.frames.append(rgb)

            def close(self) -> None:
                return None

        gif_collector = MemoryCollector()
        config = make_runner_config(
            video={
                "save_every": 1,
                "create_video": False,
                "create_gif": True,
                "gif_include_intro": True,
            },
        )

        height, width = input_img.shape[-2:]
        intro_frame = np.zeros((height, width, 3), dtype=np.uint8)

        runner = stv_optimization.OptimizationRunner(
            model,
            input_img,
            config,
            optimizer=optimizer,
            gif_collector=gif_collector,
            intro_last_frame=intro_frame,
            intro_crossfade_frames=2,
        )

        runner.run()

        assert gif_collector.frames, "GIF collector should capture frames"

    def test_timelapse_frames_saved_once_per_step_with_multi_probe_optimizer(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        mocker: MockerFixture,
        make_runner_config: Callable[..., StyleTransferConfig],
    ) -> None:
        """Frame capture happens once per accepted step even with re-entrant closure."""
        model, _, _, input_img = setup_model_and_images
        optimizer = MultiProbeSGD([input_img], probes=4)

        config = make_runner_config(
            optimization={"steps": 2},
            video={"save_every": 1},
        )

        progress = mocker.MagicMock()
        progress.set_postfix = mocker.MagicMock()
        video_writer = mocker.MagicMock()

        prepare_mock = mocker.patch.object(
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
            video_writer=video_writer,
        )

        runner.run()

        assert (
            optimizer.closure_calls
            == config.optimization.steps * optimizer.probes
        )
        assert runner._closure_calls == optimizer.closure_calls  # noqa: SLF001
        assert runner._step_index == config.optimization.steps  # noqa: SLF001
        assert video_writer.append_data.call_count == config.optimization.steps
        assert prepare_mock.call_count == config.optimization.steps

    def test_intro_crossfade_runs_once_with_multi_probe_optimizer(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        mocker: MockerFixture,
        make_runner_config: Callable[..., StyleTransferConfig],
    ) -> None:
        """Intro crossfades (video and GIF) fire once per accepted step."""
        model, _, _, input_img = setup_model_and_images
        optimizer = MultiProbeSGD([input_img], probes=3)

        config = make_runner_config(
            optimization={"steps": 1},
            video={
                "save_every": 1,
                "intro_enabled": True,
                "gif_include_intro": True,
            },
        )

        video_writer = mocker.MagicMock()
        gif_collector = mocker.MagicMock()
        progress = mocker.MagicMock()
        progress.set_postfix = mocker.MagicMock()

        prepare_mock = mocker.patch.object(
            stv_image_io,
            "prepare_image_for_output",
            return_value=torch.rand_like(input_img),
        )

        height, width = input_img.shape[-2:]
        intro_frame = np.zeros((height, width, 3), dtype=np.uint8)

        crossfade_counts = {"video": 0, "gif": 0}

        def fake_crossfade(
            writer_arg: object,
            _start_frame: np.ndarray,
            _end_frame: np.ndarray,
            _frame_count: int,
        ) -> None:
            if writer_arg is video_writer:
                crossfade_counts["video"] += 1
            if writer_arg is gif_collector:
                crossfade_counts["gif"] += 1

        mocker.patch.object(
            stv_video,
            "append_crossfade",
            side_effect=fake_crossfade,
        )

        runner = stv_optimization.OptimizationRunner(
            model,
            input_img,
            config,
            optimizer=optimizer,
            progress_bar=progress,
            video_writer=video_writer,
            gif_collector=gif_collector,
            intro_last_frame=intro_frame,
            intro_crossfade_frames=3,
        )

        runner.run()

        assert runner._closure_calls == optimizer.closure_calls  # noqa: SLF001
        assert (
            optimizer.closure_calls
            == config.optimization.steps * optimizer.probes
        )
        assert crossfade_counts["video"] == config.optimization.steps
        assert crossfade_counts["gif"] == config.optimization.steps
        assert video_writer.append_data.call_count == config.optimization.steps
        assert gif_collector.append_data.call_count == config.optimization.steps
        assert prepare_mock.call_count == config.optimization.steps

    def test_runner_csv_logger_failure_logs_error(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        mocker: MockerFixture,
        caplog: LogCaptureFixture,
        make_runner_config: Callable[..., StyleTransferConfig],
    ) -> None:
        """Test that OSError is logged and no in-memory fallback occurs."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])

        mocker.patch(
            "style_transfer_visualizer.optimization.LossCSVLogger",
            side_effect=OSError("Mocked failure"),
        )

        caplog.set_level("ERROR")

        config = make_runner_config(
            video={"save_every": 1},
            output={"log_loss": "losses.csv", "log_every": 1},
        )

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
        make_runner_config: Callable[..., StyleTransferConfig],
    ) -> None:
        """Test warning is logged for long runs without CSV logging."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])

        caplog.set_level("WARNING")

        config = make_runner_config(
            optimization={"steps": 2500},
            video={"save_every": 500},
        )

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
        make_runner_config: Callable[..., StyleTransferConfig],
    ) -> None:
        """Ensure configured callbacks fire during optimization."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])

        config = make_runner_config()

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

    def test_log_optimization_summary_skips_when_no_steps(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        make_runner_config: Callable[..., StyleTransferConfig],
        caplog: LogCaptureFixture,
    ) -> None:
        """Summary logging is suppressed before any steps run."""
        model, _, _, input_img = setup_model_and_images
        config = make_runner_config()
        runner = stv_optimization.OptimizationRunner(
            model,
            input_img,
            config,
        )

        caplog.set_level("INFO")
        runner._log_optimization_summary()  # noqa: SLF001
        assert "Optimization finished" not in caplog.text

    def test_runner_raises_when_closure_skips_metrics(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        make_runner_config: Callable[..., StyleTransferConfig],
    ) -> None:
        """Run() surfaces an error when the closure omits metrics."""
        model, _, _, input_img = setup_model_and_images
        config = make_runner_config()
        optimizer = torch.optim.SGD([input_img])

        class BrokenRunner(stv_optimization.OptimizationRunner):
            def _closure(self) -> float:  # type: ignore[override]
                return 0.0

        runner = BrokenRunner(
            model,
            input_img,
            config,
            optimizer=optimizer,
        )

        with pytest.raises(RuntimeError, match="did not record metrics"):
            runner.run()

    def test_runner_rejects_conflicting_optimizer_args(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        make_runner_config: Callable[..., StyleTransferConfig],
    ) -> None:
        """Supplying optimizer and factory together is not allowed."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])
        config = make_runner_config()

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
        make_runner_config: Callable[..., StyleTransferConfig],
    ) -> None:
        """Accessing progress_bar before run() raises."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])
        config = make_runner_config()

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
        make_runner_config: Callable[..., StyleTransferConfig],
    ) -> None:
        """Delegates optimizer construction to the supplied factory."""
        model, _, _, input_img = setup_model_and_images
        config = make_runner_config(
            optimization={"lr": 0.5},
        )

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
        make_runner_config: Callable[..., StyleTransferConfig],
    ) -> None:
        """LBFGS is the default optimizer when none supplied."""
        model, _, _, input_img = setup_model_and_images
        config = make_runner_config()

        runner = stv_optimization.OptimizationRunner(
            model,
            input_img,
            config,
        )

        assert isinstance(runner.optimizer, torch.optim.LBFGS)

    def test_lbfgs_inner_loop_respects_config(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        make_runner_config: Callable[..., StyleTransferConfig],
    ) -> None:
        """LBFGS is instantiated with configured inner-loop bounds."""
        model, _, _, input_img = setup_model_and_images
        config = make_runner_config(
            optimization={
                "lbfgs_max_iter": 3,
                "lbfgs_max_eval": 2,
            },
        )

        runner = stv_optimization.OptimizationRunner(
            model,
            input_img,
            config,
        )

        params = runner.optimizer.param_groups[0]
        assert params["max_iter"] == 3  # noqa: PLR2004
        assert params["max_eval"] == 2  # noqa: PLR2004

    def test_logging_error_callback_invoked(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        mocker: MockerFixture,
        make_runner_config: Callable[..., StyleTransferConfig],
    ) -> None:
        """Callbacks receive logging initialisation errors."""
        model, _, _, input_img = setup_model_and_images

        mocker.patch(
            "style_transfer_visualizer.optimization.LossCSVLogger",
            side_effect=OSError("boom"),
        )

        config = make_runner_config(
            output={"log_loss": "losses.csv", "log_every": 1},
        )

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
        make_runner_config: Callable[..., StyleTransferConfig],
    ) -> None:
        """Closure exits early when all steps finished."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])
        config = make_runner_config()

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
        make_runner_config: Callable[..., StyleTransferConfig],
    ) -> None:
        """Non-finite metrics trigger warnings."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])
        config = make_runner_config()

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
        make_runner_config: Callable[..., StyleTransferConfig],
    ) -> None:
        """No frame is emitted when image preparation fails."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])
        progress = mocker.MagicMock()
        progress.set_postfix = mocker.MagicMock()
        config = make_runner_config(
            video={"save_every": 1},
        )

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

        metrics = stv_optimization.StepMetrics(
            step=1,
            style_loss=1.0,
            content_loss=1.0,
            total_loss=1.0,
        )
        runner._maybe_write_video_frame(metrics)  # noqa: SLF001

        writer.append_data.assert_not_called()
        progress.set_postfix.assert_not_called()

    def test_maybe_write_video_frame_invokes_callback(
        self,
        setup_model_and_images: tuple[torch.nn.Module, Tensor, Tensor, Tensor],
        mocker: MockerFixture,
        make_runner_config: Callable[..., StyleTransferConfig],
    ) -> None:
        """on_video_frame hook fires when a frame is saved."""
        model, _, _, input_img = setup_model_and_images
        optimizer = torch.optim.Adam([input_img])
        progress = mocker.MagicMock()
        progress.set_postfix = mocker.MagicMock()
        config = make_runner_config(
            video={"save_every": 1},
        )

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

        metrics = stv_optimization.StepMetrics(
            step=1,
            style_loss=1.0,
            content_loss=1.0,
            total_loss=1.0,
        )
        runner._maybe_write_video_frame(metrics)  # noqa: SLF001

        assert frames == [1]
        assert len(writer.frames) == 1
        progress.set_postfix.assert_called_once()
