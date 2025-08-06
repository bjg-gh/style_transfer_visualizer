"""
Tests for the updated CLI parser and execution logic.

These tests verify correct CLI parsing, config fallback behavior,
flag handling, and main entry point integration.

Modules tested:
- build_arg_parser()
- run_from_args()
- main()

Simulates CLI usage with monkeypatching and verifies end-to-end flow.
"""

import argparse
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest
import torch
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from PIL import Image

import style_transfer_visualizer.cli as stv_cli
import style_transfer_visualizer.main as stv_main
from style_transfer_visualizer.config_defaults import (
    DEFAULT_CONTENT_WEIGHT,
    DEFAULT_DEVICE,
    DEFAULT_FPS,
    DEFAULT_INIT_METHOD,
    DEFAULT_LEARNING_RATE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SAVE_EVERY,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_STYLE_WEIGHT,
    DEFAULT_VIDEO_QUALITY,
)


class TestCLIArgumentParsing:
    """Unit tests for CLI flag parsing and layer argument handling."""

    def test_flag_parsing(self) -> None:
        """Test that boolean flags are parsed correctly."""
        parser = stv_cli.build_arg_parser()
        args = parser.parse_args([
            "--content", "c.jpg",
            "--style", "s.jpg",
            "--no-normalize",
            "--no-video",
            "--final-only",
            "--no-plot",
        ])

        assert args.no_normalize is True
        assert args.no_video is True
        assert args.final_only is True
        assert args.no_plot is True

    def test_required_args_missing(self, monkeypatch: MonkeyPatch) -> None:
        """Test that missing required arguments triggers SystemExit."""
        monkeypatch.setattr(sys, "argv", ["prog"])
        with pytest.raises(SystemExit):
            stv_cli.main()

    @pytest.mark.parametrize(
        ("style_str", "content_str", "expected_style", "expected_content"), [
            ("0,1,2", "3", [0, 1, 2], [3]),
            ("5", "10,15", [5], [10, 15]),
        ])
    def test_layer_args_are_parsed(
        self,
        monkeypatch: MonkeyPatch,
        style_str: str,
        content_str: str,
        expected_style: list[int],
        expected_content: list[int],
    ) -> None:
        """Verify CLI layer flags are parsed and forwarded correctly."""
        args = argparse.Namespace(
            content="c.jpg",
            style="s.jpg",
            config=None,
            validate_config_only=False,
            output="out",
            steps=100,
            save_every=5,
            style_w=1.0,
            content_w=1.0,
            lr=1.0,
            init_method="random",
            no_normalize=False,
            no_video=False,
            final_only=False,
            quality=9,
            fps=10,
            seed=42,
            device="cpu",
            style_layers=style_str,
            content_layers=content_str,
        )

        captured: dict[str, Any] = {}
        monkeypatch.setattr(stv_main, "style_transfer",
                            lambda **kwargs: captured.update(
                                kwargs) or torch.rand(1))
        monkeypatch.setattr(stv_cli, "log_parameters", lambda *_: None)

        stv_cli.run_from_args(args)
        assert captured["style_layers"] == expected_style
        assert captured["content_layers"] == expected_content

    def test_log_loss_and_log_every_flags(self) -> None:
        """Test that --log-loss and --log-every are parsed correctly."""
        parser = stv_cli.build_arg_parser()
        args = parser.parse_args([
            "--content", "cat.jpg",
            "--style", "mosaic.jpg",
            "--log-loss", "losses.csv",
            "--log-every", "25",
        ])

        assert args.log_loss == "losses.csv"
        assert args.log_every == 25  # noqa: PLR2004


class TestCLIRunFromArgs:
    """Tests config loading, CLI overrides, and loss plotting logic."""

    def test_config_only_mode(
        self,
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Test --validate-config-only short-circuits the run."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[output]\noutput = 'abc'")

        args = argparse.Namespace(
            config=str(config_path),
            validate_config_only=True,
        )

        exit_called = {}

        def fake_exit(code: int=0) -> None:
            exit_called["code"] = code
            raise SystemExit

        monkeypatch.setattr(stv_cli.sys, "exit", fake_exit)
        monkeypatch.setattr(stv_cli, "log_parameters", lambda *_: None)

        with pytest.raises(SystemExit):
            stv_cli.run_from_args(args)

        assert exit_called["code"] == 0

    def test_args_override_config(self, monkeypatch: MonkeyPatch) -> None:
        """Test CLI arguments override config values."""
        args = argparse.Namespace(
            content="cat.jpg",
            style="wave.jpg",
            config=None,
            validate_config_only=False,
            output="out",
            steps=123,
            save_every=10,
            style_w=1.2,
            content_w=3.4,
            lr=0.5,
            fps=20,
            init_method="white",
            no_normalize=False,
            no_video=False,
            final_only=True,
            quality=7,
            seed=123,
            device="cpu",
        )

        captured = {}
        monkeypatch.setattr(stv_main, "style_transfer",
            lambda **kwargs: captured.update(kwargs) or torch.rand(1))
        monkeypatch.setattr(stv_cli, "log_parameters", lambda *_: None)

        stv_cli.run_from_args(args)
        assert captured["steps"] == 123  # noqa: PLR2004
        assert captured["normalize"] is True
        assert captured["create_video"] is True

    def test_flags_flip_behavior(self, monkeypatch: MonkeyPatch) -> None:
        """Test that --no-* flags flip behavior correctly."""
        args = argparse.Namespace(
            content="cat.jpg",
            style="wave.jpg",
            config=None,
            validate_config_only=False,
            output=DEFAULT_OUTPUT_DIR,
            steps=DEFAULT_STEPS,
            save_every=DEFAULT_SAVE_EVERY,
            style_w=DEFAULT_STYLE_WEIGHT,
            content_w=DEFAULT_CONTENT_WEIGHT,
            lr=DEFAULT_LEARNING_RATE,
            fps=DEFAULT_FPS,
            init_method=DEFAULT_INIT_METHOD,
            no_normalize=True,
            no_video=True,
            final_only=False,
            quality=DEFAULT_VIDEO_QUALITY,
            seed=DEFAULT_SEED,
            device=DEFAULT_DEVICE,
        )

        captured = {}
        monkeypatch.setattr(stv_main, "style_transfer",
            lambda **kwargs: captured.update(kwargs) or torch.rand(1))
        monkeypatch.setattr(stv_cli, "log_parameters", lambda *_: None)

        stv_cli.run_from_args(args)
        assert captured["normalize"] is False
        assert captured["create_video"] is False


    def test_run_from_args_config_not_validating(
        self, monkeypatch: MonkeyPatch, tmp_path: Path,
    ) -> None:
        """Test config loads but validate_config_only is False."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[output]
output = "config_out"
[optimization]
steps = 123
style_w = 1.0
content_w = 1.0
lr = 1.0
init_method = "random"
seed = 42
normalize = true
[video]
save_every = 5
fps = 15
quality = 9
create_video = true
final_only = false
[hardware]
device = "cuda"
""")

        args = argparse.Namespace(
            content="cat.jpg",
            style="s.jpg",
            config=str(config_path),
            validate_config_only=False,
        )

        captured = {}

        monkeypatch.setattr(stv_cli, "log_parameters", lambda *_: None)
        monkeypatch.setattr(
            stv_main, "style_transfer",
            lambda **kwargs: captured.update(kwargs) or torch.rand(1),
        )

        stv_cli.run_from_args(args)
        assert captured["output_dir"] == "config_out"
        assert captured["steps"] == 123  # noqa: PLR2004

    def test_plot_disabled_when_log_loss_set(
        self, monkeypatch: MonkeyPatch, caplog: LogCaptureFixture,
    ) -> None:
        """Test that log_loss disables plot_losses and logs a warning."""
        args = argparse.Namespace(
            content="cat.jpg",
            style="wave.jpg",
            config=None,
            validate_config_only=False,
            output=DEFAULT_OUTPUT_DIR,
            steps=DEFAULT_STEPS,
            save_every=DEFAULT_SAVE_EVERY,
            style_w=DEFAULT_STYLE_WEIGHT,
            content_w=DEFAULT_CONTENT_WEIGHT,
            lr=DEFAULT_LEARNING_RATE,
            fps=DEFAULT_FPS,
            init_method=DEFAULT_INIT_METHOD,
            no_normalize=False,
            no_video=False,
            no_plot=False,  # NOT set, but --log-loss should disable plotting
            final_only=False,
            quality=DEFAULT_VIDEO_QUALITY,
            seed=DEFAULT_SEED,
            device=DEFAULT_DEVICE,
            log_loss="losses.csv",
            log_every=10,
        )

        captured = {}
        monkeypatch.setattr(stv_main, "style_transfer",
                            lambda **kwargs:
                            captured.update(kwargs) or torch.rand(1))
        monkeypatch.setattr(stv_cli, "log_parameters", lambda *_: None)

        caplog.set_level("WARNING")
        stv_cli.run_from_args(args)

        assert captured["plot_losses"] is False
        assert (
            "Loss plotting is disabled because CSV logging is enabled"
            in caplog.text
        )


class TestLogParameters:
    """Tests parameter logging output for CLI execution."""

    def test_log_parameters_logs_config(
        self,
        caplog: LogCaptureFixture,
    ) -> None:
        """Test config path is logged if provided."""
        args = argparse.Namespace(
            content="cat.jpg",
            style="s.jpg",
            config="abc.toml",
            output="out",
            steps=10,
            save_every=2,
            style_w=1.0,
            content_w=1.0,
            lr=0.5,
            style_layers_str="0,5,10",
            content_layers_str="21",
            fps=10,
            init_method="content",
            no_normalize=False,
            no_video=False,
            final_only=False,
            quality=8,
            seed=0,
            device="cpu",
            plot_losses=True,
        )

        caplog.set_level("INFO")
        stv_cli.log_parameters({
            "content_path": args.content,
            "style_path": args.style,
            "output_dir": args.output,
            "steps": args.steps,
            "save_every": args.save_every,
            "style_weight": args.style_w,
            "content_weight": args.content_w,
            "learning_rate": args.lr,
            "style_layers":  [int(x)
                              for x in args.style_layers_str.split(",")],
            "content_layers": [int(x)
                               for x in args.content_layers_str.split(",")],
            "fps": args.fps,
            "init_method": args.init_method,
            "normalize": not args.no_normalize,
            "create_video": not args.no_video,
            "final_only": args.final_only,
            "video_quality": args.quality,
            "seed": args.seed,
            "device_name": args.device,
            "plot_losses": args.plot_losses,
        }, args)

        assert any(
            "Loaded config from: abc.toml" in m for m in caplog.messages
        )

    def test_log_parameters_without_config(
        self,
        caplog: LogCaptureFixture,
    ) -> None:
        """Test log_parameters skips config logging if not provided."""
        args = argparse.Namespace(
            content="cat.jpg",
            style="s.jpg",
            output="out",
            steps=10,
            save_every=2,
            style_w=1.0,
            content_w=1.0,
            lr=0.5,
            style_layers_str="0,5,10",
            content_layers_str="21",
            fps=10,
            init_method="content",
            no_normalize=False,
            no_video=False,
            final_only=False,
            quality=8,
            seed=0,
            device="cpu",
            plot_losses=True,
        )

        caplog.set_level("INFO")
        stv_cli.log_parameters({
            "content_path": args.content,
            "style_path": args.style,
            "output_dir": args.output,
            "steps": args.steps,
            "save_every": args.save_every,
            "style_weight": args.style_w,
            "content_weight": args.content_w,
            "learning_rate": args.lr,
            "style_layers":  [int(x)
                              for x in args.style_layers_str.split(",")],
            "content_layers": [int(x)
                               for x in args.content_layers_str.split(",")],
            "fps": args.fps,
            "init_method": args.init_method,
            "normalize": not args.no_normalize,
            "create_video": not args.no_video,
            "final_only": args.final_only,
            "video_quality": args.quality,
            "seed": args.seed,
            "device_name": args.device,
            "plot_losses": args.plot_losses,
        }, args)

        assert not any("Loaded config from:" in m for m in caplog.messages)

    def test_log_parameters_includes_layer_config(
        self,
        caplog: LogCaptureFixture,
    ) -> None:
        """Ensure log_parameters prints layer config to logs."""
        args = argparse.Namespace(
            content="cat.jpg",
            style="s.jpg",
            style_layers="0,5,10",
            content_layers="21",
            config="abc.toml",
            output="out",
            steps=10,
            save_every=2,
            style_w=1.0,
            content_w=1.0,
            lr=0.5,
            fps=10,
            init_method="content",
            no_normalize=False,
            no_video=False,
            final_only=False,
            quality=8,
            seed=0,
            device="cpu",
            plot_losses=True,
        )
        params = {
            "content_path": args.content,
            "style_path": args.style,
            "output_dir": args.output,
            "steps": args.steps,
            "save_every": args.save_every,
            "style_weight": args.style_w,
            "content_weight": args.content_w,
            "learning_rate": args.lr,
            "fps": args.fps,
            "init_method": args.init_method,
            "normalize": not args.no_normalize,
            "create_video": not args.no_video,
            "final_only": args.final_only,
            "video_quality": args.quality,
            "seed": args.seed,
            "device_name": args.device,
            "style_layers": [0, 5, 10],
            "content_layers": [21],
            "plot_losses": args.plot_losses,
        }
        caplog.set_level("INFO")
        stv_cli.log_parameters(params, args)
        assert "Style Layers" in caplog.text
        assert "Content Layers" in caplog.text

    def test_no_plot_flag(self, monkeypatch: MonkeyPatch) -> None:
        """Test that --no-plot disables plotting."""
        args = argparse.Namespace(
            content="cat.jpg",
            style="wave.jpg",
            config=None,
            validate_config_only=False,
            output=DEFAULT_OUTPUT_DIR,
            steps=DEFAULT_STEPS,
            save_every=DEFAULT_SAVE_EVERY,
            style_w=DEFAULT_STYLE_WEIGHT,
            content_w=DEFAULT_CONTENT_WEIGHT,
            lr=DEFAULT_LEARNING_RATE,
            fps=DEFAULT_FPS,
            init_method=DEFAULT_INIT_METHOD,
            no_normalize=False,
            no_video=False,
            no_plot=True,
            final_only=False,
            quality=DEFAULT_VIDEO_QUALITY,
            seed=DEFAULT_SEED,
            device=DEFAULT_DEVICE,
        )
        captured = {}
        monkeypatch.setattr(stv_main, "style_transfer",
                            lambda **kwargs:
                            captured.update(kwargs) or torch.rand(1))
        monkeypatch.setattr(stv_cli, "log_parameters", lambda *_: None)

        stv_cli.run_from_args(args)
        assert captured["plot_losses"] is False


class TestCLIMainFlow:
    """Container for top-level CLI flow entry point tests."""

    def test_main_invokes_run(self, monkeypatch: MonkeyPatch) -> None:
        """Test that main() runs the CLI flow."""
        monkeypatch.setattr(sys, "argv", [
            "prog", "--content", "c.jpg", "--style", "s.jpg",
        ])

        called = {}
        monkeypatch.setattr(stv_cli, "run_from_args",
            lambda _: called.update({"ran": True}))

        stv_cli.main()
        assert called.get("ran") is True

    def test_run_from_args_validate_config_only(
        self,
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """
        Test --validate-config-only mode exits early after validation.

        This test mocks sys.exit to intercept the call and confirm that
        the config file is validated without executing the pipeline.
        """
        config_path = tmp_path / "config.toml"
        config_path.write_text('[output]\noutput = "test_output"')

        args = argparse.Namespace(
            config=str(config_path),
            validate_config_only=True,
        )

        exit_called = {}

        def fake_exit(code: int = 0) -> None:
            exit_called["code"] = code
            raise SystemExit

        monkeypatch.setattr(stv_cli.sys, "exit", fake_exit)
        monkeypatch.setattr(stv_cli, "log_parameters", lambda _: None)

        with suppress(SystemExit):
            stv_cli.run_from_args(args)

        assert exit_called["code"] == 0


@pytest.mark.integration
def test_script_main_entry(tmp_path: Path) -> None:
    """Integration test: execute script via subprocess with real images."""
    content = tmp_path / "content.jpg"
    style = tmp_path / "style.jpg"

    Image.new("RGB", (64, 64), color="blue").save(content)
    Image.new("RGB", (64, 64), color="green").save(style)

    result = subprocess.run(  # noqa: S603 - trusted subprocess call to Python CLI
        [
            sys.executable,
            "-m",
            "style_transfer_visualizer.cli",
            "--content",
            str(content),
            "--style",
            str(style),
            "--final-only",
            "--device",
            "cpu",
            "--steps",
            "2",
            "--save-every",
            "3",
            "--init-method",
            "white",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.resolve(),
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, (
        f"Script failed with return code {result.returncode}\n"
        f"--- STDOUT ---\n{result.stdout}\n"
        f"--- STDERR ---\n{result.stderr}\n"
    )
    assert "Style transfer completed" in result.stdout or result.stderr
