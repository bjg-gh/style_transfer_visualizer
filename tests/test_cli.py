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

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import torch
from PIL import Image

import style_transfer_visualizer.cli as stv_cli
import style_transfer_visualizer.main as stv_main
from style_transfer_visualizer.config import StyleTransferConfig
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
from style_transfer_visualizer.type_defs import InputPaths

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch



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
        ("style_str", "content_str", "expected_style", "expected_content"),
        [
            ("0,1,2", "3", [0, 1, 2], [3]),
            ("5", "10,15", [5], [10, 15]),
        ],
    )
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

        def fake_run(
            paths: InputPaths,
            st_config: StyleTransferConfig,
        ) -> torch.Tensor:
            captured["paths"] = paths
            captured["cfg"] = st_config
            return torch.rand(1)

        monkeypatch.setattr(stv_main, "style_transfer", fake_run)
        monkeypatch.setattr(stv_cli, "log_parameters", lambda *_: None)

        stv_cli.run_from_args(args)

        cfg_seen = captured["cfg"]
        opt_dict = cfg_seen.optimization.model_dump()

        assert opt_dict["style_layers"] == expected_style
        assert opt_dict["content_layers"] == expected_content

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

        exit_called: dict[str, int] = {}

        def fake_exit(code: int = 0) -> None:
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
            metadata_title="Title",
            metadata_artist="Artist",
        )

        captured: dict[str, Any] = {}

        def fake_run(
            paths: InputPaths,
            st_config: StyleTransferConfig,
        ) -> torch.Tensor:
            captured["paths"] = paths
            captured["cfg"] = st_config
            return torch.rand(1)

        monkeypatch.setattr(stv_main, "style_transfer", fake_run)
        monkeypatch.setattr(stv_cli, "log_parameters", lambda *_: None)

        stv_cli.run_from_args(args)
        cfg = captured["cfg"]
        assert cfg.optimization.steps == 123  # noqa: PLR2004
        assert cfg.optimization.normalize is True
        assert cfg.video.create_video is True
        assert cfg.video.final_only is True
        assert cfg.video.fps == 20  # noqa: PLR2004
        assert cfg.video.quality == 7  # noqa: PLR2004
        assert cfg.video.save_every == 10  # noqa: PLR2004
        assert cfg.video.metadata_title == "Title"
        assert cfg.video.metadata_artist == "Artist"
        assert cfg.optimization.init_method == "white"
        assert cfg.output.output == "out"

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

        captured: dict[str, Any] = {}

        def fake_run(
            _paths: InputPaths,
            st_config: StyleTransferConfig,
        ) -> torch.Tensor:
            captured["cfg"] = st_config
            return torch.rand(1)

        monkeypatch.setattr(stv_main, "style_transfer", fake_run)
        monkeypatch.setattr(stv_cli, "log_parameters", lambda *_: None)

        stv_cli.run_from_args(args)
        cfg = captured["cfg"]
        assert cfg.optimization.normalize is False
        assert cfg.video.create_video is False

    def test_run_from_args_config_not_validating(
        self,
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Test config loads but validate_config_only is False."""
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            """
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
""",
        )

        args = argparse.Namespace(
            content="cat.jpg",
            style="s.jpg",
            config=str(config_path),
            validate_config_only=False,
        )

        captured: dict[str, Any] = {}

        def fake_run(
            _paths: InputPaths,
            st_config: StyleTransferConfig,
        ) -> torch.Tensor:
            captured["cfg"] = st_config
            return torch.rand(1)

        monkeypatch.setattr(stv_cli, "log_parameters", lambda *_: None)
        monkeypatch.setattr(stv_main, "style_transfer", fake_run)

        stv_cli.run_from_args(args)
        cfg = captured["cfg"]
        assert cfg.output.output == "config_out"
        assert cfg.optimization.steps == 123  # noqa: PLR2004

    def test_plot_disabled_when_log_loss_set(
        self,
        monkeypatch: MonkeyPatch,
        caplog: LogCaptureFixture,
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
            no_plot=False,
            final_only=False,
            quality=DEFAULT_VIDEO_QUALITY,
            seed=DEFAULT_SEED,
            device=DEFAULT_DEVICE,
            log_loss="losses.csv",
            log_every=10,
        )

        captured: dict[str, Any] = {}

        def fake_run(
            _paths: InputPaths,
            st_config: StyleTransferConfig,
        ) -> torch.Tensor:
            captured["cfg"] = st_config
            return torch.rand(1)

        monkeypatch.setattr(stv_main, "style_transfer", fake_run)
        monkeypatch.setattr(stv_cli, "log_parameters", lambda *_: None)

        caplog.set_level("WARNING")
        stv_cli.run_from_args(args)

        cfg = captured["cfg"]
        assert cfg.output.plot_losses is False
        assert "Loss plotting is disabled because CSV logging is enabled" in (
            caplog.text
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
        )

        # Minimal but representative cfg for logging
        cfg = StyleTransferConfig.model_validate({})
        cfg.output.output = "out"
        cfg.optimization.steps = 10
        cfg.video.save_every = 2
        cfg.optimization.style_w = 1.0
        cfg.optimization.content_w = 1.0
        cfg.optimization.lr = 0.5
        cfg.optimization.style_layers = [0, 5, 10]
        cfg.optimization.content_layers = [21]
        cfg.video.fps = 10
        cfg.optimization.init_method = "content"
        cfg.optimization.normalize = True
        cfg.video.create_video = True
        cfg.output.plot_losses = True
        cfg.optimization.seed = 0

        paths = InputPaths(content_path=args.content, style_path=args.style)

        caplog.set_level("INFO")
        stv_cli.log_parameters(paths, cfg, args)

        assert any("Loaded config from: abc.toml" in m for m in caplog.messages
                   )

    def test_log_parameters_without_config(
        self,
        caplog: LogCaptureFixture,
    ) -> None:
        """Test log_parameters skips config logging if not provided."""
        args = argparse.Namespace(
            content="cat.jpg",
            style="s.jpg",
            config=None,
        )

        cfg = StyleTransferConfig.model_validate({})
        paths = InputPaths(content_path=args.content, style_path=args.style)

        caplog.set_level("INFO")
        stv_cli.log_parameters(paths, cfg, args)

        assert not any("Loaded config from:" in m for m in caplog.messages)

    def test_log_parameters_includes_layer_config(
        self,
        caplog: LogCaptureFixture,
    ) -> None:
        """Ensure log_parameters prints layer config to logs."""
        args = argparse.Namespace(
            content="cat.jpg",
            style="s.jpg",
            config="abc.toml",
        )
        cfg = StyleTransferConfig.model_validate({})
        cfg.optimization.style_layers = [0, 5, 10]
        cfg.optimization.content_layers = [21]
        paths = InputPaths(content_path=args.content, style_path=args.style)

        caplog.set_level("INFO")
        stv_cli.log_parameters(paths, cfg, args)
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
        captured: dict[str, Any] = {}

        def fake_run(
            _paths: InputPaths,
            st_config: StyleTransferConfig,
        ) -> torch.Tensor:
            captured["cfg"] = st_config
            return torch.rand(1)

        monkeypatch.setattr(stv_main, "style_transfer", fake_run)
        monkeypatch.setattr(stv_cli, "log_parameters", lambda *_: None)

        stv_cli.run_from_args(args)
        assert captured["cfg"].output.plot_losses is False


class TestCLIMainFlow:
    """Container for top-level CLI flow entry point tests."""

    def test_main_invokes_run(self, monkeypatch: MonkeyPatch) -> None:
        """Test that main() runs the CLI flow."""
        monkeypatch.setattr(sys, "argv", [
            "prog", "--content", "c.jpg", "--style", "s.jpg",
        ])

        called: dict[str, bool] = {}
        monkeypatch.setattr(
            stv_cli,
            "run_from_args",
            lambda _: called.update({"ran": True}),
        )

        stv_cli.main()
        assert called.get("ran") is True


@pytest.mark.integration
def test_script_main_entry(tmp_path: Path) -> None:
    """Integration test: execute script via subprocess with real images."""
    content = tmp_path / "content.jpg"
    style = tmp_path / "style.jpg"

    Image.new("RGB", (64, 64), color="blue").save(content)
    Image.new("RGB", (64, 64), color="green").save(style)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent / "src")

    result = subprocess.run(  # noqa: S603 - trusted subprocess call
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
        env=env,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, (
        f"Script failed with return code {result.returncode}\n"
        f"--- STDOUT ---\n{result.stdout}\n"
        f"--- STDERR ---\n{result.stderr}\n"
    )
    assert "Style transfer completed" in result.stdout or result.stderr


def test_parse_int_list_with_list() -> None:
    """Verify parse_int_list returns list unchanged when input is a list."""
    data = [1, 2, 3]
    result = stv_cli.parse_int_list(data)
    assert result == data
    # Ensure it's the same object type and values
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)
