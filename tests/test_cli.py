"""Tests for the updated CLI parser and execution logic.

These tests verify correct CLI parsing, config fallback behavior,
flag handling, and main entry point integration.

Modules tested:
- build_arg_parser()
- run_from_args()
- main()

Simulates CLI usage with monkeypatching and verifies end-to-end flow.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

import style_transfer_visualizer.cli as stv_cli
import style_transfer_visualizer.main as stv_main
from style_transfer_visualizer.config_defaults import (
    DEFAULT_OUTPUT_DIR, DEFAULT_STEPS, DEFAULT_SAVE_EVERY,
    DEFAULT_STYLE_WEIGHT, DEFAULT_CONTENT_WEIGHT, DEFAULT_LEARNING_RATE,
    DEFAULT_FPS, DEFAULT_VIDEO_QUALITY, DEFAULT_INIT_METHOD,
    DEFAULT_DEVICE, DEFAULT_SEED
)


class TestCLIArgumentParsing:
    def test_flag_parsing(self):
        """Test that boolean flags are parsed correctly."""
        parser = stv_cli.build_arg_parser()
        args = parser.parse_args([
            "--content", "c.jpg",
            "--style", "s.jpg",
            "--no-normalize",
            "--no-video",
            "--final-only"
        ])

        assert args.no_normalize is True
        assert args.no_video is True
        assert args.final_only is True

    def test_required_args_missing(self, monkeypatch: Any):
        """Test that missing required arguments triggers SystemExit."""
        monkeypatch.setattr(sys, "argv", ["prog"])
        with pytest.raises(SystemExit):
            stv_cli.main()


class TestCLIRunFromArgs:
    def test_config_only_mode(self, monkeypatch: Any, tmp_path: Path):
        """Test --validate-config-only short-circuits the run."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[output]\noutput = 'abc'")

        args = argparse.Namespace(
            config=str(config_path),
            validate_config_only=True
        )

        exit_called = {}

        def fake_exit(code=0):
            exit_called["code"] = code
            raise SystemExit()

        monkeypatch.setattr(stv_cli.sys, "exit", fake_exit)
        monkeypatch.setattr(stv_cli, "log_parameters", lambda *_: None)

        with pytest.raises(SystemExit):
            stv_cli.run_from_args(args)

        assert exit_called["code"] == 0

    def test_args_override_config(self, monkeypatch: Any):
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
            device="cpu"
        )

        captured = {}
        monkeypatch.setattr(stv_main, "style_transfer",
            lambda **kwargs: captured.update(kwargs) or torch.rand(1))
        monkeypatch.setattr(stv_cli, "log_parameters", lambda *_: None)

        result = stv_cli.run_from_args(args)
        assert isinstance(result, torch.Tensor)
        assert captured["steps"] == 123
        assert captured["normalize"] is True
        assert captured["create_video"] is True

    def test_flags_flip_behavior(self, monkeypatch: Any):
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
            device=DEFAULT_DEVICE
        )

        captured = {}
        monkeypatch.setattr(stv_main, "style_transfer",
            lambda **kwargs: captured.update(kwargs) or torch.rand(1))
        monkeypatch.setattr(stv_cli, "log_parameters", lambda *_: None)

        stv_cli.run_from_args(args)
        assert captured["normalize"] is False
        assert captured["create_video"] is False


    def test_run_from_args_config_not_validating(
        self, monkeypatch: Any, tmp_path: Path
    ):
        """Test config loads but validate_config_only is False, so get() is used."""
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
            validate_config_only=False
        )

        captured = {}

        monkeypatch.setattr(stv_cli, "log_parameters", lambda *_: None)
        monkeypatch.setattr(
            stv_main, "style_transfer",
            lambda **kwargs: captured.update(kwargs) or torch.rand(1)
        )

        result = stv_cli.run_from_args(args)
        assert isinstance(result, torch.Tensor)
        assert captured["output_dir"] == "config_out"
        assert captured["steps"] == 123


class TestLogParameters:
    def test_log_parameters_logs_config(self, caplog):
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
            fps=10,
            init_method="content",
            no_normalize=False,
            no_video=False,
            final_only=False,
            quality=8,
            seed=0,
            device="cpu"
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
            "fps": args.fps,
            "init_method": args.init_method,
            "normalize": not args.no_normalize,
            "create_video": not args.no_video,
            "final_only": args.final_only,
            "video_quality": args.quality,
            "seed": args.seed,
            "device_name": args.device
        }, args)

        assert any("Loaded config from: abc.toml" in m for m in caplog.messages)

    def test_log_parameters_without_config(self, caplog):
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
            fps=10,
            init_method="content",
            no_normalize=False,
            no_video=False,
            final_only=False,
            quality=8,
            seed=0,
            device="cpu"
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
            "fps": args.fps,
            "init_method": args.init_method,
            "normalize": not args.no_normalize,
            "create_video": not args.no_video,
            "final_only": args.final_only,
            "video_quality": args.quality,
            "seed": args.seed,
            "device_name": args.device
        }, args)

        assert not any("Loaded config from:" in m for m in caplog.messages)


class TestCLIMainFlow:
    def test_main_invokes_run(self, monkeypatch: Any):
        """Test that main() runs the CLI flow."""
        monkeypatch.setattr(sys, "argv", [
            "prog", "--content", "c.jpg", "--style", "s.jpg"
        ])

        called = {}
        monkeypatch.setattr(stv_cli, "run_from_args",
            lambda _: called.update({"ran": True}))

        stv_cli.main()
        assert called.get("ran") is True

    def test_run_from_args_validate_config_only(self, monkeypatch, tmp_path):
        """
        Strategy: Test --validate-config-only mode by providing only the
         required CLI args.

        We isolate the validation branch by mocking sys.exit early.
        Monkeypatch sys.exit to raise SystemExit so the code path halts
        as expected.
        """
        config_path = tmp_path / "config.toml"
        config_path.write_text("[output]\noutput = \"test_output\"")

        args = argparse.Namespace(
            config=str(config_path),
            validate_config_only=True
        )

        exit_called = {}

        def fake_exit(code=0):
            exit_called["code"] = code
            raise SystemExit()

        monkeypatch.setattr(stv_cli.sys, "exit", fake_exit)
        monkeypatch.setattr(stv_cli, "log_parameters", lambda _: None)

        try:
            stv_cli.run_from_args(args)
        except SystemExit:
            pass

        assert exit_called["code"] == 0
