"""Tests for the command-line interface in style_transfer_visualizer.

These tests verify correct behavior of argument parsing, CLI flags,
entry point invocation, and integration with style transfer execution.

Modules tested:
- build_arg_parser()
- run_from_args()
- main()

This module uses monkeypatching and integration tests to simulate
CLI use.
"""

import argparse
import sys
import subprocess
from typing import Any

import pytest
import torch
from pathlib import Path
from PIL import Image
import style_transfer_visualizer as stv

from config_defaults import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_STEPS,
    DEFAULT_SAVE_EVERY,
    DEFAULT_STYLE_WEIGHT,
    DEFAULT_CONTENT_WEIGHT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_FPS,
    DEFAULT_VIDEO_QUALITY,
    DEFAULT_INIT_METHOD,
    DEFAULT_DEVICE,
    DEFAULT_SEED,
    DEFAULT_FINAL_ONLY,
    DEFAULT_NORMALIZE,
    DEFAULT_CREATE_VIDEO
)

class TestCLIArgumentParsing:
    def test_arg_parser_defaults(self):
        """Test that default CLI arguments are parsed correctly."""
        parser = stv.build_arg_parser()
        default_args = parser.parse_args(["--content", "c.jpg", "--style",
                                          "s.jpg"])

        assert default_args.output == DEFAULT_OUTPUT_DIR
        assert default_args.steps == DEFAULT_STEPS
        assert default_args.save_every == DEFAULT_SAVE_EVERY
        assert default_args.style_w == DEFAULT_STYLE_WEIGHT
        assert default_args.content_w == DEFAULT_CONTENT_WEIGHT
        assert default_args.lr == DEFAULT_LEARNING_RATE
        assert default_args.fps == DEFAULT_FPS
        assert default_args.quality == DEFAULT_VIDEO_QUALITY
        assert default_args.init_method == DEFAULT_INIT_METHOD
        assert default_args.device == DEFAULT_DEVICE
        assert default_args.seed == DEFAULT_SEED
        assert default_args.final_only == DEFAULT_FINAL_ONLY

    def test_arg_parser_flags(self):
        """Test parsing of boolean flags like --no-normalize and
        --final-only."""
        parser = stv.build_arg_parser()
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

    @pytest.mark.parametrize("invalid_flag", ["--steps", "--fps", "--device"])
    def test_arg_parser_invalid_usage(self, invalid_flag: str):
        """Test that invalid CLI flag usage exits cleanly."""
        parser = stv.build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--content", "c.jpg", "--style", "s.jpg",
                               invalid_flag])


class TestCLIMainFlow:
    def test_run_from_args_passes_expected(self, monkeypatch: Any):
        """Test that run_from_args correctly maps args to
        style_transfer()."""
        dummy_args: argparse.Namespace = argparse.Namespace(
            content="c.jpg",
            style="s.jpg",
            output="output",
            steps=123,
            save_every=10,
            style_w=99,
            content_w=3.14,
            lr=0.42,
            fps=5,
            init_method="white",
            no_normalize=False,
            no_video=False,
            final_only=True,
            quality=8,
            seed=7,
            device="cpu",
            config=None
        )

        captured_args: dict = {}
        monkeypatch.setattr(
            stv,
            "style_transfer",
            lambda **kwargs: captured_args.update(kwargs)
            or torch.rand(1, 3, 64, 64)
        )
        monkeypatch.setattr(stv, "log_parameters", lambda x: None)

        result = stv.run_from_args(dummy_args)
        assert isinstance(result, torch.Tensor)
        assert captured_args["output_dir"] == "output"
        assert captured_args["steps"] == 123
        assert captured_args["save_every"] == 10
        assert captured_args["style_weight"] == 99
        assert captured_args["content_weight"] == 3.14
        assert captured_args["learning_rate"] == 0.42
        assert captured_args["fps"] == 5
        assert captured_args["device_name"] == "cpu"
        assert captured_args["init_method"] == "white"
        assert captured_args["normalize"] is True
        assert captured_args["create_video"] is True
        assert captured_args["final_only"] is True
        assert captured_args["video_quality"] == 8
        assert captured_args["seed"] == 7

    def test_run_from_args_flag_effect(self, monkeypatch: Any):
        """Verify --no-normalize and --no-video change internal flags."""
        args = argparse.Namespace(
            content="c.jpg",
            style="s.jpg",
            output=None,
            steps=None,
            save_every=None,
            style_w=None,
            content_w=None,
            lr=None,
            fps=None,
            init_method=None,
            no_normalize=True,
            no_video=True,
            final_only=None,
            quality=None,
            seed=None,
            device=None,
            config=None
        )
        captured = {}

        monkeypatch.setattr("style_transfer_visualizer.log_parameters",
                            lambda _: None)
        monkeypatch.setattr(
            "style_transfer_visualizer.style_transfer",
            lambda **kwargs: captured.update(kwargs)
                             or torch.rand(1, 3, 64, 64)
        )

        stv.run_from_args(args)
        assert captured["normalize"] is False
        assert captured["create_video"] is False

    def test_main_invokes_run(self, monkeypatch: Any):
        """Test __main__ entry point runs successfully."""
        monkeypatch.setattr(
            sys, "argv",
            ["prog", "--content", "c.jpg", "--style", "s.jpg"]
        )

        was_called: dict = {}
        monkeypatch.setattr(
            stv, "run_from_args",
            lambda args: was_called.update({"called": True})
        )

        stv.main()
        assert was_called.get("called") is True

    def test_arg_parser_missing_required(self, monkeypatch: Any):
        """Test that missing required CLI arguments triggers SystemExit."""
        monkeypatch.setattr(
            sys, "argv",
            ["prog"]
        )

        with pytest.raises(SystemExit):
            stv.main()

    def test_log_parameters_logs_config(self, caplog):
        """Test that log_parameters logs the config path when present."""
        args = argparse.Namespace(
            config="path/to/config.toml",
            content="c.jpg",
            style="s.jpg",
            output=DEFAULT_OUTPUT_DIR,
            steps=DEFAULT_STEPS,
            style_w=DEFAULT_STYLE_WEIGHT,
            content_w=DEFAULT_CONTENT_WEIGHT,
            lr=DEFAULT_LEARNING_RATE,
            save_every=DEFAULT_SAVE_EVERY,
            fps=DEFAULT_FPS,
            quality=DEFAULT_VIDEO_QUALITY,
            init_method=DEFAULT_INIT_METHOD,
            device=DEFAULT_DEVICE,
            seed=DEFAULT_SEED,
            final_only=DEFAULT_FINAL_ONLY,
            no_normalize=not DEFAULT_NORMALIZE,
            no_video=not DEFAULT_CREATE_VIDEO
        )
        caplog.set_level("INFO")

        stv.log_parameters(args)
        assert any("Loaded config from: path/to/config.toml"
                   in message for message in caplog.messages)

    def test_run_from_args_config_fallback(
        self,
        monkeypatch: Any,
        tmp_path: Path
    ):
        """Test that run_from_args loads config and falls back to
           defaults when needed."""
        # Create minimal config with just one field
        config_path = tmp_path / "config.toml"
        config_path.write_text("[output]\noutput = 'from_config'")

        args = argparse.Namespace(
            content="content.jpg",
            style="style.jpg",
            output=None,
            steps=None,
            save_every=None,
            style_w=None,
            content_w=None,
            lr=None,
            fps=None,
            init_method=None,
            no_normalize=None,
            no_video=None,
            final_only=None,
            quality=None,
            seed=None,
            device=None,
            config=str(config_path),
            validate_config_only=False
        )

        captured = {}

        monkeypatch.setattr("style_transfer_visualizer.log_parameters",
                            lambda _: None)
        monkeypatch.setattr(
            "style_transfer_visualizer.style_transfer",
            lambda **kwargs: captured.update(kwargs)
                             or torch.rand(1, 3, 64, 64)
        )

        result = stv.run_from_args(args)
        assert isinstance(result, torch.Tensor)
        assert captured["output_dir"] == "from_config"
        assert captured["steps"] == DEFAULT_STEPS
        assert captured["style_weight"] == DEFAULT_STYLE_WEIGHT
        assert captured["content_weight"] == DEFAULT_CONTENT_WEIGHT
        assert captured["learning_rate"] == DEFAULT_LEARNING_RATE
        assert captured["fps"] == DEFAULT_FPS
        assert captured["device_name"] == DEFAULT_DEVICE
        assert captured["init_method"] == DEFAULT_INIT_METHOD
        assert captured["normalize"] is True
        assert captured["create_video"] is True
        assert captured["final_only"] == DEFAULT_FINAL_ONLY

    def test_run_from_args_get_returns_none(self, monkeypatch: Any):
        """Trigger the 'return None' path in get() when both CLI and
           config are missing."""
        # Provide only the required CLI args
        args = argparse.Namespace(
            content="cat.jpg",
            style="wave.jpg",
            output=DEFAULT_OUTPUT_DIR,
            config=None,
            validate_config_only=False
        )

        captured = {}

        monkeypatch.setattr("style_transfer_visualizer.log_parameters",
                            lambda _: None)
        monkeypatch.setattr(
            "style_transfer_visualizer.style_transfer",
            lambda **kwargs: captured.update(kwargs)
                             or torch.rand(1, 3, 64, 64)
        )

        result = stv.run_from_args(args)
        assert isinstance(result, torch.Tensor)
        assert captured["output_dir"] == DEFAULT_OUTPUT_DIR

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

        monkeypatch.setattr(stv.sys, "exit", fake_exit)
        monkeypatch.setattr(stv, "log_parameters", lambda _: None)

        try:
            stv.run_from_args(args)
        except SystemExit:
            pass

        assert exit_called["code"] == 0


@pytest.mark.integration
def test_script_main_entry(tmp_path: Path):
    """Integration test: execute script via subprocess with real images."""
    script: Path = Path("style_transfer_visualizer.py").resolve()
    content: Path = tmp_path / "content.jpg"
    style: Path = tmp_path / "style.jpg"

    Image.new("RGB", (64, 64), color="blue").save(content)
    Image.new("RGB", (64, 64), color="green").save(style)

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--content", str(content),
            "--style", str(style),
            "--final-only",
            "--device", "cpu",
            "--steps", "2",
            "--save-every", "3",
            "--init-method", "white"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, (
        f"Script failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
    assert "Style transfer completed" in result.stdout or result.stderr
