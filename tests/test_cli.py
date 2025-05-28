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


class TestCLIArgumentParsing:
    def test_arg_parser_defaults(self):
        """Test that default CLI arguments are parsed correctly."""
        parser = stv.build_arg_parser()
        args = parser.parse_args(["--content", "a.jpg", "--style", "b.jpg"])

        assert args.content == "a.jpg"
        assert args.style == "b.jpg"
        assert args.output == "out"
        assert args.steps == 300
        assert args.fps == 10
        assert args.quality == 10
        assert args.init_method == "random"
        assert args.no_normalize is False
        assert args.no_video is False
        assert args.final_only is False

    def test_arg_parser_flags(self):
        """Test parsing of boolean flags like --no-normalize and
        --final-only."""
        parser = stv.build_arg_parser()
        args = parser.parse_args([
            "--content", "a.jpg",
            "--style", "b.jpg",
            "--no-normalize",
            "--no-video",
            "--final-only"
        ])

        assert args.no_normalize is True
        assert args.no_video is True
        assert args.final_only is True

    def test_arg_parser_missing_required(self):
        """Test that missing required CLI arguments triggers SystemExit."""
        parser = stv.build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    @pytest.mark.parametrize("invalid_flag", ["--steps", "--fps", "--device"])
    def test_arg_parser_invalid_usage(self, invalid_flag: str):
        """Test that invalid CLI flag usage exits cleanly."""
        parser = stv.build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--content", "a.jpg", "--style", "b.jpg",
                               invalid_flag])


class TestCLIMainFlow:
    def test_run_from_args_passes_expected(self, monkeypatch: Any):
        """Test that run_from_args correctly maps args to
        style_transfer()."""
        dummy_args: argparse.Namespace = argparse.Namespace(
            content="foo.jpg",
            style="bar.jpg",
            output="out",
            steps=10,
            save_every=2,
            style_w=1e6,
            content_w=1.0,
            lr=0.5,
            fps=15,
            height=720,
            device="cpu",
            init_method="content",
            no_normalize=False,
            no_video=False,
            final_only=False,
            quality=7,
            seed=123
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
        assert captured_args["content_path"] == "foo.jpg"
        assert captured_args["normalize"] is True

    def test_run_from_args_flag_effect(self, monkeypatch: Any):
        """Verify --no-normalize and --no-video change internal flags."""
        args: argparse.Namespace = argparse.Namespace(
            content="foo.jpg",
            style="bar.jpg",
            output="out",
            steps=10,
            save_every=2,
            style_w=1e6,
            content_w=1.0,
            lr=0.5,
            fps=15,
            height=720,
            device="cpu",
            init_method="random",
            no_normalize=True,
            no_video=True,
            final_only=False,
            quality=8,
            seed=1
        )

        flags: dict = {}
        monkeypatch.setattr(
            stv,
            "style_transfer",
            lambda **kwargs: flags.update(kwargs)
            or torch.rand(1, 3, 64, 64)
        )
        monkeypatch.setattr(stv, "log_parameters", lambda x: None)

        stv.run_from_args(args)
        assert flags["normalize"] is False
        assert flags["create_video"] is False

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
