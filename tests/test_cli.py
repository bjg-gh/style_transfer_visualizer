# tests/test_cli.py
"""
Tests for the updated CLI parser and execution logic.

These tests verify correct CLI parsing, config fallback behavior,
flag handling, and main entry point integration.

Modules tested:
- build_arg_parser()
- run_from_args()
- main()

Simulates CLI usage with monkeypatching and verifies end to end flow.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypedDict

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
    DEFAULT_VIDEO_OUTRO_DURATION,
    DEFAULT_VIDEO_QUALITY,
)
from style_transfer_visualizer.runtime.comparison import ComparisonRequest
from style_transfer_visualizer.type_defs import InputPaths

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch


class RenderComparisonCall(TypedDict):
    """Captured call information for render_requested_comparisons."""

    content_path: Path
    style_path: Path
    output_dir: Path
    request: ComparisonRequest


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
            "--gif",
            "--gif-include-intro",
            "--gif-include-outro",
        ])

        assert args.no_normalize is True
        assert args.no_video is True
        assert args.final_only is True
        assert args.no_plot is True
        assert args.create_gif is True
        assert args.gif_include_intro is True
        assert args.gif_include_outro is True

        args_no_gif = parser.parse_args([
            "--content", "c.jpg",
            "--style", "s.jpg",
            "--no-gif",
        ])
        assert args_no_gif.create_gif is False

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
            compare_inputs=False,
            compare_result=False,
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
        """Test --validate-config-only short circuits the run."""
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
            no_intro=False,
            intro_duration=-5.0,
            outro_duration=-5.0,
            create_gif=True,
            gif_include_intro=True,
            gif_include_outro=True,
            compare_inputs=False,
            compare_result=False,
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
        assert cfg.video.intro_enabled is True
        assert cfg.video.intro_duration_seconds == 0.0
        assert cfg.video.outro_duration_seconds == 0.0
        assert cfg.video.create_gif is True
        assert cfg.video.gif_include_intro is True
        assert cfg.video.gif_include_outro is True
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
            no_intro=True,
            quality=DEFAULT_VIDEO_QUALITY,
            seed=DEFAULT_SEED,
            device=DEFAULT_DEVICE,
            compare_inputs=False,
            compare_result=False,
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
        assert cfg.video.intro_enabled is False

    def test_run_from_args_defaults_final_frame_compare_on(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """Final comparison frame should be enabled by default."""
        args = argparse.Namespace(
            content="cat.jpg",
            style="wave.jpg",
            config=None,
            validate_config_only=False,
            output=DEFAULT_OUTPUT_DIR,
            steps=DEFAULT_STEPS,
            save_every=DEFAULT_SAVE_EVERY,
            fps=DEFAULT_FPS,
            quality=DEFAULT_VIDEO_QUALITY,
            no_video=False,
            final_only=False,
            no_intro=False,
            no_normalize=False,
            compare_inputs=False,
            compare_result=False,
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
        assert cfg.video.final_frame_compare is True
        assert cfg.video.outro_duration_seconds == DEFAULT_VIDEO_OUTRO_DURATION

    def test_run_from_args_disables_final_frame_compare(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """--no-final-frame-compare should disable the feature."""
        args = argparse.Namespace(
            content="cat.jpg",
            style="wave.jpg",
            config=None,
            validate_config_only=False,
            output=DEFAULT_OUTPUT_DIR,
            steps=DEFAULT_STEPS,
            save_every=DEFAULT_SAVE_EVERY,
            fps=DEFAULT_FPS,
            quality=DEFAULT_VIDEO_QUALITY,
            no_video=False,
            final_only=False,
            final_frame_compare=False,
            no_intro=False,
            no_normalize=False,
            compare_inputs=False,
            compare_result=False,
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
        assert cfg.video.final_frame_compare is False

    def test_run_from_args_sets_outro_duration(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """--outro-duration should override the default value."""
        override_outro = 3.5
        args = argparse.Namespace(
            content="cat.jpg",
            style="wave.jpg",
            config=None,
            validate_config_only=False,
            output=DEFAULT_OUTPUT_DIR,
            steps=DEFAULT_STEPS,
            save_every=DEFAULT_SAVE_EVERY,
            fps=DEFAULT_FPS,
            quality=DEFAULT_VIDEO_QUALITY,
            no_video=False,
            final_only=False,
            final_frame_compare=True,
            outro_duration=override_outro,
            no_intro=False,
            no_normalize=False,
            compare_inputs=False,
            compare_result=False,
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
        assert cfg.video.outro_duration_seconds == override_outro

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
            compare_inputs=False,
            compare_result=False,
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
            compare_inputs=False,
            compare_result=False,
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

    def test_compare_inputs_saves_gallery(
        self,
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """The inputs only comparison saves a gallery image."""
        content = tmp_path / "c.jpg"
        style = tmp_path / "s.jpg"
        Image.new("RGB", (32, 24), "red").save(content)
        Image.new("RGB", (32, 24), "blue").save(style)

        out_dir = tmp_path / "out"
        out_dir.mkdir()

        args = argparse.Namespace(
            content=str(content),
            style=str(style),
            config=None,
            validate_config_only=False,
            output=str(out_dir),
            steps=1,
            save_every=1,
            style_w=1.0,
            content_w=1.0,
            lr=0.1,
            init_method="random",
            no_normalize=False,
            no_video=True,
            final_only=True,
            quality=10,
            fps=5,
            seed=0,
            device="cpu",
            compare_inputs=True,
            compare_result=False,
        )

        recorded: dict[str, RenderComparisonCall] = {}

        def fake_render_requested_comparisons(
            *,
            content_path: Path,
            style_path: Path,
            output_dir: Path,
            request: ComparisonRequest,
        ) -> list[Path]:
            recorded["call"] = {
                "content_path": content_path,
                "style_path": style_path,
                "output_dir": output_dir,
                "request": request,
            }
            return []

        monkeypatch.setattr(
            stv_cli,
            "render_requested_comparisons",
            fake_render_requested_comparisons,
        )
        monkeypatch.setattr(stv_main, "style_transfer", lambda *_: None)
        monkeypatch.setattr(stv_cli, "log_parameters", lambda *_: None)

        stv_cli.run_from_args(args)
        call = recorded["call"]
        assert call["request"].include_inputs is True
        assert call["request"].include_result is False
        assert call["output_dir"] == Path(out_dir)

    def test_compare_result_requests_renderer(
        self,
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """The result comparison delegates to the comparison renderer."""
        content = tmp_path / "c.jpg"
        style = tmp_path / "s.jpg"
        Image.new("RGB", (16, 16), "red").save(content)
        Image.new("RGB", (16, 16), "blue").save(style)

        out_dir = tmp_path / "out"
        out_dir.mkdir()

        args = argparse.Namespace(
            content=str(content),
            style=str(style),
            config=None,
            validate_config_only=False,
            output=str(out_dir),
            steps=1,
            save_every=1,
            style_w=1.0,
            content_w=1.0,
            lr=0.1,
            init_method="random",
            no_normalize=False,
            no_video=True,
            final_only=True,
            quality=10,
            fps=5,
            seed=0,
            device="cpu",
            compare_inputs=False,
            compare_result=True,
        )

        call_count: dict[str, int] = {"count": 0}

        def fake_render_requested_comparisons(
            *,
            content_path: Path,
            style_path: Path,
            output_dir: Path,
            request: ComparisonRequest,
        ) -> list[Path]:
            call_count["count"] += 1
            return []

        monkeypatch.setattr(
            stv_cli,
            "render_requested_comparisons",
            fake_render_requested_comparisons,
        )
        monkeypatch.setattr(stv_main, "style_transfer", lambda *_: None)
        monkeypatch.setattr(stv_cli, "log_parameters", lambda *_: None)

        stv_cli.run_from_args(args)
        assert call_count["count"] == 1


class TestLogParameters:
    """Tests parameter logging output for CLI execution."""

    def test_log_parameters_logs_config(
        self,
        caplog: LogCaptureFixture,
        make_style_transfer_config: Callable[..., StyleTransferConfig],
        make_input_paths: Callable[..., InputPaths],
    ) -> None:
        """Test config path is logged if provided."""
        args = argparse.Namespace(
            content="cat.jpg",
            style="s.jpg",
            config="abc.toml",
        )

        cfg = make_style_transfer_config(
            optimization={
                "steps": 10,
                "style_w": 1.0,
                "content_w": 1.0,
                "lr": 0.5,
                "style_layers": [0, 5, 10],
                "content_layers": [21],
                "init_method": "content",
                "normalize": True,
                "seed": 0,
            },
            video={
                "save_every": 2,
                "fps": 10,
                "create_video": True,
            },
            output={"output": "out", "plot_losses": True},
        )

        paths = make_input_paths(content=args.content, style=args.style)

        caplog.set_level("INFO")
        stv_cli.log_parameters(paths, cfg, args)

        assert any("Loaded config from: abc.toml"
                   in m for m in caplog.messages)

    def test_log_parameters_reports_gif_settings(
        self,
        caplog: LogCaptureFixture,
        make_style_transfer_config: Callable[..., StyleTransferConfig],
        make_input_paths: Callable[..., InputPaths],
    ) -> None:
        """GIF-related log entries should reflect config flags."""
        args = argparse.Namespace(content="cat.jpg", style="s.jpg", config=None)
        cfg = make_style_transfer_config(
            video={
                "create_gif": True,
                "gif_include_intro": True,
                "gif_include_outro": True,
            },
        )
        paths = make_input_paths(content=args.content, style=args.style)
        caplog.set_level("INFO")
        stv_cli.log_parameters(paths, cfg, args)
        assert any("GIF Export: Enabled" in msg for msg in caplog.messages)
        assert any("GIF Intro Included: Yes" in msg for msg in caplog.messages)
        assert any("GIF Outro Included: Yes" in msg for msg in caplog.messages)

    def test_log_parameters_without_config(
        self,
        caplog: LogCaptureFixture,
        make_style_transfer_config: Callable[..., StyleTransferConfig],
        make_input_paths: Callable[..., InputPaths],
    ) -> None:
        """Test log_parameters skips config logging if not provided."""
        args = argparse.Namespace(
            content="cat.jpg",
            style="s.jpg",
            config=None,
        )

        cfg = make_style_transfer_config()
        paths = make_input_paths(content=args.content, style=args.style)

        caplog.set_level("INFO")
        stv_cli.log_parameters(paths, cfg, args)

        assert not any("Loaded config from:" in m for m in caplog.messages)

    def test_log_parameters_includes_layer_config(
        self,
        caplog: LogCaptureFixture,
        make_style_transfer_config: Callable[..., StyleTransferConfig],
        make_input_paths: Callable[..., InputPaths],
    ) -> None:
        """Ensure log_parameters prints layer config to logs."""
        args = argparse.Namespace(
            content="cat.jpg",
            style="s.jpg",
            config="abc.toml",
        )
        cfg = make_style_transfer_config(
            optimization={
                "style_layers": [0, 5, 10],
                "content_layers": [21],
            },
        )
        paths = make_input_paths(content=args.content, style=args.style)

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
            compare_inputs=False,
            compare_result=False,
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

    result = subprocess.run(  # noqa: S603
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
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)
