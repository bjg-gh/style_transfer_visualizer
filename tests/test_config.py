"""
Unit tests for the config module used in Style Transfer Visualizer.

Covers:
- Successful loading of a valid config.toml
- Default fallbacks for missing values
- Error handling for missing files
- Structural validation of the merged configuration object
"""
import tempfile
from typing import Any

import pytest
import tomlkit
from pydantic import ValidationError

import style_transfer_visualizer.config as stv_config
from style_transfer_visualizer.config_defaults import (
    DEFAULT_CONTENT_LAYERS,
    DEFAULT_CONTENT_WEIGHT,
    DEFAULT_DEVICE,
    DEFAULT_FPS,
    DEFAULT_INIT_METHOD,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOG_EVERY,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SAVE_EVERY,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_STYLE_LAYERS,
    DEFAULT_STYLE_WEIGHT,
    DEFAULT_VIDEO_INTRO_DURATION,
    DEFAULT_VIDEO_OUTRO_DURATION,
    DEFAULT_VIDEO_QUALITY,
)


def create_toml_file(data: dict[str, Any]) -> str:
    """Write a TOML string to a temporary file and return its path."""
    doc = tomlkit.document()
    doc.update(data)
    toml_str = tomlkit.dumps(doc)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".toml",
        mode="w",
        encoding="utf-8",
    ) as temp:
        temp.write(toml_str)
        return temp.name



def test_load_valid_config() -> None:
    """Test that a well-formed config.toml loads successfully."""
    config_data = {
        "output": {"output": "results"},
        "optimization": {
            "steps": 500,
            "style_w": 123456.0,
            "init_method": "content",
        },
        "video": {"fps": 30, "final_only": True},
        "hardware": {"device": "cpu"},
    }
    path = create_toml_file(config_data)
    cfg = stv_config.ConfigLoader.load(path)

    assert isinstance(cfg, stv_config.StyleTransferConfig)
    assert cfg.output.output == "results"
    assert cfg.optimization.steps == 500  # noqa: PLR2004
    assert cfg.optimization.style_w == 123456.0  # noqa: PLR2004
    assert cfg.optimization.init_method == "content"
    assert cfg.video.fps == 30  # noqa: PLR2004
    assert cfg.video.final_only is True
    assert cfg.hardware.device == "cpu"
    assert cfg.video.outro_duration_seconds == DEFAULT_VIDEO_OUTRO_DURATION


def test_missing_file_raises() -> None:
    """Ensure FileNotFoundError is raised for nonexistent config."""
    with pytest.raises(FileNotFoundError):
        stv_config.ConfigLoader.load("nonexistent_file.toml")


def test_partial_config_uses_defaults() -> None:
    """ConfigLoader should fall back to defaults for missing sections."""
    path = create_toml_file({
        "optimization": {"steps": 42},
    })
    cfg = stv_config.ConfigLoader.load(path)

    assert cfg.optimization.steps == 42  # noqa: PLR2004
    assert cfg.optimization.lr == DEFAULT_LEARNING_RATE
    assert cfg.video.fps == DEFAULT_FPS
    assert cfg.hardware.device == DEFAULT_DEVICE
    assert cfg.video.outro_duration_seconds == DEFAULT_VIDEO_OUTRO_DURATION


def test_video_config_invalid_fps() -> None:
    """Ensure invalid fps raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        stv_config.VideoConfig(
            fps=0,
            quality=DEFAULT_VIDEO_QUALITY,
            save_every=DEFAULT_SAVE_EVERY,
            intro_duration_seconds=DEFAULT_VIDEO_INTRO_DURATION,
            outro_duration_seconds=DEFAULT_VIDEO_OUTRO_DURATION,
        )
    assert "fps" in str(exc_info.value)


def test_video_config_invalid_quality() -> None:
    """Ensure invalid quality raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        stv_config.VideoConfig(
            quality=20,
            fps=DEFAULT_FPS,
            save_every=DEFAULT_SAVE_EVERY,
            intro_duration_seconds=DEFAULT_VIDEO_INTRO_DURATION,
            outro_duration_seconds=DEFAULT_VIDEO_OUTRO_DURATION,
        )
    assert "quality" in str(exc_info.value)


def test_optimization_config_negative_steps() -> None:
    """Ensure invalid steps raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        stv_config.OptimizationConfig(steps=-1, style_w=DEFAULT_STYLE_WEIGHT,
                                      content_w=DEFAULT_CONTENT_WEIGHT,
                                      lr=DEFAULT_LEARNING_RATE,
                                      seed=DEFAULT_SEED,
                                      init_method=DEFAULT_INIT_METHOD)
    assert "steps" in str(exc_info.value)


def test_optimization_config_invalid_lr() -> None:
    """Ensure invalid learning rate raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        stv_config.OptimizationConfig(lr=0, steps=DEFAULT_STEPS,
                                      style_w=DEFAULT_STYLE_WEIGHT,
                                      content_w=DEFAULT_CONTENT_WEIGHT,
                                      seed=DEFAULT_SEED,
                                      init_method=DEFAULT_INIT_METHOD)
    assert "lr" in str(exc_info.value)


def test_optimization_config_negative_style_w() -> None:
    """Ensure invalid style weight raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        stv_config.OptimizationConfig(style_w=-100, steps=DEFAULT_STEPS,
                                      content_w=DEFAULT_CONTENT_WEIGHT,
                                      lr=DEFAULT_LEARNING_RATE,
                                      seed=DEFAULT_SEED,
                                      init_method=DEFAULT_INIT_METHOD)
    assert "style_w" in str(exc_info.value)


def test_optimization_config_negative_content_w() -> None:
    """Ensure invalid content weight raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        stv_config.OptimizationConfig(content_w=-5.0, steps=DEFAULT_STEPS,
                                      style_w=DEFAULT_STYLE_WEIGHT,
                                      lr=DEFAULT_LEARNING_RATE,
                                      seed=DEFAULT_SEED,
                                      init_method=DEFAULT_INIT_METHOD)
    assert "content_w" in str(exc_info.value)


def test_optimization_config_negative_seed() -> None:
    """Ensure invalid seed raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        stv_config.OptimizationConfig(seed=-42, steps=DEFAULT_STEPS,
                                      style_w=DEFAULT_STYLE_WEIGHT,
                                      content_w=DEFAULT_CONTENT_WEIGHT,
                                      lr=DEFAULT_LEARNING_RATE,
                                      init_method=DEFAULT_INIT_METHOD)
    assert "seed" in str(exc_info.value)


def test_video_config_fps_upper_bound() -> None:
    """Ensure fps upper bound is enforced (must be <= 60)."""
    with pytest.raises(ValidationError) as exc_info:
        stv_config.VideoConfig(
            fps=100,
            quality=DEFAULT_VIDEO_QUALITY,
            save_every=DEFAULT_SAVE_EVERY,
            intro_duration_seconds=DEFAULT_VIDEO_INTRO_DURATION,
            outro_duration_seconds=DEFAULT_VIDEO_OUTRO_DURATION,
        )
    assert "fps" in str(exc_info.value)


def test_output_config_defaults() -> None:
    """Validate OutputConfig defaults and types."""
    cfg = stv_config.OutputConfig.model_validate({})
    assert cfg.output == DEFAULT_OUTPUT_DIR
    assert cfg.log_every == DEFAULT_LOG_EVERY
    assert cfg.log_loss is None
    assert cfg.plot_losses is True


def test_loader_with_empty_toml_uses_all_defaults() -> None:
    """Empty TOML should yield a fully defaulted config object."""
    path = create_toml_file({})
    cfg = stv_config.ConfigLoader.load(path)

    assert cfg.optimization.steps == DEFAULT_STEPS
    assert cfg.optimization.lr == DEFAULT_LEARNING_RATE
    assert cfg.video.fps == DEFAULT_FPS
    assert cfg.video.quality == DEFAULT_VIDEO_QUALITY
    assert cfg.hardware.device == DEFAULT_DEVICE
    assert cfg.video.outro_duration_seconds == DEFAULT_VIDEO_OUTRO_DURATION
    assert cfg.output.output == DEFAULT_OUTPUT_DIR
    assert cfg.output.log_every == DEFAULT_LOG_EVERY


def test_loader_raises_on_invalid_types() -> None:
    """Type errors in TOML should propagate as ValidationError."""
    path = create_toml_file({
        "optimization": {"steps": "not_an_int"},
    })
    with pytest.raises(ValidationError):
        stv_config.ConfigLoader.load(path)


def test_default_layer_indices_match_constants() -> None:
    """Default style and content layer indices should match constants."""
    cfg = stv_config.StyleTransferConfig.model_validate({})
    assert cfg.optimization.style_layers == list(DEFAULT_STYLE_LAYERS)
    assert cfg.optimization.content_layers == list(DEFAULT_CONTENT_LAYERS)
