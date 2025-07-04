"""
Unit tests for the config module used in Style Transfer Visualizer.

Covers:
- Successful loading of a valid config.toml
- Default fallbacks for missing values
- Error handling for missing files
- Structural validation of the merged configuration object
"""
import tempfile

import pytest
import tomlkit
from pydantic import ValidationError

from style_transfer_visualizer.config_defaults import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_FPS,
    DEFAULT_DEVICE,
    DEFAULT_VIDEO_QUALITY,
    DEFAULT_SAVE_EVERY,
    DEFAULT_STYLE_WEIGHT,
    DEFAULT_CONTENT_WEIGHT,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_INIT_METHOD
)
import style_transfer_visualizer.config as stv_config


def create_toml_file(data: dict) -> str:
    """Write a TOML string to a temporary file and return its path."""
    doc = tomlkit.document()
    doc.update(data)
    toml_str = tomlkit.dumps(doc)
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".toml", mode="w",
                                       encoding="utf-8")
    temp.write(toml_str)
    temp.close()
    return temp.name


def test_load_valid_config():
    """Test that a well-formed config.toml loads successfully."""
    config_data = {
        "output": {"output": "results"},
        "optimization": {
            "steps": 500,
            "style_w": 123456.0,
            "init_method": "content"
        },
        "video": {"fps": 30, "final_only": True},
        "hardware": {"device": "cpu"}
    }
    path = create_toml_file(config_data)
    cfg = stv_config.ConfigLoader.load(path)

    assert isinstance(cfg, stv_config.StyleTransferConfig)
    assert cfg.output.output == "results"
    assert cfg.optimization.steps == 500
    assert cfg.optimization.style_w == 123456.0
    assert cfg.optimization.init_method == "content"
    assert cfg.video.fps == 30
    assert cfg.video.final_only is True
    assert cfg.hardware.device == "cpu"


def test_missing_file_raises():
    """Ensure FileNotFoundError is raised for nonexistent config."""
    with pytest.raises(FileNotFoundError):
        stv_config.ConfigLoader.load("nonexistent_file.toml")


def test_partial_config_uses_defaults():
    """ConfigLoader should fall back to defaults for missing sections."""
    path = create_toml_file({
        "optimization": {"steps": 42}
    })
    cfg = stv_config.ConfigLoader.load(path)

    assert cfg.optimization.steps == 42
    assert cfg.optimization.lr == DEFAULT_LEARNING_RATE
    assert cfg.video.fps == DEFAULT_FPS
    assert cfg.hardware.device == DEFAULT_DEVICE

def test_video_config_invalid_fps():
    """Ensure invalid fps raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        stv_config.VideoConfig(fps=0, quality=DEFAULT_VIDEO_QUALITY,
                               save_every=DEFAULT_SAVE_EVERY)
    assert "fps" in str(exc_info.value)


def test_video_config_invalid_quality():
    """Ensure invalid quality raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        stv_config.VideoConfig(quality=20, fps=DEFAULT_FPS,
                               save_every=DEFAULT_SAVE_EVERY)
    assert "quality" in str(exc_info.value)


def test_optimization_config_negative_steps():
    """Ensure invalid steps raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        stv_config.OptimizationConfig(steps=-1, style_w=DEFAULT_STYLE_WEIGHT,
                                      content_w=DEFAULT_CONTENT_WEIGHT,
                                      lr=DEFAULT_LEARNING_RATE,
                                      seed=DEFAULT_SEED,
                                      init_method=DEFAULT_INIT_METHOD)
    assert "steps" in str(exc_info.value)


def test_optimization_config_invalid_lr():
    """Ensure invalid learning rate raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        stv_config.OptimizationConfig(lr=0, steps=DEFAULT_STEPS,
                                      style_w=DEFAULT_STYLE_WEIGHT,
                                      content_w=DEFAULT_CONTENT_WEIGHT,
                                      seed=DEFAULT_SEED,
                                      init_method=DEFAULT_INIT_METHOD)
    assert "lr" in str(exc_info.value)


def test_optimization_config_negative_style_w():
    """Ensure invalid style weight raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        stv_config.OptimizationConfig(style_w=-100, steps=DEFAULT_STEPS,
                                      content_w=DEFAULT_CONTENT_WEIGHT,
                                      lr=DEFAULT_LEARNING_RATE,
                                      seed=DEFAULT_SEED,
                                      init_method=DEFAULT_INIT_METHOD)
    assert "style_w" in str(exc_info.value)


def test_optimization_config_negative_content_w():
    """Ensure invalid content weight raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        stv_config.OptimizationConfig(content_w=-5.0, steps=DEFAULT_STEPS,
                                      style_w=DEFAULT_STYLE_WEIGHT,
                                      lr=DEFAULT_LEARNING_RATE,
                                      seed=DEFAULT_SEED,
                                      init_method=DEFAULT_INIT_METHOD)
    assert "content_w" in str(exc_info.value)


def test_optimization_config_negative_seed():
    """Ensure invalid seed raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        stv_config.OptimizationConfig(seed=-42, steps=DEFAULT_STEPS,
                                      style_w=DEFAULT_STYLE_WEIGHT,
                                      content_w=DEFAULT_CONTENT_WEIGHT,
                                      lr=DEFAULT_LEARNING_RATE,
                                      init_method=DEFAULT_INIT_METHOD)
    assert "seed" in str(exc_info.value)
