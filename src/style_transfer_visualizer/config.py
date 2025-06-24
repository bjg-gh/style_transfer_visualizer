"""
Configuration schema and loader for the Style Transfer Visualizer.

Defines Pydantic models representing structured configuration sections
and a TOML-based config loader with validation support.
"""

from pathlib import Path

import tomlkit
from pydantic import BaseModel, Field

# Import internal constants for shared use
from style_transfer_visualizer.config_defaults import (
    DEFAULT_STEPS,
    DEFAULT_STYLE_WEIGHT,
    DEFAULT_CONTENT_WEIGHT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_INIT_METHOD,
    DEFAULT_SEED,
    DEFAULT_NORMALIZE,
    DEFAULT_SAVE_EVERY,
    DEFAULT_FPS,
    DEFAULT_VIDEO_QUALITY,
    DEFAULT_CREATE_VIDEO,
    DEFAULT_FINAL_ONLY,
    DEFAULT_DEVICE,
    DEFAULT_OUTPUT_DIR
)


class OptimizationConfig(BaseModel):
    steps: int = Field(DEFAULT_STEPS, ge=1)
    style_w: float = Field(DEFAULT_STYLE_WEIGHT, ge=0)
    content_w: float = Field(DEFAULT_CONTENT_WEIGHT, ge=0)
    lr: float = Field(DEFAULT_LEARNING_RATE, gt=0)
    init_method: str = Field(DEFAULT_INIT_METHOD)
    seed: int = Field(DEFAULT_SEED, ge=0)
    normalize: bool = DEFAULT_NORMALIZE


class VideoConfig(BaseModel):
    save_every: int = Field(DEFAULT_SAVE_EVERY, ge=1)
    fps: int = Field(DEFAULT_FPS, ge=1, le=60)
    quality: int = Field(DEFAULT_VIDEO_QUALITY, ge=1, le=10)
    create_video: bool = DEFAULT_CREATE_VIDEO
    final_only: bool = DEFAULT_FINAL_ONLY


class HardwareConfig(BaseModel):
    device: str = Field(DEFAULT_DEVICE)


class OutputConfig(BaseModel):
    output: str = Field(DEFAULT_OUTPUT_DIR)


class StyleTransferConfig(BaseModel):
    """
    Root configuration object combining all supported sections.

    Mirrors the structure of config.toml, grouping related parameters
    under logical categories.
    """
    output: OutputConfig = Field(default_factory=OutputConfig)
    optimization: OptimizationConfig = Field(default_factory
                                             =OptimizationConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)


class ConfigLoader:
    """
    Loads and parses a TOML configuration file into a typed config object.

    Falls back to defaults for any missing subsections or fields.
    """

    @staticmethod
    def load(path: str) -> StyleTransferConfig:
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")

        with config_path.open("r", encoding="utf-8") as f:
            doc = tomlkit.load(f)

        return StyleTransferConfig.model_validate(doc)
