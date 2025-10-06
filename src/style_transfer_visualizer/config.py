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
    DEFAULT_CONTENT_LAYERS,
    DEFAULT_CONTENT_WEIGHT,
    DEFAULT_CREATE_VIDEO,
    DEFAULT_DEVICE,
    DEFAULT_FINAL_ONLY,
    DEFAULT_FPS,
    DEFAULT_INIT_METHOD,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOG_EVERY,
    DEFAULT_NORMALIZE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SAVE_EVERY,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_STYLE_LAYERS,
    DEFAULT_STYLE_WEIGHT,
    DEFAULT_VIDEO_INTRO_DURATION,
    DEFAULT_VIDEO_INTRO_ENABLED,
    DEFAULT_VIDEO_QUALITY,
)
from style_transfer_visualizer.constants import (
    VIDEO_QUALITY_MAX,
    VIDEO_QUALITY_MIN,
)
from style_transfer_visualizer.type_defs import InitMethod


class OptimizationConfig(BaseModel):
    """Control optimization settings for style transfer."""

    steps: int = Field(DEFAULT_STEPS, ge=1)
    style_w: float = Field(DEFAULT_STYLE_WEIGHT, ge=0)
    content_w: float = Field(DEFAULT_CONTENT_WEIGHT, ge=0)
    lr: float = Field(DEFAULT_LEARNING_RATE, gt=0)
    init_method: InitMethod = Field(DEFAULT_INIT_METHOD)
    seed: int = Field(DEFAULT_SEED, ge=0)
    normalize: bool = DEFAULT_NORMALIZE
    style_layers: list[int] = Field(
        default_factory=lambda: list(DEFAULT_STYLE_LAYERS),
    )
    content_layers: list[int] = Field(
        default_factory=lambda: list(DEFAULT_CONTENT_LAYERS),
    )

class VideoConfig(BaseModel):
    """Control settings for video output."""

    save_every: int = Field(DEFAULT_SAVE_EVERY, ge=1)
    fps: int = Field(DEFAULT_FPS, ge=1, le=60)
    quality: int = Field(
        DEFAULT_VIDEO_QUALITY,
        ge=VIDEO_QUALITY_MIN,
        le=VIDEO_QUALITY_MAX,
    )
    create_video: bool = DEFAULT_CREATE_VIDEO
    final_only: bool = DEFAULT_FINAL_ONLY
    intro_enabled: bool = DEFAULT_VIDEO_INTRO_ENABLED
    intro_duration_seconds: float = Field(
        DEFAULT_VIDEO_INTRO_DURATION,
        ge=0.0,
    )
    metadata_title: str | None = None
    metadata_artist: str | None = None


class HardwareConfig(BaseModel):
    """Select hardware acceleration device."""

    device: str = Field(DEFAULT_DEVICE)


class OutputConfig(BaseModel):
    """Configure output directory and logging interval."""

    output: str = Field(DEFAULT_OUTPUT_DIR)
    log_every: int = Field(DEFAULT_LOG_EVERY, ge=1)
    log_loss: str | None = None
    plot_losses: bool = True


class StyleTransferConfig(BaseModel):
    """
    Root configuration object combining all supported sections.

    Mirrors the structure of config.toml, grouping related parameters
    under logical categories.
    """

    # model_validate({}) allows pyright to be happy playing with Pydantic.
    # Pydantic v2 populates defaults from Field(...) declarations when
    # validating an empty dict. Pyright is satisfied because a zero-arg lambda
    # invokes a classmethod, not a constructor with missing args.
    output: OutputConfig = Field(
        default_factory=lambda: OutputConfig.model_validate({}),
    )
    optimization: OptimizationConfig = Field(
        default_factory=lambda: OptimizationConfig.model_validate({}),
    )
    video: VideoConfig = Field(
        default_factory=lambda: VideoConfig.model_validate({}),
    )
    hardware: HardwareConfig = Field(
        default_factory=lambda: HardwareConfig.model_validate({}),
    )


class ConfigLoader:
    """
    Loads and parses a TOML configuration file into a typed config object.

    Falls back to defaults for any missing subsections or fields.
    """

    @staticmethod
    def load(path: str) -> StyleTransferConfig:
        """
        Load a style transfer configuration from a TOML file.

        Returns a validated StyleTransferConfig instance based on the file
        contents.
        """
        config_path = Path(path)
        if not config_path.is_file():
            msg = f"Config file not found: {path}"
            raise FileNotFoundError(msg)

        with config_path.open("r", encoding="utf-8") as f:
            doc = tomlkit.load(f)

        return StyleTransferConfig.model_validate(doc)
