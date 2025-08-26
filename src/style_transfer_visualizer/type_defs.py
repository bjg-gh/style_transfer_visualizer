"""
Defines shared type aliases for the style transfer visualizer.

Centralizes reusable type hints to improve consistency and readability.
"""
from dataclasses import dataclass
from typing import Literal

import torch

InitMethod = Literal["content", "random", "white"]
TensorList = list[torch.Tensor]
LossHistory = dict[str, list[float]]

@dataclass(slots=True)
class InputPaths:
    """Content and style input image paths."""

    content_path: str
    style_path: str

@dataclass(slots=True)
class SaveOptions:
    """Names and output flags for the final save step."""

    content_name: str
    style_name: str
    video_name: str | None = None
    normalize: bool = True
    video_created: bool = True
    plot_losses: bool = True
