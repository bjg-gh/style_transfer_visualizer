"""
Defines shared type aliases for the style transfer visualizer.

Centralizes reusable type hints to improve consistency and readability.
"""

from typing import Literal

import torch

InitMethod = Literal["content", "random", "white"]
TensorList = list[torch.Tensor]
LossHistory = dict[str, list[float]]
