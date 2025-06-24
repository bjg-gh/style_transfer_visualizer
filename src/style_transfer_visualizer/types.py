"""
types.py

Shared type aliases used in the style transfer visualizer project.

This file centralizes common type definitions.
"""
from typing import Dict, List, Literal

import torch

InitMethod = Literal["content", "random", "white"]
TensorList = List[torch.Tensor]
LossMetrics = Dict[str, List[float]]
