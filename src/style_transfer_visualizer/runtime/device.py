"""Device configuration and deterministic runtime helpers."""

from __future__ import annotations

import random

import torch

from style_transfer_visualizer import random_utils as stv_random
from style_transfer_visualizer.logging_utils import logger


def setup_device(device_name: str) -> torch.device:
    """
    Return the torch device to use for execution.

    Falls back to CPU when CUDA is requested but unavailable and always logs
    the selected device for observability.
    """
    if device_name == "cuda" and not torch.cuda.is_available():
        logger.warning(
            "CUDA requested but not available. Falling back to CPU.",
        )
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    logger.info("Using device: %s", device)
    return device


def setup_random_seed(seed: int) -> None:
    """
    Seed all supported random number generators for determinism.

    Currently seeds torch (CPU and CUDA), NumPy, and Python's
    ``random`` module. Extend this helper if additional frameworks
    are introduced.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    stv_random.seed_numpy_rng(seed)
