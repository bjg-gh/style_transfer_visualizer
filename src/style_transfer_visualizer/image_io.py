"""Image loading, preprocessing, and normalization logic."""
from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

import torch
from PIL import Image
from torchvision import transforms

from style_transfer_visualizer.constants import (
    COLOR_MODE_RGB,
    DENORM_VIEW_SHAPE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAX_DIMENSION,
    MIN_DIMENSION,
)
from style_transfer_visualizer.logging_utils import logger


def load_image(path: str) -> Image.Image:
    """
    Load an image from a file path and convert to RGB.

    Args:
        path: Path to the image file

    Returns:
        PIL Image in RGB mode

    Raises:
        FileNotFoundError: If the image file does not exist
        IOError: If the image cannot be opened or processed

    """
    try:
        return Image.open(path).convert(COLOR_MODE_RGB)
    except FileNotFoundError as e:
        msg = f"Image file not found: '{path}'"
        raise FileNotFoundError(msg) from e
    except OSError as e:
        msg = f"Error loading image '{path}': {e!s}"
        raise OSError(msg) from e


def validate_image_dimensions(img: Image.Image) -> None:
    """Ensure image is within minimum and maximum size constraints."""
    if img.width < MIN_DIMENSION or img.height < MIN_DIMENSION:
        msg = (f"Image too small: {img.width}x{img.height}. "
               f"Minimum dimension is {MIN_DIMENSION}px."
               )
        raise ValueError(msg)
    if img.width > MAX_DIMENSION or img.height > MAX_DIMENSION:
        logger.warning(
            "Image is large: %dx%d. This may slow processing.",
            img.width,
            img.height,
        )


def apply_transforms(
    img: Image.Image,
    device: torch.device,
    *,
    normalize: bool,
) -> torch.Tensor:
    """Convert PIL image to tensor and optionally apply normalization."""
    pipeline: list[Callable[[Image.Image], torch.Tensor]] = [
        transforms.ToTensor(),
    ]
    if normalize:
        pipeline.append(
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        )
    loader: Callable[[Image.Image], torch.Tensor]
    loader = transforms.Compose(pipeline)
    tensor = cast("torch.Tensor", loader(img))
    return tensor.unsqueeze(0).to(device)


def load_image_to_tensor(
    path: str,
    device: torch.device,
    *,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Load and preprocess an image for style transfer.

    Loads image as-is (no resizing or padding). Validates dimensions
    and applies optional normalization.

    Args:
        path: Path to the image file
        device: Device to load the tensor to
        normalize: Whether to apply ImageNet normalization

    Returns:
        Preprocessed image tensor on the specified device

    Raises:
        FileNotFoundError: If the image file doesn't exist
        IOError: If the image cannot be opened or processed
        ValueError: If image dimensions are invalid

    """
    img = load_image(path)
    validate_image_dimensions(img)
    return apply_transforms(img, device, normalize=normalize)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization on tensor."""
    mean = torch.tensor(IMAGENET_MEAN).view(*DENORM_VIEW_SHAPE).to(
        tensor.device,
    )
    std = torch.tensor(IMAGENET_STD).view(*DENORM_VIEW_SHAPE).to(
        tensor.device,
    )
    return tensor * std + mean


def prepare_image_for_output(
    tensor: torch.Tensor,
    *,
    normalize: bool,
) -> torch.Tensor:
    """
    Prepare an image tensor for saving.

    If the tensor was normalized using ImageNet statistics, reverse the
    normalization. Then replace NaNs and infinities for numerical
    stability and clamp values to the [0, 1] range.

    Args:
        tensor: Input image tensor to process.
        normalize: Whether to apply ImageNet-style denormalization.

    Returns:
        A clamped image tensor with values in the [0, 1] range,
        suitable for saving.

    """
    img = denormalize(tensor) if normalize else tensor
    img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    return img.clamp(0, 1)
