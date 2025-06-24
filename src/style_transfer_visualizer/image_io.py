"""Image loading, preprocessing, and normalization logic."""

import torch
import torchvision.transforms as T
from PIL import Image

from style_transfer_visualizer.constants import (
    COLOR_MODE_RGB, IMAGENET_MEAN, IMAGENET_STD, MIN_DIMENSION,
    MAX_DIMENSION, DENORM_VIEW_SHAPE
)
from style_transfer_visualizer.logging_utils import logger


def load_image(path: str) -> Image.Image:
    """Load an image from a file path and convert to RGB.

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
        raise FileNotFoundError(f"Image file not found: '{path}'") from e
    except IOError as e:
        raise IOError(f"Error loading image '{path}': {str(e)}") from e


def validate_image_dimensions(img: Image.Image) -> None:
    """Ensure image is within minimum and maximum size constraints."""
    if img.width < MIN_DIMENSION or img.height < MIN_DIMENSION:
        raise ValueError(f"Image too small: {img.width}x{img.height}. "
                         f"Minimum dimension is {MIN_DIMENSION}px.")
    if img.width > MAX_DIMENSION or img.height > MAX_DIMENSION:
        logger.warning("Image is large: %dx%d. This may slow"
                       " processing.",img.width, img.height)


def apply_transforms(
    img: Image.Image,
    normalize: bool,
    device: torch.device
) -> torch.Tensor:
    """Convert PIL image to tensor and optionally apply normalization."""
    transforms = [T.ToTensor()]
    if normalize:
        transforms.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    loader = T.Compose(transforms)
    return loader(img).unsqueeze(0).to(device)


def load_image_to_tensor(
    path: str,
    device: torch.device,
    normalize: bool = False
) -> torch.Tensor:
    """Load and preprocess an image for style transfer.

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
    return apply_transforms(img, normalize, device)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization on tensor."""
    mean = torch.tensor(
            IMAGENET_MEAN).view(*DENORM_VIEW_SHAPE).to(tensor.device)
    std = torch.tensor(
            IMAGENET_STD).view(*DENORM_VIEW_SHAPE).to(tensor.device)
    return tensor * std + mean


def prepare_image_for_output(
    tensor: torch.Tensor,
    normalize: bool
) -> torch.Tensor:
    """Prepares a tensor for saving as an image by denormalizing and
    clamping values.

    Args:
        tensor: Image tensor to prepare
        normalize: Whether the tensor uses ImageNet normalization and
        needs denormalization

    Returns:
        Tensor with values clamped to [0,1] range, ready for saving
    """
    img = denormalize(tensor) if normalize else tensor
    img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    return img.clamp(0, 1)
