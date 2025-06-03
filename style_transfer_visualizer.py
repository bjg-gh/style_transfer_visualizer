#!/usr/bin/env python3
"""style_transfer_visualizer.py — Neural Style Transfer Script (PyTorch)

This script performs neural style transfer using a VGG19-based feature
loss.  It supports content and style images, customizable optimization
parameters, and generates a timelapse MP4 video and loss plot from the
transfer process.

Key Features:
- LBFGS optimizer with white/random/content image initialization
- Frame-by-frame image saving with loss tracking
- Optional timelapse video and matplotlib loss plot
- Clean modular structure with logging and command-line arguments

Usage:
    python style_transfer_visualizer.py --content <content.jpg> \
        --style <style.jpg> [options]

Note:
    Input images must be pre-sized by the user. Minimum size: 64px;
    processing may be slow above 3000px.
"""

import argparse
import imageio
import time
import logging
import random
from pathlib import Path
from typing import Literal, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision.models import vgg19, VGG19_Weights
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from tqdm import tqdm

from __version__ import __version__

# Type aliases for improved readability
LossMetrics = Dict[str, List[float]]
TensorList = List[torch.Tensor]
InitMethod = Literal["content", "random", "white"]

# Constants
VERSION = __version__
SEED = 0

# Video encoding constants
DEFAULT_INIT_METHOD = "random"
DEFAULT_FPS = 10
DEFAULT_VIDEO_QUALITY = 10
VIDEO_CODEC = "libx264"
ASPECT_RATIO_16_9 = 16 / 9
ENCODING_BLOCK_SIZE = 16  # Videos are encoded in 16x16 macroblocks

# From torchvision.models.vgg19.
# See:
# https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
# https://medium.com/@ferlatti.aldo/neural-style-transfer-nst-theory-and-implementation-c26728cf969d
STYLE_LAYERS = [0, 5, 10, 19, 28]
CONTENT_LAYERS = [21]

# Standard ImageNet normalization values used in torchvision.models
# See: https://pytorch.org/vision/stable/models.html#classification
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Numerical stability constants
GRAM_MATRIX_CLAMP_MAX = 5e5

# Image processing constants
COLOR_MODE_RGB = "RGB"
COLOR_BLACK = (0, 0, 0)
MIN_DIMENSION = 64
MAX_DIMENSION = 3000

# Tensor reshaping constant to broadcast normalization values across image
# tensor
DENORM_VIEW_SHAPE = (1, 3, 1, 1)


def setup_logger(
        name: str = __name__,
        level: int = logging.INFO,
        formatter: logging.Formatter = None,
        handler: logging.Handler = None
) -> logging.Logger:
    """Configure and return a module-level logger with optional custom
    settings.

    Args:
        name: Name of the module
        level: Level of the logger
        formatter: Format for log messages
        handler: Custom handler for log messages

    Returns:
        Logger: A module-level logger with optional custom settings.
    """
    logger_instance = logging.getLogger(name)
    logger_instance.setLevel(level)
    if not logger_instance.handlers:
        if formatter is None:
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s")
        if handler is None:
            handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger_instance.addHandler(handler)
        logger_instance.propagate = False
    return logger_instance


logger = setup_logger()


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization to convert tensor back to RGB values.
    
    Args:
        tensor: Normalized image tensor with ImageNet mean and std
        
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(
            IMAGENET_MEAN).view(*DENORM_VIEW_SHAPE).to(tensor.device)
    std = torch.tensor(
            IMAGENET_STD).view(*DENORM_VIEW_SHAPE).to(tensor.device)
    return tensor * std + mean


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
        # Re-raise FileNotFoundError directly
        raise FileNotFoundError(f"Image file not found: '{path}'") from e
    except IOError as e:
        raise IOError(f"Error loading image '{path}': {str(e)}") from e


def validate_image_dimensions(img: Image.Image) -> None:
    """Ensure the image dimensions fall within allowed range.

    Args:
        img: PIL Image

    Raises:
        ValueError: If image dimensions are too small or too large
    """
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
    """Convert a PIL Image to a tensor and optionally normalize.

    Args:
        img: Input PIL Image
        normalize: Whether to apply ImageNet normalization
        device: Device to load the tensor to

    Returns:
        Transformed tensor on the specified device
    """
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


def gram_matrix(
        tensor: torch.Tensor,
        clamp_max: float = GRAM_MATRIX_CLAMP_MAX
) -> torch.Tensor:
    """Computes Gram matrix from feature activations for style
    representation.

    The Gram matrix captures style information by measuring feature
    correlations. Values are clamped to prevent numerical instability
    during backpropagation.
    
    Note on dimensions: This implementation flattens the batch dimension
    into the channel dimension before computing the Gram matrix. This
    means a 4D input tensor of shape [batch, channels, height, width]
    results in a 2D Gram matrix of shape [channels, channels], with
    batch information integrated into the correlation values. This
    approach is optimized for single-image style transfer where batch=1.

    Args:
        tensor: Feature tensor of shape [batch, channels, height, width]
        clamp_max: Maximum value to clamp the matrix elements to prevent
            numerical instability

    Returns:
        Normalized Gram matrix with correlation coefficients as a 2D
        tensor of shape [channels, channels]
    """
    b, c, h, w = tensor.size()
    features = tensor.reshape(b * c, h * w)
    # Clamp to prevent extremely large values that can cause numerical
    # instability
    G = torch.mm(features, features.t()).clamp(max=clamp_max)  # pylint: disable=invalid-name

    # Normalize by the total number of elements
    return G.div(b * c * h * w)


def initialize_input(
        content_img: torch.Tensor,
        method: InitMethod
) -> torch.Tensor:
    """Initialize input tensor for optimization via a specified method.

    Args:
        content_img: Content image tensor to use as reference
        method: Initialization method ("content", "random", or "white")

    Returns:
        Initialized tensor with requires_grad=True

    Raises:
        ValueError: If method is not one of the supported initialization
            methods
    """
    if not isinstance(content_img, torch.Tensor):
        raise TypeError(
            "Expected content_img to be a torch.Tensor,"
            f" got {type(content_img).__name__}")

    if method == "content":
        input_img = content_img.clone()
    elif method == "random":
        input_img = torch.randn_like(content_img)
    elif method == "white":
        input_img = torch.ones_like(content_img)
    else:
        raise ValueError(
            f"Unknown init method: {method}. Expected one of: content, "
            f"random, white")

    return input_img.requires_grad_(True)


def initialize_vgg() -> nn.Module:
    """Initialize and freeze the VGG19 model.

    Returns:
        nn.Module: Frozen VGG19 features module in eval mode
    """
    vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
    for p in vgg.parameters():
        p.requires_grad_(False)
    return vgg


def create_feature_blocks(
        vgg: nn.Module,
        style_layers: list[int],
        content_layers: list[int]
) -> tuple[nn.ModuleList, list[int], list[int]]:
    """Create feature blocks from VGG19 model based on specified
    layer indices.

    Args:
        vgg: The VGG19 model features
        style_layers: Indices of style features layers
        content_layers: Indices of content features layers

    Returns:
        Tuple of (vgg_blocks, content_ids, style_ids)
    """
    vgg_blocks = nn.ModuleList()
    content_ids = []
    style_ids = []

    i = 0
    block = nn.Sequential()
    for layer in vgg.children():
        block.add_module(str(i), layer)

        # Replace inplace ReLU to avoid modifying shared features
        if isinstance(layer, nn.ReLU):
            block[-1] = nn.ReLU(inplace=False)

        # If we've reached a target layer, save the block and start a new one
        if i in style_layers or i in content_layers:
            vgg_blocks.append(block)
            block = nn.Sequential()

        # Track which blocks correspond to style and content layers
        if i in style_layers:
            style_ids.append(len(vgg_blocks) - 1)
        if i in content_layers:
            content_ids.append(len(vgg_blocks) - 1)

        i += 1

    return vgg_blocks, content_ids, style_ids


class StyleContentModel(nn.Module):
    """Manages feature extraction from VGG19 for neural style transfer.

    This model slices a pretrained VGG19 network into sequential blocks
    based on specified layer indices for extracting different levels of
    features. Each block's activations are used to compute either style
    or content losses during optimization. The model precomputes target
    activations from the style and content images and stores them for
    comparison during forward passes.

    The style representation uses Gram matrices of features, while
    content representation uses direct feature activations, following
    Gatys et al.

    Attributes:
        vgg_blocks (nn.ModuleList): Sequential feature blocks from
            VGG19.
        style_ids (list[int]): Indices of slices used for style
            features.
        content_ids (list[int]): Indices of slices used for content
            features.
        style_targets (list[Tensor]): Precomputed Gram matrices from
            style image.
        content_targets (list[Tensor]): Precomputed activations from
            content image.
    """

    def __init__(
            self,
            style_layers: list[int],
            content_layers: list[int]
    ) -> None:
        super().__init__()

        # Step 1: Initialize and freeze the VGG model
        vgg = initialize_vgg()

        # Step 2: Create feature blocks and track indices
        self.vgg_blocks, self.content_ids, self.style_ids = \
            create_feature_blocks(vgg, style_layers, content_layers)

        # Initialize targets (will be set later)
        self.style_targets = None
        self.content_targets = None

    def _extract_style_features(
            self,
            style_img: torch.Tensor
    ) -> list[torch.Tensor]:
        """Extract style features (Gram matrices) from the style image.

        Args:
            style_img: The style image tensor

        Returns:
            list[torch.Tensor]: List of Gram matrices for style features
        """
        style_targets = []
        x = style_img

        for j, block in enumerate(self.vgg_blocks):
            x = block(x)
            if j in self.style_ids:
                style_targets.append(gram_matrix(x).detach())

        return style_targets

    def _extract_content_features(
            self,
            content_img: torch.Tensor
    ) -> list[torch.Tensor]:
        """Extract content features from the content image.

        Args:
            content_img: The content image tensor

        Returns:
            list[torch.Tensor]: Feature tensors for content features
        """
        content_targets = []
        x = content_img

        for j, block in enumerate(self.vgg_blocks):
            x = block(x)
            if j in self.content_ids:
                content_targets.append(x.detach())

        return content_targets

    def set_targets(
            self,
            style_img: torch.Tensor,
            content_img: torch.Tensor
    ) -> None:
        """Extracts and stores feature representations from style and
        content images.

        Must be called before the forward pass to establish target
        features for both style (Gram matrices) and content features
        (direct activations) that will be used during optimization.

        Args:
            style_img: Style image tensor to extract style features from
            content_img: Content image tensor to extract content
                features from
        """
        # Extract style features (Gram matrices)
        self.style_targets = self._extract_style_features(style_img)

        # Extract content features
        self.content_targets = self._extract_content_features(content_img)

    def _compute_style_losses(
            self,
            features: torch.Tensor,
            block_idx: int
    ) -> Optional[torch.Tensor]:
        """Computes MSE loss between Gram matrices of current features
        and style targets.

        Only processes blocks that have been designated as style layers
        during initialization.  For non-style blocks, returns None to
        avoid unnecessary computation.

        Args:
            features: The feature tensor from the current block
            block_idx: The index of the current block

        Returns:
            Optional[torch.Tensor]: Style loss tensor if this is a style
            block, None otherwise
        """
        if block_idx not in self.style_ids:
            return None

        G = gram_matrix(features)  # pylint: disable=invalid-name
        target = self.style_targets[self.style_ids.index(block_idx)]
        return nn.functional.mse_loss(G, target)

    def _compute_content_losses(
            self,
            features: torch.Tensor,
            block_idx: int
    ) -> Optional[torch.Tensor]:
        """Computes MSE loss between current features and content
        targets.

        Only processes blocks that have been designated as content
        layers during initialization.  This measures how well content
        features are preserved in the generated image.

        Args:
            features: The feature tensor from the current block
            block_idx: The index of the current block

        Returns:
            Optional[torch.Tensor]: Content loss tensor if this is a
            content block, None otherwise
        """
        if block_idx not in self.content_ids:
            return None

        target = self.content_targets[self.content_ids.index(block_idx)]
        return nn.functional.mse_loss(features, target)

    def forward(self, x: torch.Tensor) -> Tuple[TensorList, TensorList]:
        """Forward pass through the model to compute style and content
        losses.

        Args:
            x: Input image tensor

        Returns:
            Tuple[TensorList, TensorList]: Lists of style and content
                losses
        """
        style_losses, content_losses = [], []

        for j, block in enumerate(self.vgg_blocks):
            # Pass input through the current block
            x = block(x)

            # Compute style loss if this is a style block
            style_loss = self._compute_style_losses(x, j)
            if style_loss is not None:
                style_losses.append(style_loss)

            # Compute content loss if this is a content block
            content_loss = self._compute_content_losses(x, j)
            if content_loss is not None:
                content_losses.append(content_loss)

        return style_losses, content_losses


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


def log_parameters(args: argparse.Namespace) -> None:
    """Logs all user-provided command-line parameters.

    Provides a record of the exact configuration used for a particular
    style transfer run.

    Args:
        args: Namespace containing parsed command-line arguments
    """
    logger.info("Content image loaded: %s", args.content)
    logger.info("Style image loaded: %s", args.style)
    logger.info("Output Directory: %s", args.output)
    logger.info("Steps: %d", args.steps)
    logger.info("Save Every: %d", args.save_every)
    logger.info("Style Weight: %g", args.style_w)
    logger.info("Content Weight: %g", args.content_w)
    logger.info("Learning Rate: %g", args.lr)
    logger.info("FPS for Timelapse Video: %d", args.fps)
    logger.info("Video Quality: %d (1–10 scale)", args.quality)
    logger.info("Initialization Method: %s", args.init_method)
    logger.info(
        "Normalization: %s",
        "Enabled" if not args.no_normalize else "Disabled")
    logger.info(
        "Video Creation: %s",
        "Disabled" if args.no_video else "Enabled")
    logger.info("Random Seed: %d", args.seed)


def prepare_model_and_input(
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        device: torch.device,
        init_method: InitMethod = DEFAULT_INIT_METHOD,
        learning_rate: float = 1.0
) -> Tuple[nn.Module, torch.Tensor, optim.Optimizer]:
    """Initialize the model and input image for style transfer.

    Args:
        content_img: Content image tensor
        style_img: Style image tensor
        device: Device to run the model on
        init_method: Method to initialize the input image
        learning_rate: Learning rate for the optimizer

    Returns:
        Tuple of (model, input_img, optimizer)
    """
    model = StyleContentModel(STYLE_LAYERS, CONTENT_LAYERS).to(device)
    model.set_targets(style_img, content_img)
    input_img = initialize_input(content_img, init_method)
    optimizer = optim.LBFGS([input_img], lr=learning_rate)
    return model, input_img, optimizer


def validate_input_paths(content_path: str, style_path: str) -> None:
    """Validate that the content and style image paths exist.

    Args:
        content_path: Path to the content image
        style_path: Path to the style image

    Raises:
        FileNotFoundError: If either image file doesn't exist
    """
    if not Path(content_path).is_file():
        raise FileNotFoundError(f"Content image not found: {content_path}")
    if not Path(style_path).is_file():
        raise FileNotFoundError(f"Style image not found: {style_path}")


def validate_parameters(video_quality: int) -> None:
    """Validates that parameters fall within acceptable ranges.

    Currently only checks video quality, but can be expanded to validate
    other parameters as the function signature suggests.

    Args:
        video_quality: Quality setting for output video (1-10)

    Raises:
        ValueError: If parameters are outside acceptable ranges
    """
    if video_quality < 1 or video_quality > 10:
        raise ValueError(
            f"Video quality must be between 1 and 10, got {video_quality}")


def setup_random_seed(seed: int) -> None:
    """Sets random seeds for all libraries.

    Ensures deterministic behavior by setting seeds for PyTorch (both
    CPU and CUDA), Python's random module, and NumPy. This is important
    for scientific reproducibility and debugging.

    Args:
        seed: Random seed value used for all random number generators
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Set Python's random seed as well for complete reproducibility
    random.seed(seed)
    # Set NumPy random seed if available
    np.random.seed(seed)


def setup_device(device_name: str) -> torch.device:
    """Set up the device for computation.

    Args:
        device_name: Device to run on ("cuda" or "cpu")

    Returns:
        torch.device: The device to use
    """
    if device_name == "cuda" and not torch.cuda.is_available():
        logger.warning(
            "CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    logger.info("Using device: %s", device)
    return device


def setup_output_directory(output_path, path_factory=Path):
    """Create and return the output directory path.

    Args:
        output_path: Directory to save outputs
        path_factory: Factory for creating paths - test hook

    Returns:
        Path: Path object for the output directory
    """
    output_path = path_factory(output_path)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception:
        fallback_path = path_factory("style_transfer_output")
        fallback_path.mkdir(parents=True, exist_ok=True)
        return fallback_path
    return output_path


def setup_video_writer(
        output_path: Path,
        video_name: str,
        fps: int,
        video_quality: int,
        create_video: bool
) -> Optional[imageio.plugins.ffmpeg.FfmpegFormat.Writer]:
    """Set up the video writer if video creation is enabled.

    Args:
        output_path: Path to the output directory
        video_name: Name of the video file
        fps: Frames per second for the video
        video_quality: Quality setting for the video (1-10)
        create_video: Whether to create a video

    Returns:
        Optional video writer object
    """
    if not create_video:
        return None

    return imageio.get_writer(
        output_path / video_name,
        fps=fps,
        codec=VIDEO_CODEC,
        quality=video_quality,
        mode="I",  # Explicitly set mode for clarity
        # Ensure compatibility with encoding blocks
        macro_block_size=ENCODING_BLOCK_SIZE
    )


def optimization_step(
        model: nn.Module,
        input_img: torch.Tensor,
        optimizer: optim.Optimizer,
        style_weight: float,
        content_weight: float,
        loss_metrics: LossMetrics,
        step: int,
        save_every: int,
        video_writer: Optional[imageio.plugins.ffmpeg.FfmpegFormat.Writer],
        normalize: bool,
        progress_bar: tqdm
) -> float:
    """Perform a single optimization step.

    Args:
        model: The style transfer model
        input_img: The input image tensor
        optimizer: The optimizer
        style_weight: Weight for style loss
        content_weight: Weight for content loss
        loss_metrics: Dictionary to track loss metrics
        step: Current step number
        save_every: Save frame every N steps
        video_writer: Video writer object (if video creation is enabled)
        normalize: Whether to use ImageNet normalization
        progress_bar: Progress bar object

    Returns:
        float: The current loss value
    """
    optimizer.zero_grad()
    style_losses, content_losses = model(input_img)
    style_score = sum(style_losses)
    content_score = sum(content_losses)
    loss = style_weight * style_score + content_weight * content_score
    loss.backward()

    # Check for non-finite values in both style and content scores
    if not torch.isfinite(style_score):
        logger.warning("Non-finite style score at step %d", step)

    if not torch.isfinite(content_score):
        logger.warning("Non-finite content score at step %d", step)

    if not torch.isfinite(loss):
        logger.warning(
            "Non-finite total loss at step %d, using previous loss", step)

    logger.debug(
        "Step %d: Style %s, Content %.4e, Total %.4e",
        step, [s.item() for s in style_losses], content_score.item(),
        loss.item()
    )

    loss_metrics["style_loss"].append(style_score.item())
    loss_metrics["content_loss"].append(content_score.item())
    loss_metrics["total_loss"].append(loss.item())

    if step % save_every == 0:
        with torch.no_grad():
            img = prepare_image_for_output(input_img, normalize)
            if img is not None and video_writer is not None:
                img_np = (img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                          * 255).astype("uint8")
                video_writer.append_data(img_np)
        progress_bar.set_postfix({
            "style": f"{style_score.item():.4f}",
            "content": f"{content_score.item():.4f}",
            "loss": f"{loss.item():.4f}"
        })

    progress_bar.update(1)
    return loss.item()


def run_optimization_loop(
        model: nn.Module,
        input_img: torch.Tensor,
        optimizer: optim.Optimizer,
        steps: int,
        save_every: int,
        style_weight: float,
        content_weight: float,
        normalize: bool,
        video_writer: Optional[imageio.plugins.ffmpeg.FfmpegFormat.Writer]
) -> Tuple[torch.Tensor, LossMetrics, float]:
    """Run the optimization loop for style transfer.

    Args:
        model: The style transfer model
        input_img: The input image tensor
        optimizer: The optimizer
        steps: Number of optimization steps
        save_every: Save frame every N steps
        style_weight: Weight for style loss
        content_weight: Weight for content loss
        normalize: Whether to use ImageNet normalization
        video_writer: Video writer object (if video creation is enabled)

    Returns:
        (final image tensor, loss metrics dictionary, elapsed time)
    """
    # Track loss metrics
    loss_metrics = {"style_loss": [], "content_loss": [], "total_loss": []}

    # Initialize progress tracking
    step = 0
    progress_bar = tqdm(total=steps, desc="Style Transfer")
    start_time = time.time()

    # Define optimization closure for LBFGS
    def closure() -> float:
        nonlocal step
        loss = optimization_step(
            model,
            input_img,
            optimizer,
            style_weight,
            content_weight,
            loss_metrics,
            step,
            save_every,
            video_writer,
            normalize,
            progress_bar)
        step += 1
        return loss

    # Run optimization
    while step < steps:
        optimizer.step(closure)

    # Clean up
    progress_bar.close()
    elapsed = time.time() - start_time

    return input_img, loss_metrics, elapsed


def style_transfer(
        content_path: str,
        style_path: str,
        output_dir: str = "out",
        steps: int = 300,
        save_every: int = 20,
        style_weight: float = 1e6,
        content_weight: float = 1,
        learning_rate: float = 1.0,
        fps: int = DEFAULT_FPS,
        device_name: str = "cuda",
        init_method: InitMethod = DEFAULT_INIT_METHOD,
        normalize: bool = True,
        create_video: bool = True,
        final_only: bool = False,
        video_quality: int = DEFAULT_VIDEO_QUALITY,
        seed: int = SEED
) -> torch.Tensor:
    """Perform neural style transfer and save outputs.

    Args:
        content_path: Path to the content image
        style_path: Path to the style image
        output_dir: Directory to save outputs
        steps: Number of optimization steps
        save_every: Save frame every N steps
        style_weight: Weight for style loss
        content_weight: Weight for content loss
        learning_rate: Learning rate for optimizer
        fps: Frames per second for timelapse video
        device_name: Device to run on ("cuda" or "cpu")
        init_method: Method to initialize the input image
        normalize: Whether to use ImageNet normalization
        create_video: Whether to create a timelapse video
        final_only: Whether to only save the final image
        video_quality: Quality setting for output video (1-10)
        seed: Random seed for reproducibility

    Returns:
        The final stylized image tensor

    Note:
        Input images must be pre-sized by the user. Minimum size: 64px;
        processing may be slow above 3000px.
    """
    # Step 1: Validate inputs and parameters
    validate_input_paths(content_path, style_path)
    validate_parameters(video_quality)

    # Step 2: Adjust parameters for final_only mode
    if final_only:
        create_video = False
        save_every = steps + 1

    # Step 3: Set up environment
    setup_random_seed(seed)
    device = setup_device(device_name)

    # Step 4: Load and preprocess images
    content_img = load_image_to_tensor(content_path, device,
                                       normalize=normalize)
    style_img = load_image_to_tensor(style_path, device, normalize=normalize)

    # Step 5: Initialize model and optimizer
    model, input_img, optimizer = prepare_model_and_input(
        content_img,
        style_img,
        device,
        init_method,
        learning_rate
    )

    # Step 6: Set up output directory and file names
    output_path = setup_output_directory(output_dir)
    content_name = Path(content_path).stem
    style_name = Path(style_path).stem
    video_name = f"timelapse_{content_name}_x_{style_name}.mp4"

    # Step 7: Initialize video writer if needed
    video_writer = setup_video_writer(
        output_path,
        video_name,
        fps,
        video_quality,
        create_video)

    # Step 8: Run optimization loop
    input_img, loss_metrics, elapsed = run_optimization_loop(
        model, input_img, optimizer, steps, save_every,
        style_weight, content_weight, normalize, video_writer
    )

    # Step 9: Clean up and save outputs
    if video_writer:
        video_writer.close()

    save_outputs(
        input_img,
        loss_metrics,
        output_path,
        elapsed,
        content_name,
        style_name,
        video_name,
        normalize,
        create_video
    )

    return input_img.detach().clamp(0, 1)


def plot_loss_curves(metrics: LossMetrics, output_dir: Path) -> None:
    """Plot and save the loss curves using matplotlib.

    Args:
        metrics: Dictionary of loss metrics over time
        output_dir: Directory to save the plot
    """
    # Check if we have any metrics to plot
    if not metrics:
        logger.warning("No loss metrics dictionary provided.")
        return

    if not any(len(values) > 0 for values in metrics.values()):
        logger.warning("Loss metrics dictionary is empty, nothing to plot.")
        return

    try: # import is here because this functionality is optional
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ImportError:
        logger.warning("matplotlib not found: skipping loss plot.")
        return

    fig = plt.figure(figsize=(10, 6))
    try:
        for k in metrics:
            if metrics[k]:  # Only plot if we have data
                plt.plot(metrics[k], label=k)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Loss Curves")
        plt.legend()
        plt.tight_layout()
        loss_plot_path = output_dir / "loss_plot.png"
        plt.savefig(loss_plot_path)
        logger.info("Loss plot saved to: %s", loss_plot_path)
    finally:
        plt.close(fig)  # Ensure figure is closed to prevent memory leaks


def save_outputs(
        input_img: torch.Tensor,
        loss_metrics: LossMetrics,
        output_dir: Path,
        elapsed: float,
        content_name: str,
        style_name: str,
        video_name: Optional[str] = None,
        normalize: bool = True,
        video_created: bool = True
) -> None:
    """Save the final image, timelapse video, and loss plot to disk.

    Args:
        input_img: The final stylized image tensor
        loss_metrics: Dictionary of loss metrics over time
        output_dir: Directory to save outputs
        elapsed: Time elapsed during style transfer
        content_name: Name of the content image (without extension)
        style_name: Name of the style image (without extension)
        video_name: Name of the video file (if created)
        normalize: Whether to denormalize the image before saving
        video_created: Whether a video was created during optimization
    """
    # Step 1: Ensure output directory exists
    try:
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created output directory: %s", output_dir)
    except (PermissionError, OSError) as e:
        logger.error("Failed to create output directory: %s", e)
        # Create a fallback directory in the current working directory
        fallback_dir = Path("style_transfer_output")
        fallback_dir.mkdir(exist_ok=True)
        logger.info("Using fallback directory: %s", fallback_dir)
        output_dir = fallback_dir

    # Step 2: Save the final stylized image
    final_path = output_dir / f"stylized_{content_name}_x_{style_name}.png"
    img_to_save = prepare_image_for_output(input_img, normalize)
    save_image(img_to_save, final_path)

    # Step 3: Log video information
    if video_created and video_name:
        # Video already written during optimization
        logger.info("Video saved to: %s", output_dir / video_name)

    # Step 4: Create and save loss plot
    plot_loss_curves(loss_metrics, output_dir)

    # Step 5: Log completion information
    logger.info("Style transfer completed in %.2f seconds", elapsed)
    logger.info("Final stylized image saved to: %s", final_path)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the command-line interface."""
    p = argparse.ArgumentParser(
        description="Neural Style Transfer with PyTorch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # pylint: disable=line-too-long
        epilog=f"""
Examples:
python {Path(__file__).name} --content cat.jpg --style starry_night.jpg
python {Path(__file__).name} --content cat.jpg --style starry_night.jpg --final-only
python {Path(__file__).name} --content cat.jpg --style starry_night.jpg --steps 1000 --fps 30

Note:
  Normalization is enabled by default. Use --no-normalize to disable it
"""
        # pylint: enable=line-too-long
    )

    # General arguments
    p.add_argument("--version", action="version",
                   version=f"%(prog)s {VERSION}")

    # Input/Output group
    io_group = p.add_argument_group("Input/Output Options")
    io_group.add_argument("--content", required=True,
                          help="content image path")
    io_group.add_argument("--style", required=True,
                          help="style image path")
    io_group.add_argument("--output", default="out",
                          help="output directory")

    # Optimization parameters group
    optim_group = p.add_argument_group("Optimization Parameters")
    optim_group.add_argument("--steps", type=int, default=300,
                             help="number of optimization steps")
    optim_group.add_argument("--style-w", type=float, default=1e6,
                             help="style weight")
    optim_group.add_argument("--content-w", type=float, default=1,
                             help="content weight")
    optim_group.add_argument("--lr", type=float, default=1.0,
                             help="learning rate")
    optim_group.add_argument("--init-method", default=DEFAULT_INIT_METHOD,
                             choices=["content", "random", "white"],
                             help="initialization method")
    optim_group.add_argument("--no-normalize", action="store_true",
                             dest="no_normalize", default=False,
                             help="disable ImageNet normalization "
                                  "(normalization is enabled by default)")
    optim_group.add_argument("--seed", type=int, default=SEED,
                             help="random seed for reproducibility "
                                  f"(default: {SEED})")

    # Video output options group
    video_group = p.add_argument_group("Video Output Options")
    video_group.add_argument("--save-every", type=int, default=20,
                             help="save frame every N steps")
    video_group.add_argument("--fps", type=int, default=DEFAULT_FPS,
                             help="fps for timelapse video")
    video_group.add_argument("--no-video", action="store_true",
                             help="skip creating timelapse video")
    video_group.add_argument("--final-only", action="store_true",
                             help="only save final stylized image")
    video_group.add_argument("--quality", type=int,
                             default=DEFAULT_VIDEO_QUALITY,
                             help="quality setting for output video "
                                  f"(1–{DEFAULT_VIDEO_QUALITY}, "
                                  "higher is better)")

    # Hardware options group
    hw_group = p.add_argument_group("Hardware Options")
    hw_group.add_argument("--device", default="cuda",
                          choices=["cpu", "cuda"],
                          help="force device (default: cuda)")

    return p


def run_from_args(args: argparse.Namespace) -> torch.Tensor:
    """Run style transfer from command-line arguments."""
    log_parameters(args)

    return style_transfer(
        content_path=args.content,
        style_path=args.style,
        output_dir=args.output,
        steps=args.steps,
        save_every=args.save_every,
        style_weight=args.style_w,
        content_weight=args.content_w,
        learning_rate=args.lr,
        fps=args.fps,
        device_name=args.device,
        init_method=args.init_method,
        normalize=not args.no_normalize,
        create_video=not args.no_video,
        final_only=args.final_only,
        video_quality=args.quality,
        seed=args.seed
    )


def main() -> None:
    """Main entry point for the CLI."""
    args = build_arg_parser().parse_args()
    run_from_args(args)


if __name__ == "__main__":
    main()
