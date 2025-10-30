"""
Core model components for neural style transfer.

Defines the StyleContentModel, a neural module that extracts feature
representations from VGG19 and computes style and content losses during
optimization. This module serves as the backbone for the style transfer
loop, separating content and style objectives across configurable
layers.

Classes:
    StyleContentModel: Computes activations and losses for content and
    style layers using a frozen VGG19 encoder.
"""

from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import nn
from torch.nn.functional import mse_loss
from torchvision.models import VGG19_Weights, vgg19

from style_transfer_visualizer.config import OptimizationConfig
from style_transfer_visualizer.constants import GRAM_MATRIX_CLAMP_MAX
from style_transfer_visualizer.logging_utils import logger
from style_transfer_visualizer.type_defs import InitMethod, TensorList


def gram_matrix(
    tensor: torch.Tensor,
    clamp_max: float = GRAM_MATRIX_CLAMP_MAX,
) -> torch.Tensor:
    """
    Compute the Gram matrix from feature activations.

    The Gram matrix captures style by measuring channel-wise feature
    correlations. Values are clamped to a maximum threshold to reduce
    the risk of exploding gradients during backpropagation.

    This implementation flattens the batch dimension into the channel
    dimension before computing correlations. For a 4D input tensor of
    shape [batch, channels, height, width], the output is a 2D tensor
    of shape [channels, channels], with batch effects merged into the
    correlation statistics.

    Args:
        tensor: Features with shape [batch, channels, height, width].
        clamp_max: Maximum value for clamping matrix elements to ensure
            numerical stability.

    Returns:
        A 2D Gram matrix tensor of shape [channels, channels],
        normalized and clamped.

    """
    b, c, h, w = tensor.size()
    features = tensor.reshape(b * c, h * w)
    # Clamp to prevent extremely large values that can cause numerical
    # instability
    gram = torch.mm(features, features.t()).clamp(max=clamp_max)

    # Normalize by the total number of elements
    return gram.div(b * c * h * w)


def initialize_input(
    content_img: torch.Tensor,
    method: InitMethod,
) -> torch.Tensor:
    """
    Initialize input tensor for optimization via a specified method.

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
        msg = f"Expected content_img to be a Tensor, got {type(content_img)}"
        raise TypeError(msg)

    input_img: torch.Tensor
    if method == "content":
        input_img = content_img.clone()
    elif method == "random":
        input_img = torch.randn_like(content_img)
    elif method == "white":
        input_img = torch.ones_like(content_img)
    else:
        msg = f"Unsupported initialization method: {method}"
        raise ValueError(msg)

    return input_img.requires_grad_(True)  # noqa: FBT003


def initialize_vgg() -> nn.Module:
    """Load pretrained VGG19 model for feature extraction."""
    weights = VGG19_Weights.IMAGENET1K_V1
    cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
    cache_path = cache_dir / Path(urlparse(weights.url).path).name

    if cache_path.exists():
        logger.info("Using cached VGG19 weights at %s", cache_path)
    else:
        logger.info("Downloading VGG19 weights to %s", cache_path)

    vgg = vgg19(weights=weights).features.eval()
    for p in vgg.parameters():
        p.requires_grad_(False)  # noqa: FBT003
    return vgg


def create_feature_blocks(
    vgg: nn.Module,
    style_layers: list[int],
    content_layers: list[int],
) -> tuple[nn.ModuleList, list[int], list[int]]:
    """Extract sequential feature blocks from VGG19 by layer index."""
    vgg_blocks = nn.ModuleList()
    content_ids = []
    style_ids = []

    block = nn.Sequential()
    for i, layer in enumerate(vgg.children()):
        block.add_module(str(i), layer)

        if isinstance(layer, nn.ReLU):
            block[-1] = nn.ReLU(inplace=False)

        if i in style_layers or i in content_layers:
            vgg_blocks.append(block)
            block = nn.Sequential()

        if i in style_layers:
            style_ids.append(len(vgg_blocks) - 1)
        if i in content_layers:
            content_ids.append(len(vgg_blocks) - 1)

    return vgg_blocks, content_ids, style_ids


class StyleContentModel(nn.Module):
    """
    Manages feature extraction from VGG19 for neural style transfer.

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
        content_layers: list[int],
    ) -> None:
        super().__init__()
        vgg = initialize_vgg()
        self.vgg_blocks, self.content_ids, self.style_ids = \
            create_feature_blocks(vgg, style_layers, content_layers)

        # Targets will be set later
        self.style_targets: list[torch.Tensor] | None = None
        self.content_targets: list[torch.Tensor] | None = None

    def _extract_style_features(
        self,
        style_img: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Extract style features (Gram matrices) from the style image."""
        style_targets = []
        x = style_img
        for j, block in enumerate(self.vgg_blocks):
            x = block(x)
            if j in self.style_ids:
                style_targets.append(gram_matrix(x).detach())
        return style_targets

    def _extract_content_features(
        self,
        content_img: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Extract content features from the content image."""
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
        content_img: torch.Tensor,
    ) -> None:
        """
        Set target features for style and content loss computations.

        Extracts Gram matrices from the style image and activations from
        the content image. This method must be called before the forward
        pass to initialize target feature representations used during
        optimization.
        """
        self.style_targets = self._extract_style_features(style_img)
        self.content_targets = self._extract_content_features(content_img)

    def _compute_style_losses(
        self,
        features: torch.Tensor,
        block_idx: int,
    ) -> torch.Tensor | None:
        """
        Compute style loss for a specific block using Gram matrices.

        Applies MSE loss between the Gram matrix of the current features
        and the corresponding style target. Only active on blocks
        designated as style layers; returns None for all others to avoid
        unnecessary computation.

        Args:
            features: Feature tensor from the current block.
            block_idx: Index of the current block in the VGG hierarchy.

        Returns:
            A scalar tensor representing style loss, or None if the
            block is not part of the style layers.

        """
        if self.style_targets is None:
            msg = "style_targets must be set before computing losses."
            raise RuntimeError(msg)

        if block_idx not in self.style_ids:
            return None
        gram = gram_matrix(features)
        target = self.style_targets[self.style_ids.index(block_idx)]
        return mse_loss(gram, target)

    def _compute_content_losses(
        self,
        features: torch.Tensor,
        block_idx: int,
    ) -> torch.Tensor | None:
        """
        Compute content loss for a specific block using MSE.

        Compares current activations to stored content targets for blocks
        marked as content layers. This measures how well the generated
        image preserves content structure. Returns None for non-content
        blocks to skip unnecessary computation.

        Args:
            features: Feature tensor from the current block.
            block_idx: Index of the current block in the VGG hierarchy.

        Returns:
            A scalar tensor representing content loss, or None if the block
            is not part of the content layers.

        """
        if self.content_targets is None:
            msg = "content_targets must be set before computing losses."
            raise RuntimeError(msg)

        if block_idx not in self.content_ids:
            return None
        target = self.content_targets[self.content_ids.index(block_idx)]
        return mse_loss(features, target)

    def forward(self, x: torch.Tensor) -> tuple[TensorList, TensorList]:
        """
        Compute style and content losses for the input image.

        Passes the input through the model and collects loss values from
        designated style and content layers. Assumes target features have
        already been set via `set_targets()`.

        Args:
            x: Input image tensor of shape [1, C, H, W].

        Returns:
            A tuple of two lists: style losses and content losses, each
            containing scalar tensors.

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


def prepare_model_and_input(
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    device: torch.device,
    optimization: OptimizationConfig,
) -> tuple[nn.Module, torch.Tensor, torch.optim.Optimizer]:
    """Create model, initialize input, and build optimizer."""
    model = StyleContentModel(
        style_layers=optimization.style_layers,
        content_layers=optimization.content_layers,
    ).to(device)
    model.set_targets(style_img, content_img)
    input_img = initialize_input(content_img, optimization.init_method)
    optimizer = torch.optim.LBFGS([input_img], lr=optimization.lr)
    return model, input_img, optimizer
