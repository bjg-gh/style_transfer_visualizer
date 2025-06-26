"""Core model logic:
 StyleContentModel, Gram matrix, input initialization.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

from style_transfer_visualizer.config_defaults import DEFAULT_LBFGS_LR, \
    DEFAULT_OPTIMIZER
from style_transfer_visualizer.constants import (
    STYLE_LAYERS, CONTENT_LAYERS, GRAM_MATRIX_CLAMP_MAX
)
from style_transfer_visualizer.types import InitMethod, TensorList


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
        raise TypeError("Expected torch.Tensor,"
                        f" got {type(content_img)}")

    if method == "content":
        input_img = content_img.clone()
    elif method == "random":
        input_img = torch.randn_like(content_img)
    elif method == "white":
        input_img = torch.ones_like(content_img)
    else:
        raise ValueError(
            f"Unknown init method: {method}. Expected one of: content, "
            f"random, white"
        )
    return input_img.requires_grad_(True)


def initialize_vgg() -> nn.Module:
    """Load pretrained VGG19 model for feature extraction."""
    vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
    for p in vgg.parameters():
        p.requires_grad_(False)
    return vgg


def create_feature_blocks(
    vgg: nn.Module,
    style_layers: list[int],
    content_layers: list[int]
) -> tuple[nn.ModuleList, list[int], list[int]]:
    """Extract sequential feature blocks from VGG19 by layer index."""
    vgg_blocks = nn.ModuleList()
    content_ids = []
    style_ids = []

    i = 0
    block = nn.Sequential()
    for layer in vgg.children():
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
        vgg = initialize_vgg()
        self.vgg_blocks, self.content_ids, self.style_ids = \
            create_feature_blocks(vgg, style_layers, content_layers)

        # Targets will be set later
        self.style_targets = None
        self.content_targets = None

    def _extract_style_features(
        self,
        style_img: torch.Tensor
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
        content_img: torch.Tensor
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
        content_img: torch.Tensor
    ) -> None:
        """Extracts and stores feature representations from style and
        content images.

        Must be called before the forward pass to establish target
        features for both style (Gram matrices) and content features
        (direct activations) that will be used during optimization.
        """
        self.style_targets = self._extract_style_features(style_img)
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
        return F.mse_loss(G, target)

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
        return F.mse_loss(features, target)

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


def prepare_model_and_input(
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    device: torch.device,
    init_method: InitMethod = "random",
    optimizer_name: str = DEFAULT_OPTIMIZER,
    learning_rate: float = DEFAULT_LBFGS_LR
) -> Tuple[nn.Module, torch.Tensor, torch.optim.Optimizer]:
    """Initialize the model and input image for style transfer.

    Args:
        content_img: Content image tensor
        style_img: Style image tensor
        device: Device to run the model on
        init_method: Method to initialize the input image
        optimizer_name: Optimizer name: "lbfgs" or "adam"
        learning_rate: Learning rate for the optimizer

    Returns:
        Tuple of (model, input_img, optimizer)
    """
    model = StyleContentModel(STYLE_LAYERS, CONTENT_LAYERS).to(device)
    model.set_targets(style_img, content_img)
    input_img = initialize_input(content_img, init_method)

    if optimizer_name == "lbfgs":
        optimizer = torch.optim.LBFGS([input_img], lr=learning_rate)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam([input_img], lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    return model, input_img, optimizer
