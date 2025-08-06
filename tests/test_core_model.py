"""
Tests for core model logic in style_transfer_visualizer.

Covers:
- Input initialization strategies
- Gram matrix properties
- StyleContentModel forward loss computation
- Optimization step and loop
"""

from typing import TYPE_CHECKING, cast

import pytest
import torch
from pytest_mock import MockerFixture
from torch import Tensor

import style_transfer_visualizer.core_model as stv_core_model

if TYPE_CHECKING:
    from style_transfer_visualizer.type_defs import InitMethod


class TestInitializeInput:
    """Test for input tensor initialization strategies."""

    def test_content_method(self, sample_tensor: Tensor) -> None:
        """Test content-based initialization returns correct values."""
        result = stv_core_model.initialize_input(sample_tensor, "content")
        assert torch.allclose(result, sample_tensor)
        assert result.requires_grad
        assert not sample_tensor.requires_grad

    def test_random_method(self, sample_tensor: Tensor) -> None:
        """Test random initialization returns non-content values."""
        result = stv_core_model.initialize_input(sample_tensor, "random")
        assert result.shape == sample_tensor.shape
        assert not torch.allclose(result, sample_tensor)
        assert result.requires_grad

    def test_white_method(self, sample_tensor: Tensor) -> None:
        """Test white initialization returns ones tensor."""
        result = stv_core_model.initialize_input(sample_tensor, "white")
        assert torch.allclose(result, torch.ones_like(sample_tensor))
        assert result.requires_grad

    def test_invalid_method(self, sample_tensor: Tensor) -> None:
        """Test unknown init method raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported initialization"):
            stv_core_model.initialize_input(
                sample_tensor, cast("InitMethod", "invalid"),
            )

    def test_invalid_input_type(self) -> None:
        """Test non-tensor input raises TypeError."""
        with pytest.raises(TypeError, match="Expected content_img.*Tensor"):
            stv_core_model.initialize_input(
                cast("torch.Tensor", "not_a_tensor"),
                "content",
            )

    @pytest.mark.parametrize("device_name", ["cpu", "cuda"])
    def test_device_preservation(
        self,
        sample_tensor: Tensor,
        device_name: str,
    ) -> None:
        """Test output remains on the same device as input tensor."""
        if device_name == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device_name)
        tensor = sample_tensor.to(device)
        for method in ["content", "random", "white"]:
            result = stv_core_model.initialize_input(
                tensor, cast("InitMethod", method),
            )
            assert result.device.type == device.type


def test_gram_matrix_properties(sample_tensor: Tensor) -> None:
    """Test symmetry and shape of computed Gram matrix."""
    _, c, _, _ = sample_tensor.shape
    gram = stv_core_model.gram_matrix(sample_tensor)
    assert isinstance(gram, torch.Tensor)
    assert gram.shape == (c, c)
    assert torch.allclose(gram, gram.t())
    eigvals = torch.linalg.eigvals(gram).real
    assert torch.all(eigvals >= -1e-6)  # noqa: PLR2004


class TestStyleContentModel:
    """Test for StyleContentModel loss computation behavior."""

    def test_forward_loss_accumulation(
        self,
        style_layers: list[int],
        content_layers: list[int],
    ) -> None:
        """Test forward pass accumulates correct number of losses."""
        model = stv_core_model.StyleContentModel(style_layers, content_layers)
        input_tensor = torch.randn(1, 3, 128, 128)
        model.set_targets(input_tensor, input_tensor)
        s_losses, c_losses = model(input_tensor)
        assert len(s_losses) == len(style_layers)
        assert len(c_losses) == len(content_layers)
        for loss in s_losses + c_losses:
            assert loss.dim() == 0

    def test_forward_raises_if_style_targets_not_set(self) -> None:
        """Test RuntimeError is raised if style_targets is not set."""
        model = stv_core_model.StyleContentModel([1], [2])
        input_tensor = torch.randn(1, 3, 128, 128)
        model.content_targets = [input_tensor]  # Set only content targets
        with pytest.raises(RuntimeError, match="style_targets must be set"):
            model(input_tensor)

    def test_forward_raises_if_content_targets_not_set(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test RuntimeError is raised if content_targets is not set."""
        model = stv_core_model.StyleContentModel([1], [2])
        input_tensor = torch.randn(1, 3, 64, 64)
        model.style_targets = [input_tensor]

        # Mock style loss computation to bypass unrelated errors
        mocker.patch.object(model, "_compute_style_losses",
                            return_value=[torch.tensor(0.0)])

        with pytest.raises(RuntimeError, match="content_targets must be set"):
            model(input_tensor)
