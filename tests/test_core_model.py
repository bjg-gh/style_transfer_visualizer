"""Tests for core model logic in style_transfer_visualizer.

Covers:
- Input initialization strategies
- Gram matrix properties
- StyleContentModel forward loss computation
- Optimization step and loop
"""

from typing import Any, cast

import pytest
import torch
from torch import Tensor

import style_transfer_visualizer.core_model as stv_core_model
from style_transfer_visualizer.types import InitMethod


class TestInitializeInput:
    def test_content_method(self, sample_tensor: Tensor):
        """Test content-based initialization returns correct values."""
        result = stv_core_model.initialize_input(sample_tensor, "content")
        assert torch.allclose(result, sample_tensor)
        assert result.requires_grad
        assert not sample_tensor.requires_grad

    def test_random_method(self, sample_tensor: Tensor):
        """Test random initialization returns non-content values."""
        result = stv_core_model.initialize_input(sample_tensor, "random")
        assert result.shape == sample_tensor.shape
        assert not torch.allclose(result, sample_tensor)
        assert result.requires_grad

    def test_white_method(self, sample_tensor: Tensor):
        """Test white initialization returns ones tensor."""
        result = stv_core_model.initialize_input(sample_tensor, "white")
        assert torch.allclose(result, torch.ones_like(sample_tensor))
        assert result.requires_grad

    def test_invalid_method(self, sample_tensor: Tensor):
        """Test unknown init method raises ValueError."""
        with pytest.raises(ValueError):
            stv_core_model.initialize_input(sample_tensor, cast(Any,
                                                                "invalid"))

    def test_invalid_input_type(self):
        """Test non-tensor input raises TypeError."""
        with pytest.raises(TypeError):
            stv_core_model.initialize_input(cast(Any, "not_a_tensor"),
                                            "content")

    @pytest.mark.parametrize("device_name", ["cpu", "cuda"])
    def test_device_preservation(self,
                                  sample_tensor: Tensor,
                                  device_name: str):
        """Test output remains on the same device as input tensor."""
        if device_name == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device_name)
        tensor = sample_tensor.to(device)
        for method in ["content", "random", "white"]:
            result = stv_core_model.initialize_input(
                tensor, cast(InitMethod, method)
            )
            assert result.device.type == device.type


def test_gram_matrix_properties(sample_tensor: Tensor):
    """Test symmetry and shape of computed Gram matrix."""
    _, c, _, _ = sample_tensor.shape
    gram = stv_core_model.gram_matrix(sample_tensor)
    assert isinstance(gram, torch.Tensor)
    assert gram.shape == (c, c)
    assert torch.allclose(gram, gram.t())
    eigvals = torch.linalg.eigvals(gram).real  # type: ignore[attr-defined]
    assert torch.all(eigvals >= -1e-6)


class TestStyleContentModel:
    def test_forward_loss_accumulation(self,
                                       style_layers: list[int],
                                       content_layers: list[int]):
        """Test forward pass accumulates correct number of losses."""
        model = stv_core_model.StyleContentModel(style_layers, content_layers)
        input_tensor = torch.randn(1, 3, 128, 128)
        model.set_targets(input_tensor, input_tensor)
        s_losses, c_losses = model(input_tensor)
        assert len(s_losses) == len(style_layers)
        assert len(c_losses) == len(content_layers)
        for loss in s_losses + c_losses:
            assert loss.dim() == 0
