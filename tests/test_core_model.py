"""
Tests for core model logic in style_transfer_visualizer.

Covers:
- Input initialization strategies
- Gram matrix properties
- StyleContentModel forward loss computation
- Factory: prepare_model_and_input uses OptimizationConfig
"""
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast
from urllib.parse import urlparse

import pytest
import torch
from pytest_mock import MockerFixture
from torch import Tensor

import style_transfer_visualizer.core_model as stv_core_model
from style_transfer_visualizer.config import OptimizationConfig

if TYPE_CHECKING:
    from style_transfer_visualizer.type_defs import InitMethod


class TestInitializeInput:
    """Tests for input tensor initialization strategies."""

    def test_content_method(self, sample_tensor: Tensor) -> None:
        """Test content-based init returns correct values."""
        result = stv_core_model.initialize_input(sample_tensor, "content")
        assert torch.allclose(result, sample_tensor)
        assert result.requires_grad
        assert not sample_tensor.requires_grad

    def test_random_method(self, sample_tensor: Tensor) -> None:
        """Test random init returns non-content values."""
        result = stv_core_model.initialize_input(sample_tensor, "random")
        assert result.shape == sample_tensor.shape
        assert not torch.allclose(result, sample_tensor)
        assert result.requires_grad

    def test_white_method(self, sample_tensor: Tensor) -> None:
        """Test white init returns ones tensor."""
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
        with pytest.raises(TypeError, match=r"Expected content_img.*Tensor"):
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
    """Tests for StyleContentModel loss computation behavior."""

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
        mocker.patch.object(
            model, "_compute_style_losses", return_value=torch.tensor(0.0),
        )

        with pytest.raises(RuntimeError, match="content_targets must be set"):
            model(input_tensor)


class TestFactoryFunction:
    """Tests for prepare_model_and_input factory behavior."""

    def test_prepare_model_and_input_uses_optimization_config(
        self,
        sample_tensor: Tensor,
        mocker: MockerFixture,
    ) -> None:
        """Prepare a model using optimization config."""
        # Build a tiny fake "VGG" to avoid heavyweight downloads.
        fake_vgg = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, 8, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, 8, 3, padding=1),
        )
        mocker.patch.object(stv_core_model, "initialize_vgg",
                            return_value=fake_vgg)

        # Prepare small content/style tensors on CPU.
        device = torch.device("cpu")
        content = sample_tensor.to(device)
        style = sample_tensor.to(device)

        # Construct an OptimizationConfig with simple indices.
        opt_cfg = OptimizationConfig.model_validate({
            "style_layers": [1],  # ReLU index in our fake net
            "content_layers": [2],  # Conv index after first ReLU
            "init_method": "content",
            "lr": 0.5,
        })

        model, input_img, optimizer = stv_core_model.prepare_model_and_input(
            content_img=content,
            style_img=style,
            device=device,
            optimization=opt_cfg,
        )

        # Validate model wiring and optimizer type/params.
        assert isinstance(model, stv_core_model.StyleContentModel)
        assert input_img.requires_grad is True
        assert any(p is input_img for p in optimizer.param_groups[0]["params"])
        assert optimizer.param_groups[0]["lr"] == pytest.approx(opt_cfg.lr)

        # Targets should be set by the factory.
        assert model.style_targets is not None
        assert model.content_targets is not None

    def test_relu_inplace_disabled_in_blocks(
        self,
        mocker: MockerFixture,
    ) -> None:
        """
        Test that ReLU layers inside created blocks are not inplace.

        This safeguards against autograd issues.
        """
        fake_vgg = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3, padding=1),
            torch.nn.ReLU(inplace=True),  # Will be replaced to inplace=False
            torch.nn.Conv2d(4, 4, 3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        mocker.patch.object(stv_core_model, "initialize_vgg",
                            return_value=fake_vgg)

        model = stv_core_model.StyleContentModel([1], [2])
        # Inspect internal blocks for ReLU inplace flags.
        found_relu = False
        for block in model.vgg_blocks:
            for layer in block.children():
                if isinstance(layer, torch.nn.ReLU):
                    found_relu = True
                    assert layer.inplace is False
        assert found_relu is True


class TestInitializeVGG:
    """Tests for logging behavior around VGG19 initialization."""

    @pytest.fixture(autouse=True)
    def _patch_vgg(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Patch vgg19 constructor to avoid heavy model creation."""
        fake_features = mocker.Mock()
        fake_features.eval.return_value = fake_features
        fake_features.parameters.return_value = []
        mock_vgg = mocker.Mock(features=fake_features)
        mocker.patch.object(stv_core_model, "vgg19", return_value=mock_vgg)

    @staticmethod
    def _weight_path(tmp_path: Path) -> Path:
        filename = Path(
            urlparse(stv_core_model.VGG19_Weights.IMAGENET1K_V1.url).path,
        ).name
        return tmp_path / "checkpoints" / filename

    def test_logs_download_when_weights_missing(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Ensure a download notice is logged when weights are absent."""
        caplog.set_level(logging.INFO, logger="style_transfer")
        mocker.patch("torch.hub.get_dir", return_value=str(tmp_path))
        expected_path = self._weight_path(tmp_path)

        stv_core_model.initialize_vgg()

        assert any(
            "Downloading VGG19 weights" in message
            and str(expected_path) in message
            for message in caplog.messages
        )

    def test_logs_cache_hit_when_weights_present(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Ensure a cache notice is logged when weights already exist."""
        caplog.set_level(logging.INFO, logger="style_transfer")
        mocker.patch("torch.hub.get_dir", return_value=str(tmp_path))
        cached_path = self._weight_path(tmp_path)
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        cached_path.touch()

        stv_core_model.initialize_vgg()

        assert any(
            "Using cached VGG19 weights" in message
            and str(cached_path) in message
            for message in caplog.messages
        )
