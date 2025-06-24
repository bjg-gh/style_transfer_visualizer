"""Tests for image I/O and preprocessing in style_transfer_visualizer.

Covers:
- Loading images and error handling
- Tensor transformation and normalization
- Device compatibility and data integrity
"""

import tempfile
from typing import Tuple

import pytest
import torch
from PIL import Image, ImageDraw
from torch import Tensor

from style_transfer_visualizer.constants import COLOR_MODE_RGB
import style_transfer_visualizer.image_io as stv_image_io


class TestImageLoading:
    def test_load_image_valid(self, content_image: str):
        """Test loading a valid image path."""
        img = stv_image_io.load_image(content_image)
        assert isinstance(img, Image.Image)
        assert img.mode == COLOR_MODE_RGB

    def test_load_image_invalid_path(self):
        """Test that nonexistent image path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            stv_image_io.load_image("nonexistent_image.jpg")

    def test_load_image_invalid_data(self):
        """Test that invalid image content raises IOError."""
        with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            f.write(b"not an image data")
            f.flush()
            with pytest.raises(Exception):
                stv_image_io.load_image(f.name)

    @pytest.mark.parametrize("image_size", [(100, 100), (224, 224),
                                            (512, 512)])
    def test_load_image_different_sizes(self,
                                        image_size: Tuple[int, int],
                                        test_dir: str):
        """Test loading images of various sizes."""
        width, height = image_size
        img = Image.new("RGB", (width, height))
        path = f"{test_dir}/test_img.jpg"
        img.save(path)
        loaded = stv_image_io.load_image(path)
        assert loaded.size == (width, height)

    @pytest.mark.parametrize("device_name", ["cpu", "cuda"])
    def test_device_loading(self, device_name: str, content_image: str):
        """Test loading image to specified device."""
        if device_name == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device_name)
        img = stv_image_io.load_image(content_image)
        tensor = stv_image_io.apply_transforms(img, normalize=True,
                                               device=device)
        assert tensor.device.type == device.type


class TestTransforms:
    def test_denormalize_tensor(self, sample_tensor: Tensor):
        """Test denormalizing a tensor restores original values."""
        denorm = stv_image_io.denormalize(sample_tensor)
        assert isinstance(denorm, torch.Tensor)
        assert denorm.shape == sample_tensor.shape
        assert not torch.allclose(denorm, sample_tensor)

    def test_denormalize_multi_batch(self):
        """Test that denormalize handles batch size > 1."""
        tensor = torch.randn(2, 3, 100, 100)
        result = stv_image_io.denormalize(tensor)
        assert result.shape[0] == 2

    def test_apply_transforms_normalized(self, test_device: torch.device):
        """Test normalized tensor shape and device."""
        img = Image.new(COLOR_MODE_RGB, (100, 100))
        t = stv_image_io.apply_transforms(img, normalize=True,
                                          device=test_device)
        assert t.shape[0] == 1 and t.shape[1] == 3
        assert t.device.type == test_device.type

    def test_apply_transforms_black_white(self, test_device: torch.device):
        """Test normalized pixel ranges for black and white images."""
        black = Image.new(COLOR_MODE_RGB, (100, 100))
        t_black = stv_image_io.apply_transforms(black, normalize=True,
                                                device=test_device)
        assert t_black.min().item() < -2

        white = Image.new(COLOR_MODE_RGB, (100, 100), color=(255, 255, 255))
        t_white = stv_image_io.apply_transforms(white, normalize=True,
                                                device=test_device)
        assert t_white.max().item() > 2.2

    def test_apply_transforms_without_normalization(
        self, test_device: torch.device
    ):
        """Test unnormalized values for black, white, and mixed images."""
        black = Image.new(COLOR_MODE_RGB, (100, 100))
        t = stv_image_io.apply_transforms(black, normalize=False,
                                          device=test_device)
        assert t.min().item() == 0.0 and t.max().item() == 0.0

        white = Image.new(COLOR_MODE_RGB, (100, 100), color=(255, 255, 255))
        t = stv_image_io.apply_transforms(white, normalize=False,
                                          device=test_device)
        assert t.min().item() == 1.0 and t.max().item() == 1.0

        mixed = Image.new(COLOR_MODE_RGB, (100, 100))
        draw = ImageDraw.Draw(mixed)
        draw.rectangle((0, 0, 50, 50), fill=(255, 255, 255))
        t = stv_image_io.apply_transforms(mixed, normalize=False,
                                          device=test_device)
        assert t.min().item() == 0.0 and t.max().item() == 1.0

class TestImageValidation:
    def test_valid_dimensions(self, caplog):
        """Test that valid dimensions work."""
        img = Image.new(COLOR_MODE_RGB, (512, 512))
        stv_image_io.validate_image_dimensions(img)  # Should not raise

    def test_too_small_dimensions(self):
        """Test that too small of a dimension raises ValidationError."""
        img = Image.new(COLOR_MODE_RGB, (32, 100))
        with pytest.raises(ValueError, match="Image too small"):
            stv_image_io.validate_image_dimensions(img)

    def test_too_large_dimensions(self, caplog):
        """Test that too large of a dimension generates a warning."""
        caplog.set_level("WARNING")
        img = Image.new(COLOR_MODE_RGB, (4000, 4000))
        stv_image_io.validate_image_dimensions(img)
        assert "may slow processing" in caplog.text


class TestImageOutputPreparation:
    def test_prepare_image_with_normalization(self):
        """Test normalization and clamping behavior."""
        tensor = torch.tensor([
            [[-3.0, -2.0, -1.0], [0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        ]).view(1, 1, 3, 3).repeat(1, 3, 1, 1)

        out = stv_image_io.prepare_image_for_output(tensor, normalize=True)
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert not torch.allclose(out, tensor.clamp(0, 1))
        assert out.shape == tensor.shape

    def test_prepare_image_without_normalization(self):
        """Test clamping behavior without normalization."""
        tensor = torch.tensor([[-0.5, 0.2, 0.7], [0.0, 1.0, 1.5]])
        tensor = tensor.view(1, 1, 2, 3).repeat(1, 3, 1, 1)
        out = stv_image_io.prepare_image_for_output(tensor, normalize=False)
        assert torch.allclose(out, tensor.clamp(0, 1))
        assert out.shape == tensor.shape

    def test_extreme_values(self):
        """Test clamping for large magnitude values."""
        tensor = torch.tensor([[-100.0, -50.0, 0.0], [1.0, 50.0, 100.0]])
        tensor = tensor.view(1, 1, 2, 3).repeat(1, 3, 1, 1)
        for norm in [True, False]:
            out = stv_image_io.prepare_image_for_output(tensor,
                                                        normalize=norm)
            assert out.min() >= 0.0 and out.max() <= 1.0
            assert out.shape == tensor.shape

    @pytest.mark.parametrize("device_name", ["cpu", "cuda"])
    def test_device_preserved(self, device_name: str):
        """Test output remains on the original device."""
        if device_name == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device_name)
        tensor = torch.rand(1, 3, 10, 10).to(device)
        for norm in [True, False]:
            out = stv_image_io.prepare_image_for_output(tensor,
                                                        normalize=norm)
            assert out.device.type == device.type

    def test_batch_handling(self):
        """Test image output handles batch size > 1."""
        batch = torch.rand(2, 3, 10, 10)
        out = stv_image_io.prepare_image_for_output(batch, normalize=True)
        assert out.shape == batch.shape
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_nan_handling(self):
        """Test NaN and infinity values are sanitized."""
        tensor = torch.tensor([[[[float("nan"),
                                   float("inf"),
                                   -float("inf")]]]]).repeat(1, 3, 1, 1)
        out = stv_image_io.prepare_image_for_output(tensor, normalize=False)
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert out.shape == tensor.shape
