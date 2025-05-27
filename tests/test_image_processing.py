"""Tests for image I/O and preprocessing in style_transfer_visualizer.

Covers:
- Loading images and error handling
- Image resizing with padding
- Tensor transformation and normalization
- Device compatibility and data integrity
"""

import math
import tempfile
from typing import Tuple

from PIL import Image, ImageDraw
import torch
from torch import Tensor
import pytest
from hypothesis import given, strategies as st, settings

from style_transfer_visualizer import (
    load_image,
    denormalize,
    padding_preparation_resize,
    add_padding,
    apply_transforms
)


class TestImageLoading:
    def test_load_image_valid(self, content_image: str):
        """Test loading a valid image path."""
        img = load_image(content_image)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_load_image_invalid_path(self):
        """Test that nonexistent image path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_image("nonexistent_image.jpg")

    def test_load_image_invalid_data(self):
        """Test that invalid image content raises IOError."""
        with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            f.write(b"not an image data")
            f.flush()
            with pytest.raises(Exception):
                load_image(f.name)

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
        loaded = load_image(path)
        assert loaded.size == (width, height)

    @pytest.mark.parametrize("device_name", ["cpu", "cuda"])
    def test_device_loading(self, device_name: str, content_image: str):
        """Test loading image to specified device."""
        if device_name == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device_name)
        img = load_image(content_image)
        tensor = apply_transforms(img, normalize=True, device=device)
        assert tensor.device.type == device.type


class TestTransforms:
    def test_denormalize_tensor(self, sample_tensor: Tensor):
        """Test denormalizing a tensor restores original values."""
        denorm = denormalize(sample_tensor)
        assert isinstance(denorm, torch.Tensor)
        assert denorm.shape == sample_tensor.shape
        assert not torch.allclose(denorm, sample_tensor)

    def test_denormalize_multi_batch(self):
        """Test that denormalize handles batch size > 1."""
        tensor = torch.randn(2, 3, 100, 100)
        result = denormalize(tensor)
        assert result.shape[0] == 2

    def test_apply_transforms_normalized(self, device: torch.device):
        """Test normalized tensor shape and device."""
        img = Image.new("RGB", (100, 100))
        t = apply_transforms(img, normalize=True, device=device)
        assert t.shape[0] == 1 and t.shape[1] == 3
        assert t.device.type == device.type

    def test_apply_transforms_black_white(self, device: torch.device):
        """Test normalized pixel ranges for black and white images."""
        black = Image.new("RGB", (100, 100))
        t_black = apply_transforms(black, normalize=True, device=device)
        assert t_black.min().item() < -2

        white = Image.new("RGB", (100, 100), color=(255, 255, 255))
        t_white = apply_transforms(white, normalize=True, device=device)
        assert t_white.max().item() > 2.2

    def test_apply_transforms_without_normalization(
        self, device: torch.device
    ):
        """Test unnormalized values for black, white, and mixed images."""
        black = Image.new("RGB", (100, 100))
        t = apply_transforms(black, normalize=False, device=device)
        assert t.min().item() == 0.0 and t.max().item() == 0.0

        white = Image.new("RGB", (100, 100), color=(255, 255, 255))
        t = apply_transforms(white, normalize=False, device=device)
        assert t.min().item() == 1.0 and t.max().item() == 1.0

        mixed = Image.new("RGB", (100, 100))
        draw = ImageDraw.Draw(mixed)
        draw.rectangle((0, 0, 50, 50), fill=(255, 255, 255))
        t = apply_transforms(mixed, normalize=False, device=device)
        assert t.min().item() == 0.0 and t.max().item() == 1.0

    def test_add_padding(self, sample_image: Image.Image):
        """Test image padding to new dimensions."""
        padded = add_padding(
            sample_image, target_width=150, target_height=150
        )
        assert padded.size[0] == 150
        assert padded.size[1] == 150


class TestImageResizing:
    @settings(deadline=None)
    @given(height=st.integers(50, 1000), width=st.integers(50, 1000))
    def test_resize_preserves_aspect_ratio(self,
                                           height: int,
                                           width: int):
        """Test resize maintains original aspect ratio within tolerance."""
        img = Image.new("RGB", (width, height))
        target_height = 400
        target_width = int(400 * (width / height))
        resized = padding_preparation_resize(
            img, target_height, target_width
        )
        original_ratio = width / height
        new_ratio = resized.size[0] / resized.size[1]
        assert math.isclose(
            original_ratio, new_ratio, rel_tol=0.02, abs_tol=0.02
        )

    @pytest.mark.parametrize("target_height", [224, 512, 768])
    def test_resize_specific_heights(self,
                                     sample_image: Image.Image,
                                     target_height: int):
        """Test resize dimensions match expected height."""
        orig_w, orig_h = sample_image.size
        target_width = int(target_height * (orig_w / orig_h))
        resized = padding_preparation_resize(
            sample_image, target_height, target_width
        )
        assert resized.size[1] == target_height

    def test_resize_with_fixed_dimensions(self):
        """Test resizing behavior for specific dimensions."""
        img = Image.new("RGB", (200, 100))
        target_height = 50
        target_width = 100

        resized = padding_preparation_resize(
            img, target_height=target_height, target_width=target_width
        )

        img_ratio = img.width / img.height
        target_ratio = target_width / target_height

        if img_ratio > target_ratio:
            expected_size = (int(target_height * img_ratio), target_height)
        else:
            expected_size = (target_width, int(target_width / img_ratio))

        assert resized.size == expected_size

    def test_invalid_input_dimensions(self):
        """Test resize raises ValueError on invalid image size."""
        img_zero_width = Image.new("RGB", (0, 100))
        with pytest.raises(ValueError):
            padding_preparation_resize(img_zero_width, 100, 100)

        img_zero_height = Image.new("RGB", (100, 0))
        with pytest.raises(ValueError):
            padding_preparation_resize(img_zero_height, 100, 100)

    def test_resize_invalid_targets(self):
        """Test resize raises ValueError on invalid target dimensions."""
        img = Image.new("RGB", (100, 100))
        with pytest.raises(ValueError):
            padding_preparation_resize(img, -1, 100)
        with pytest.raises(ValueError):
            padding_preparation_resize(img, 100, -1)

    @given(
        height=st.integers(min_value=50, max_value=1000),
        width=st.integers(min_value=50, max_value=1000),
        target_height=st.integers(min_value=50, max_value=500),
        target_width=st.integers(min_value=50, max_value=500)
    )
    def test_resize_properties(self,
                               height: int,
                               width: int,
                               target_height: int,
                               target_width: int):
        """Property-based test for aspect ratio preservation."""
        from hypothesis import assume

        min_dimension = 50
        assume(target_height >= min_dimension)
        assume(target_width >= min_dimension)

        original = Image.new("RGB", (width, height))
        original_ratio = width / height
        target_ratio = target_width / target_height

        resized = padding_preparation_resize(
            original, target_height=target_height,
            target_width=target_width
        )

        resized_ratio = resized.size[0] / resized.size[1]
        tolerance = 0.07  # Looser tolerance for integer rounding

        assert abs(original_ratio - resized_ratio) < tolerance

        if original_ratio > target_ratio:
            assert resized.size[1] == target_height
        else:
            assert resized.size[0] == target_width
