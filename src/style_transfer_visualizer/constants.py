"""
Constants used internally by the Style Transfer Visualizer.

These are implementation-level defaults that should not be overridden
via config files or CLI arguments.
"""

# Standard ImageNet normalization values used in torchvision.models
# See: https://pytorch.org/vision/stable/models.html#classification
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Clamp max value for gram matrix stabilization
GRAM_MATRIX_CLAMP_MAX = 5e5

# Video encoding
VIDEO_CODEC = "libx264"
ENCODING_BLOCK_SIZE = 16

# Video quality
VIDEO_QUALITY_MAX = 10
VIDEO_QUALITY_MIN = 1

# Image processing constants
MIN_DIMENSION = 64
MAX_DIMENSION = 3000

# Internal color constants
COLOR_MODE_RGB = "RGB"
COLOR_BLACK = (0, 0, 0)

# Shape used for denormalization broadcasting
DENORM_VIEW_SHAPE = (1, 3, 1, 1)

# Loss logging
CSV_LOGGING_RECOMMENDED_STEPS = 2000
