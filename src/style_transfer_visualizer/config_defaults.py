"""Shared default values for user-facing configuration settings."""

# Optimization
DEFAULT_STEPS = 1500
DEFAULT_STYLE_WEIGHT = 1e5
DEFAULT_CONTENT_WEIGHT = 1.0
DEFAULT_LEARNING_RATE = 1.0
DEFAULT_INIT_METHOD = "random"
DEFAULT_SEED = 0
DEFAULT_NORMALIZE = True
# From torchvision.models.vgg19.
# See:
# https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
# https://medium.com/@ferlatti.aldo/neural-style-transfer-nst-theory-and-implementation-c26728cf969d
STYLE_LAYER_DEFAULTS = [0, 5, 10, 19, 28]
CONTENT_LAYER_DEFAULTS = [21]

# Video
DEFAULT_SAVE_EVERY = 20
DEFAULT_FPS = 10
DEFAULT_VIDEO_QUALITY = 10
DEFAULT_CREATE_VIDEO = True
DEFAULT_FINAL_ONLY = False

# Hardware
DEFAULT_DEVICE = "cuda"

# Output
DEFAULT_OUTPUT_DIR = "out"
