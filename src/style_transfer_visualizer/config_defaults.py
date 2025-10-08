"""Shared default values for user-facing configuration settings."""
from style_transfer_visualizer.type_defs import InitMethod

# Optimization
DEFAULT_STEPS = 1500
DEFAULT_STYLE_WEIGHT = 1e5
DEFAULT_CONTENT_WEIGHT = 1.0
DEFAULT_LEARNING_RATE = 1.0
DEFAULT_INIT_METHOD: InitMethod = "random"
DEFAULT_SEED = 0
DEFAULT_NORMALIZE = True
# From torchvision.models.vgg19.
# See:
# https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
# https://medium.com/@ferlatti.aldo/neural-style-transfer-nst-theory-and-implementation-c26728cf969d
DEFAULT_STYLE_LAYERS: tuple[int, ...] = (0, 5, 10, 19, 28)
DEFAULT_CONTENT_LAYERS: tuple[int, ...] = (21,)

# Video
DEFAULT_SAVE_EVERY = 20
DEFAULT_FPS = 10
DEFAULT_VIDEO_QUALITY = 10
DEFAULT_CREATE_VIDEO = True
DEFAULT_FINAL_ONLY = False
DEFAULT_VIDEO_INTRO_ENABLED = True
DEFAULT_VIDEO_INTRO_DURATION = 10.0
DEFAULT_VIDEO_OUTRO_DURATION = 10.0
DEFAULT_VIDEO_FINAL_FRAME_COMPARE = True

# Hardware
DEFAULT_DEVICE = "cuda"

# Output
DEFAULT_LOG_EVERY = 10
DEFAULT_OUTPUT_DIR = "out"
