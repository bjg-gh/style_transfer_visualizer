"""
Neural style transfer with timelapse video, loss logging, and CLI.

Modules:
    core_model         — Model construction and initialization
    image_io           — Image loading, preprocessing, normalization
    optimization       — Optimization loop and training step logic
    video              — Timelapse video writer configuration
    cli                — Command-line interface and argument parsing
    config             — Typed configuration schema and TOML loader
    config_defaults    — Shared default values
    constants          — Internal constants for model and video encoding
    type_defs          — Shared type aliases
    logging_utils      — Application-wide logger and setup
    loss_logger        — CSV loss metrics logger
    main               — High-level pipeline orchestration
"""
