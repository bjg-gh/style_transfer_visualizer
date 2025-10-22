"""CLI argument parsing and main entry point."""

import argparse
import sys
from pathlib import Path

import style_transfer_visualizer.config as stv_config
import style_transfer_visualizer.main as stv_main
from style_transfer_visualizer.config_defaults import (
    DEFAULT_LOG_EVERY,
    DEFAULT_VIDEO_INTRO_DURATION,
    DEFAULT_VIDEO_OUTRO_DURATION,
)
from style_transfer_visualizer.constants import (
    VIDEO_QUALITY_MAX,
    VIDEO_QUALITY_MIN,
)
from style_transfer_visualizer.logging_utils import logger
from style_transfer_visualizer.runtime.comparison import (
    ComparisonRequest,
    render_requested_comparisons,
)
from style_transfer_visualizer.type_defs import InputPaths


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the command-line interface."""
    p = argparse.ArgumentParser(
        description="Neural Style Transfer with PyTorch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Examples:\n"
            f"python {Path(__file__).name} --content cat.jpg "
            f"--style starry_night.jpg\n"
            f"python {Path(__file__).name} --content cat.jpg "
            f"--style starry_night.jpg --final-only\n"
            f"python {Path(__file__).name} --content cat.jpg "
            f"--style starry_night.jpg --steps 1000 --fps 30\n\n"
            "Note:\n"
            "  Normalization is enabled by default. "
            "Use --no-normalize to disable it"
        ),
    )


    required = p.add_argument_group("required arguments")
    required.add_argument(
        "--content", type=str, help="Path to content image")
    required.add_argument(
        "--style", type=str, help="Path to style image")

    output = p.add_argument_group("output")
    output.add_argument(
        "--output", type=str, help="Output directory",
        default=argparse.SUPPRESS)
    output.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable loss plotting",
    )
    output.add_argument(
        "--log-loss", type=str,
        help=(
            "Path to CSV file for logging loss metrics. When enabled, "
            "loss metrics are written directly to disk instead of kept in "
            "memory, and matplotlib loss plotting is automatically disabled."
        ),
    )
    output.add_argument(
        "--log-every", type=int, default=DEFAULT_LOG_EVERY,
        help=(
            "Log losses to CSV every N steps (default: "
            f"{DEFAULT_LOG_EVERY}). Ignored if --log-loss is not set."
        ),
    )
    output.add_argument(
        "--compare-inputs",
        action="store_true",
        help=(
            "Save a labeled comparison image of content and style to the "
            "output directory and exit."
        ),
    )
    output.add_argument(
        "--compare-result",
        action="store_true",
        help=(
            "Save a labeled comparison image of content, style, and result to "
            "the output directory and exit. The stylized image path is "
            "derived from the input filenames."
        ),
    )

    opt = p.add_argument_group("optimization")
    opt.add_argument(
        "--steps", type=int, help="Number of optimization steps",
        default=argparse.SUPPRESS)
    opt.add_argument(
        "--style-w", type=float, help="Style weight",
        default=argparse.SUPPRESS)
    opt.add_argument(
        "--content-w", type=float, help="Content weight",
        default=argparse.SUPPRESS)
    opt.add_argument(
        "--lr", type=float, help="Learning rate",
        default=argparse.SUPPRESS)
    opt.add_argument(
        "--init-method", choices=["random", "white", "content"],
        help="Initialization method", default=argparse.SUPPRESS)
    opt.add_argument(
        "--seed", type=int, help="Random seed",
        default=argparse.SUPPRESS)
    opt.add_argument(
        "--no-normalize", action="store_true",
        help="Disable VGG19 normalization")
    opt.add_argument(
        "--style-layers", type=str,
        help="Comma-separated VGG19 layer indices for style loss")
    opt.add_argument(
        "--content-layers", type=str,
        help="Comma-separated VGG19 layer indices for content loss")

    video = p.add_argument_group("video")
    video.add_argument(
        "--save-every", type=int,
        help="Save image every N steps",
        default=argparse.SUPPRESS)
    video.add_argument(
        "--fps", type=int, help="Frames per second for video",
        default=argparse.SUPPRESS)
    video.add_argument(
        "--quality", type=int,
        help="Video quality (lower is better)",
        default=argparse.SUPPRESS)
    video.add_argument(
        "--no-video", action="store_true",
        help="Disable video creation")
    video.add_argument(
        "--final-only", action="store_true",
        help="Only save final image")
    video.add_argument(
        "--no-intro",
        action="store_true",
        help="Disable the intro comparison segment in the video",
    )
    video.add_argument(
        "--intro-duration",
        type=float,
        help=(
            "Seconds to display the intro comparison frame before the "
            "stylization timelapse (default: "
            f"{DEFAULT_VIDEO_INTRO_DURATION})"
        ),
        default=argparse.SUPPRESS,
    )
    video.add_argument(
        "--no-final-frame-compare",
        dest="final_frame_compare",
        action="store_false",
        default=argparse.SUPPRESS,
        help=(
            "Disable the final comparison frame so the timelapse ends on the "
            "last stylization step."
        ),
    )
    video.add_argument(
        "--outro-duration",
        type=float,
        help=(
            "Seconds to display the final comparison frame at the end of the "
            "video (default: "
            f"{DEFAULT_VIDEO_OUTRO_DURATION})"
        ),
        default=argparse.SUPPRESS,
    )
    video.add_argument(
        "--metadata-title",
        type=str,
        help="Custom title to embed in MP4 metadata",
        default=argparse.SUPPRESS,
    )
    video.add_argument(
        "--metadata-artist",
        type=str,
        help="Custom artist/author to embed in MP4 metadata",
        default=argparse.SUPPRESS,
    )

    hw = p.add_argument_group("hardware")
    hw.add_argument(
        "--device", type=str,
        help="Device to run on (e.g., 'cuda' or 'cpu')",
        default=argparse.SUPPRESS)

    cfg = p.add_argument_group("config")
    cfg.add_argument(
        "--config", type=str,
        help="Path to config.toml file")
    cfg.add_argument(
        "--validate-config-only", action="store_true",
        help="Validate config file and exit without running style transfer")

    return p


def log_parameters(
    paths: InputPaths,
    cfg: stv_config.StyleTransferConfig,
    args: argparse.Namespace,
) -> None:
    """Log all user-provided parameters."""
    logger.info("Content image loaded: %s", paths.content_path)
    logger.info("Style image loaded: %s", paths.style_path)
    if getattr(args, "config", None):
        logger.info("Loaded config from: %s", args.config)
    logger.info("Output Directory: %s", cfg.output.output)
    logger.info("Steps: %d", cfg.optimization.steps)
    logger.info("Save Every: %d", cfg.video.save_every)
    logger.info("Style Weight: %g", cfg.optimization.style_w)
    logger.info("Content Weight: %g", cfg.optimization.content_w)
    logger.info("Learning Rate: %g", cfg.optimization.lr)
    logger.info("Style Layers: %s", cfg.optimization.style_layers)
    logger.info("Content Layers: %s", cfg.optimization.content_layers)
    logger.info("FPS for Timelapse Video: %d", cfg.video.fps)
    logger.info("Video Quality: %d (%d-%d scale)", cfg.video.quality,
                VIDEO_QUALITY_MIN, VIDEO_QUALITY_MAX)
    logger.info("Initialization Method: %s", cfg.optimization.init_method)
    logger.info("Normalization: %s",
                "Enabled" if cfg.optimization.normalize else "Disabled")
    logger.info("Video Creation: %s",
                "Enabled" if cfg.video.create_video else "Disabled")
    logger.info("Video Intro: %s",
                "Enabled" if cfg.video.intro_enabled else "Disabled")
    logger.info("Intro Duration (s): %.2f", cfg.video.intro_duration_seconds)
    logger.info("Outro Duration (s): %.2f", cfg.video.outro_duration_seconds)
    logger.info(
        "Final Frame Compare: %s",
        "Enabled" if cfg.video.final_frame_compare else "Disabled",
    )
    logger.info("Loss Plotting: %s",
                "Enabled" if cfg.output.plot_losses else "Disabled")
    logger.info("Random Seed: %d", cfg.optimization.seed)
    logger.info("Metadata Title: %s", cfg.video.metadata_title or "(default)")
    logger.info("Metadata Artist: %s",
                cfg.video.metadata_artist or "(default)")


def parse_int_list(s: str | list[int]) -> list[int]:
    """
    Convert a comma-separated string or list of ints into a list of ints.

    Args:
        s: A string like "0,1,2" or a list of integers.

    Returns:
        A list of integers.

    """
    return stv_config.parse_int_list(s)


def run_from_args(args: argparse.Namespace) -> None:
    """Run style transfer from command-line arguments."""
    base_cfg: stv_config.StyleTransferConfig | None = None
    if args.config:
        base_cfg = stv_config.ConfigLoader.load(args.config)
        if args.validate_config_only:
            logger.info("Config %s validated successfully.", args.config)
            sys.exit(0)

    cfg = stv_config.build_config_from_cli(vars(args), base_config=base_cfg)

    paths = InputPaths(content_path=args.content, style_path=args.style)
    log_parameters(paths, cfg, args)

    stv_main.style_transfer(paths, cfg)

    # Optionally write comparison images.
    if args.compare_inputs or args.compare_result:
        render_requested_comparisons(
            content_path=Path(args.content),
            style_path=Path(args.style),
            output_dir=Path(cfg.output.output),
            request=ComparisonRequest(
                include_inputs=args.compare_inputs,
                include_result=args.compare_result,
            ),
        )


def main() -> None:
    """Run the command-line interface for style transfer execution."""
    arg_parser = build_arg_parser()
    args = arg_parser.parse_args()
    if not args.validate_config_only and (not args.content or not args.style):
        arg_parser.error("the following arguments are required: --content,"
                         " --style")

    run_from_args(args)


if __name__ == "__main__":  # pragma: no cover
    main()
