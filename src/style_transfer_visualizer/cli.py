"""CLI argument parsing and main entry point."""

import argparse
import sys
from pathlib import Path

from PIL import Image

import style_transfer_visualizer.config as stv_config
import style_transfer_visualizer.main as stv_main
from style_transfer_visualizer.config_defaults import DEFAULT_LOG_EVERY
from style_transfer_visualizer.constants import (
    COLOR_GREY,
    VIDEO_QUALITY_MAX,
    VIDEO_QUALITY_MIN,
)
from style_transfer_visualizer.image_grid import (
    default_comparison_name,
    save_gallery_comparison,
)
from style_transfer_visualizer.logging_utils import logger
from style_transfer_visualizer.type_defs import InputPaths, LayoutName
from style_transfer_visualizer.utils import stylized_image_path_from_paths


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
    output.add_argument("--no-plot", action="store_true",
                        help = "Disable loss plotting")
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
            "the output directory and exit. Requires --result to point to the "
            "stylized image file."
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
    if isinstance(s, list):
        return s
    return list(map(int, s.split(",")))


def _apply_output_overrides(
    cfg: stv_config.StyleTransferConfig,
    args: argparse.Namespace,
) -> None:
    """Apply CLI overrides for the [output] section."""
    if hasattr(args, "output"):
        cfg.output.output = args.output
    if hasattr(args, "log_every"):
        cfg.output.log_every = args.log_every
    if hasattr(args, "log_loss"):
        cfg.output.log_loss = args.log_loss  # type: ignore[attr-defined]
    if getattr(args, "no_plot", False):
        cfg.output.plot_losses = False


def _apply_optimization_overrides(
    cfg: stv_config.StyleTransferConfig,
    args: argparse.Namespace,
) -> None:
    """Apply CLI overrides for the [optimization] section."""
    if hasattr(args, "steps"):
        cfg.optimization.steps = args.steps
    if hasattr(args, "style_w"):
        cfg.optimization.style_w = args.style_w
    if hasattr(args, "content_w"):
        cfg.optimization.content_w = args.content_w
    if hasattr(args, "lr"):
        cfg.optimization.lr = args.lr
    if hasattr(args, "init_method"):
        cfg.optimization.init_method = args.init_method
    if hasattr(args, "seed"):
        cfg.optimization.seed = args.seed
    if getattr(args, "no_normalize", False):
        cfg.optimization.normalize = False
    if getattr(args, "style_layers", None):
        cfg.optimization.style_layers = parse_int_list(args.style_layers)
    if getattr(args, "content_layers", None):
        cfg.optimization.content_layers = parse_int_list(args.content_layers)


def _apply_video_overrides(
    cfg: stv_config.StyleTransferConfig,
    args: argparse.Namespace,
) -> None:
    """Apply CLI overrides for the [video] section."""
    if hasattr(args, "save_every"):
        cfg.video.save_every = args.save_every
    if hasattr(args, "fps"):
        cfg.video.fps = args.fps
    if hasattr(args, "quality"):
        cfg.video.quality = args.quality
    if getattr(args, "no_video", False):
        cfg.video.create_video = False
    if getattr(args, "final_only", False):
        cfg.video.final_only = True
    if hasattr(args, "metadata_title"):
        cfg.video.metadata_title = args.metadata_title
    if hasattr(args, "metadata_artist"):
        cfg.video.metadata_artist = args.metadata_artist


def _apply_hardware_overrides(
    cfg: stv_config.StyleTransferConfig,
    args: argparse.Namespace,
) -> None:
    """Apply CLI overrides for the [hardware] section."""
    if hasattr(args, "device"):
        cfg.hardware.device = args.device


def _comparison_out_path(
    out_dir: str,
    content_p: Path,
    style_p: Path,
    *,
    include_result: bool,
) -> Path:
    """
    Build the deterministic comparison image path under out_dir.

    The inputs-only variant uses the standard comparison name.
    The result variant appends '_final' before the suffix.
    """
    base = default_comparison_name(content_p, style_p, Path(out_dir))
    if include_result:
        return base.parent / f"{base.stem}_final{base.suffix}"
    return base


def _save_comparison_image(
    content_path: str,
    style_path: str,
    out_dir: str,
    *,
    include_result: bool,
    result_path: str | None,
) -> Path:
    """
    Render and save the gallery-wall comparison image under out_dir.

    Labels are always enabled, frame tone is gold, wall uses default
    project color, and target canvas equals the content image size.
    """
    content_p = Path(content_path)
    style_p = Path(style_path)
    result_p = Path(result_path) if include_result and result_path else None

    # Canvas equals content size.
    with Image.open(content_p) as im:
        target_size = im.size

    layout: LayoutName = (
        "gallery-stacked-left" if include_result else "gallery-two-across"
    )
    out_path = _comparison_out_path(
        out_dir, content_p, style_p, include_result=include_result,
    )

    out_path = save_gallery_comparison(
        content_path=content_p,
        style_path=style_p,
        result_path=result_p,
        out_path=out_path,
        target_size=target_size,
        layout=layout,
        wall_color=COLOR_GREY,
        frame_tone="gold",
        show_labels=True,
    )
    logger.info("Saved comparison image: %s", out_path)
    return out_path


def _enforce_csv_plot_rule(cfg: stv_config.StyleTransferConfig) -> None:
    """Disable plotting when CSV logging is enabled, with a warning."""
    if getattr(cfg.output, "log_loss", None) and cfg.output.plot_losses:
        logger.warning(
            "Loss plotting is disabled because CSV logging is enabled. "
            "Only loss CSV will be created.",
        )
        cfg.output.plot_losses = False


def run_from_args(args: argparse.Namespace) -> None:
    """Run style transfer from command-line arguments."""
    cfg = stv_config.StyleTransferConfig.model_validate({})  # defaults
    if args.config:
        cfg = stv_config.ConfigLoader.load(args.config)
        if args.validate_config_only:
            logger.info("Config %s validated successfully.", args.config)
            sys.exit(0)

    # Apply CLI overrides by section
    _apply_output_overrides(cfg, args)
    _apply_optimization_overrides(cfg, args)
    _apply_video_overrides(cfg, args)
    _apply_hardware_overrides(cfg, args)

    # CSV disables plot
    _enforce_csv_plot_rule(cfg)

    paths = InputPaths(content_path=args.content, style_path=args.style)
    log_parameters(paths, cfg, args)

    stv_main.style_transfer(paths, cfg)

    # Optionally write comparison images.
    if args.compare_inputs or args.compare_result:
        out_dir = cfg.output.output
        content_p = Path(args.content)
        style_p = Path(args.style)

        # inputs-only comparison
        if args.compare_inputs:
            _save_comparison_image(
                content_path=str(content_p),
                style_path=str(style_p),
                out_dir=out_dir,
                include_result=False,
                result_path=None,
            )

        # content+style+result comparison
        if args.compare_result:
            result_p = stylized_image_path_from_paths(
                Path(out_dir),
                content_p,
                style_p,
            )
            if not result_p.exists():
                logger.warning(
                    "Expected stylized result missing: %s. "
                    "Skipping content+style+result comparison.",
                    result_p,
                )
            else:
                _save_comparison_image(
                    content_path=str(content_p),
                    style_path=str(style_p),
                    out_dir=out_dir,
                    include_result=True,
                    result_path=str(result_p),
                )


def main() -> None:
    """Run the command-line interface for style transfer execution."""
    arg_parser = build_arg_parser()
    args = arg_parser.parse_args()
    if not args.validate_config_only and (not args.content or not args.style):
        arg_parser.error("the following arguments are required: --content,"
                         " --style")

    run_from_args(args)


if __name__ == "__main__":
    main()
