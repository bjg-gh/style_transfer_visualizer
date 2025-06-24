"""CLI argument parsing and main entry point."""

import argparse
import sys

from style_transfer_visualizer.config_defaults import (
    DEFAULT_OUTPUT_DIR, DEFAULT_STEPS, DEFAULT_STYLE_WEIGHT,
    DEFAULT_CONTENT_WEIGHT, DEFAULT_LEARNING_RATE, DEFAULT_INIT_METHOD,
    DEFAULT_SEED, DEFAULT_SAVE_EVERY, DEFAULT_FPS, DEFAULT_VIDEO_QUALITY,
    DEFAULT_FINAL_ONLY, DEFAULT_DEVICE
)
from style_transfer_visualizer.logging_utils import logger
import style_transfer_visualizer.config as stv_config
import style_transfer_visualizer.main as stv_main


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the command-line interface."""

    p = argparse.ArgumentParser(
        description="Neural Style Transfer with PyTorch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""Examples:
python {{Path(__file__).name}} --content cat.jpg --style starry_night.jpg
python {{Path(__file__).name}} --content cat.jpg --style starry_night.jpg --final-only
python {{Path(__file__).name}} --content cat.jpg --style starry_night.jpg --steps 1000 --fps 30

Note:
  Normalization is enabled by default. Use --no-normalize to disable it"""
    )

    required = p.add_argument_group("required arguments")
    required.add_argument("--content", type=str, help="Path to content image")
    required.add_argument("--style", type=str, help="Path to style image")

    output = p.add_argument_group("output")
    output.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory")

    opt = p.add_argument_group("optimization")
    opt.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                     help="Number of optimization steps")
    opt.add_argument("--style-w", type=float, default=DEFAULT_STYLE_WEIGHT,
                     help="Style weight")
    opt.add_argument("--content-w", type=float,
                     default=DEFAULT_CONTENT_WEIGHT, help="Content weight")
    opt.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE,
                     help="Learning rate")
    opt.add_argument("--init-method", choices=["random", "white", "content"],
                     default=DEFAULT_INIT_METHOD,
                     help="Initialization method")
    opt.add_argument("--seed", type=int, default=DEFAULT_SEED,
                     help="Random seed")
    opt.add_argument("--no-normalize", action="store_true",
                     help="Disable VGG19 normalization")

    video = p.add_argument_group("video")
    video.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY,
                       help="Save image every N steps")
    video.add_argument("--fps", type=int, default=DEFAULT_FPS,
                       help="Frames per second for video")
    video.add_argument("--quality", type=int, default=DEFAULT_VIDEO_QUALITY,
                       help="Video quality (lower is better)")
    video.add_argument("--no-video", action="store_true",
                       help="Disable video creation")
    video.add_argument("--final-only", action="store_true",
                       default=DEFAULT_FINAL_ONLY,
                       help="Only save final image")

    hw = p.add_argument_group("hardware")
    hw.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                    help="Device to run on (e.g., 'cuda' or 'cpu')")

    cfg = p.add_argument_group("config")
    cfg.add_argument("--config", type=str, help="Path to config.toml file")
    cfg.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Validate config file and exit without running style transfer"
    )
    return p


def log_parameters(args: argparse.Namespace) -> None:
    """Logs all user-provided command-line parameters."""
    logger.info("Content image loaded: %s", args.content)
    logger.info("Style image loaded: %s", args.style)
    if getattr(args, "config", None):
        logger.info("Loaded config from: %s", args.config)
    logger.info("Output Directory: %s", args.output)
    logger.info("Steps: %d", args.steps)
    logger.info("Save Every: %d", args.save_every)
    logger.info("Style Weight: %g", args.style_w)
    logger.info("Content Weight: %g", args.content_w)
    logger.info("Learning Rate: %g", args.lr)
    logger.info("FPS for Timelapse Video: %d", args.fps)
    logger.info("Video Quality: %d (1–10 scale)", args.quality)
    logger.info("Initialization Method: %s", args.init_method)
    logger.info("Normalization: %s",
                "Enabled" if not args.no_normalize else "Disabled")
    logger.info("Video Creation: %s",
                "Disabled" if args.no_video else "Enabled")
    logger.info("Random Seed: %d", args.seed)


def run_from_args(args: argparse.Namespace):
    """Run style transfer from command-line arguments."""
    config = stv_config.StyleTransferConfig()
    if args.config:
        config = stv_config.ConfigLoader.load(args.config)
        if args.validate_config_only:
            logger.info("Config %s validated successfully.", args.config)
            sys.exit(0)

    def get(attr, section):
        if hasattr(args, attr) and getattr(args, attr) is not None:
            return getattr(args, attr)
        return getattr(getattr(config, section), attr)

    log_parameters(args)

    return stv_main.style_transfer(
        content_path=args.content,
        style_path=args.style,
        output_dir=get("output", "output"),
        steps=get("steps", "optimization"),
        save_every=get("save_every", "video"),
        style_weight=get("style_w", "optimization"),
        content_weight=get("content_w", "optimization"),
        learning_rate=get("lr", "optimization"),
        fps=get("fps", "video"),
        device_name=get("device", "hardware"),
        init_method=get("init_method", "optimization"),
        normalize=(not args.no_normalize
                   if hasattr(args, "no_normalize")
                   else get("normalize", "optimization")),
        create_video=(not args.no_video
                      if hasattr(args, "no_video")
                      else get("create_video", "video")),
        final_only=(args.final_only
                    if hasattr(args, "final_only") and args.final_only
                    else get("final_only", "video")),
        video_quality=get("quality", "video"),
        seed=get("seed", "optimization")
    )


def main() -> None:
    """Main entry point for the CLI."""
    arg_parser = build_arg_parser()
    args = arg_parser.parse_args()
    if not args.validate_config_only and (not args.content or not args.style):
        arg_parser.error("the following arguments are required: --content,"
                         " --style")
    run_from_args(args)
