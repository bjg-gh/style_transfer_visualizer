"""CLI argument parsing and main entry point."""

import argparse
import sys
from pathlib import Path

from style_transfer_visualizer.logging_utils import logger
import style_transfer_visualizer.config as stv_config
import style_transfer_visualizer.main as stv_main


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the command-line interface."""

    p = argparse.ArgumentParser(
        description="Neural Style Transfer with PyTorch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""Examples:
python {Path(__file__).name} --content cat.jpg --style starry_night.jpg
python {Path(__file__).name} --content cat.jpg --style starry_night.jpg --final-only
python {Path(__file__).name} --content cat.jpg --style starry_night.jpg --steps 1000 --fps 30

Note:
  Normalization is enabled by default. Use --no-normalize to disable it"""
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


def log_parameters(p: dict, args: argparse.Namespace) -> None:
    """Logs all user-provided command-line parameters."""
    logger.info("Content image loaded: %s", p["content_path"])
    logger.info("Style image loaded: %s", p["style_path"])
    if hasattr(args, "config"):
        logger.info("Loaded config from: %s", args.config)
    logger.info("Output Directory: %s", p["output_dir"])
    logger.info("Steps: %d", p["steps"])
    logger.info("Save Every: %d", p["save_every"])
    logger.info("Style Weight: %g", p["style_weight"])
    logger.info("Content Weight: %g", p["content_weight"])
    logger.info("Learning Rate: %g", p["learning_rate"])
    logger.info("Style Layers: %s", p["style_layers"])
    logger.info("Content Layers: %s", p["content_layers"])
    logger.info("FPS for Timelapse Video: %d", p["fps"])
    logger.info("Video Quality: %d (1–10 scale)", p["video_quality"])
    logger.info("Initialization Method: %s", p["init_method"])
    logger.info("Normalization: %s",
                "Enabled" if p["normalize"] else "Disabled")
    logger.info("Video Creation: %s",
                "Enabled" if p["create_video"] else "Disabled")
    logger.info("Loss Plotting: %s",
                "Enabled" if p["plot_losses"] else "Disabled")
    logger.info("Random Seed: %d", p["seed"])


def parse_int_list(s: str | list[int]) -> list[int]:
    """Convert a comma-separated string or list of ints into a list of ints.

    Args:
        s: A string like "0,1,2" or a list of integers.

    Returns:
        A list of integers.
    """
    if isinstance(s, list):
        return s
    return list(map(int, s.split(",")))


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

    params = {
        "content_path": args.content,
        "style_path": args.style,
        "output_dir": get("output", "output"),
        "steps": get("steps", "optimization"),
        "save_every": get("save_every", "video"),
        "style_weight": get("style_w", "optimization"),
        "content_weight": get("content_w", "optimization"),
        "learning_rate": get("lr", "optimization"),
        "style_layers": parse_int_list(get("style_layers", "optimization")),
        "content_layers": parse_int_list(get("content_layers",
                                             "optimization")),
        "fps": get("fps", "video"),
        "device_name": get("device", "hardware"),
        "init_method": get("init_method", "optimization"),
        "normalize": not getattr(args, "no_normalize", False),
        "create_video": not getattr(args, "no_video", False),
        "final_only": getattr(args, "final_only", False),
        "video_quality": get("quality", "video"),
        "seed": get("seed", "optimization"),
        "plot_losses": not getattr(args, "no_plot", False)
    }

    log_parameters(params, args)

    return stv_main.style_transfer(**params)


def main() -> None:
    """Main entry point for the CLI."""
    arg_parser = build_arg_parser()
    args = arg_parser.parse_args()
    if not args.validate_config_only and (not args.content or not args.style):
        arg_parser.error("the following arguments are required: --content,"
                         " --style")

    run_from_args(args)