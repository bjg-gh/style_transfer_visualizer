"""
run_visualizer.py — CLI Entry Point

This script serves as the command-line interface entry point for the
Style Transfer Visualizer project. It forwards execution to the
modularized CLI logic defined in `src/style_transfer_visualizer/cli.py`.

Usage:
    python run_visualizer.py --content path/to/content.jpg --style path/to/style.jpg [options]

This wrapper allows you to run the tool directly without needing to
modify PYTHONPATH or install the project as a package.

For help on available options, run:
    python run_visualizer.py --help
"""
import sys
# Source code in src/ subdirectory
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import style_transfer_visualizer.cli as stv_cli

if __name__ == "__main__":
    stv_cli.main()
