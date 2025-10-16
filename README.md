# Style Transfer Visualizer

![Python CI](https://github.com/bjg-gh/style_transfer_visualizer/actions/workflows/python-ci.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![codecov](https://codecov.io/gh/bjg-gh/style_transfer_visualizer/branch/ci-test/graph/badge.svg)](https://codecov.io/gh/bjg-gh/style_transfer_visualizer)
![GitHub Release](https://img.shields.io/github/v/release/bjg-gh/style_transfer_visualizer?sort=semver)
[![Release Notes](https://img.shields.io/badge/Release_Notes-📄%20View-blue)](./RELEASES.md)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-005f73?logo=python&logoColor=white)](https://docs.astral.sh/ruff/)
[![Mypy](https://img.shields.io/badge/types-checked-blue?logo=python&logoColor=white)](http://mypy-lang.org/)

**Author**: [@bjg-gh](https://github.com/bjg-gh)

A command-line tool that applies neural style transfer to images using PyTorch. It supports timelapse video generation, configuration via CLI or TOML, and flexible output control.

---

## Features

- Neural style transfer using VGG19
- Configurable via CLI or TOML
- Timelapse video generation with adjustable FPS and quality
- Intro and outro comparison segments for videos with configurable timing
- Optional normalization and initialization methods
- Save intermediate steps or final image only
- Deterministic execution with `--seed`
- Unit and integration tests with full coverage
- Optional CSV loss logging for long runs

---

## Installation

Set up your environment with uv (<https://docs.astral.sh/uv/>) and Python 3.12:

```bash
uv venv --python=3.12
uv sync --extra cu128 --extra dev
```

---

## Development Setup

Run tests:

```bash
uv run pytest
```

Open `htmlcov/index.html` in your browser for a visual coverage report.

Run pre-commit hooks:

```bash
uv run pre-commit run --all-files
```

---

## Usage

Run the tool from the command line:

```bash
uv run style-visualizer --content path/to/content.jpg --style path/to/style.jpg
```

Or with a config file:

```bash
uv run style-visualizer --config config.toml
```

### Common Options

- `--steps`, `--save-every`, `--style-w`, `--content-w`, `--lr`
- `--init-method {random,white,content}`
- `--fps`, `--quality`
- `--no-normalize`, `--no-video`, `--final-only`
- `--intro-duration N` (seconds to show the intro comparison frame, default: 10)
- `--outro-duration N` (seconds to hold the outro comparison frame, default: 10)
- `--no-intro` (skip the intro comparison segment)
- `--no-final-frame-compare` (skip the outro comparison segment)
- `--device cpu|cuda`
- `--seed`
- `--config config.toml`
- `--log-loss path/to/losses.csv` (log loss metrics to CSV)
- `--log-every N` (log every N steps, default: 10)
- `--metadata-title "Custom Title"` (override MP4 Title metadata)
- `--metadata-artist "Custom Artist"` (override MP4 Artist metadata)

---

## Video Intro & Outro Segments

Every timelapse now starts and ends with comparison sequences by default.

The intro matches the `--compare-inputs` gallery layout and:

- Fades in from black (~1 second), holds for `--intro-duration` seconds
  (default `10`), then crossfades into the first stylization frame.
- Adapts to your output FPS so the transition remains smooth regardless of
  frame rate.
- Can be disabled entirely with `--no-intro` if you prefer to jump straight to
  the optimization timelapse.

After the final optimization frame the timelapse now holds on the stylized
image for roughly one second before crossfading into a gallery-style comparison
of the content, style, and result images:

- Holds the stylized frame on screen momentarily before dissolving into the outro grid.
- Uses the same stacked layout as `--compare-result` with labels and frames.
- Holds for `--outro-duration` seconds (default `10`) before the video ends.
- Disable it with `--no-final-frame-compare` to retain the original ending.

These options map to `video.intro_enabled`, `video.intro_duration_seconds`,
`video.final_frame_compare`, and `video.outro_duration_seconds` in TOML configs.

---

## Loss Logging Options

The Style Transfer Visualizer supports two methods for tracking loss metrics during optimization:

### In-Memory Loss Tracking (Default)

By default, all loss metrics (style, content, and total loss) are stored in memory. At the end of the run, a loss plot is generated using matplotlib and saved to the output directory:

```text
loss_plot.png
```

---

### CSV Loss Logging (Optional)

For long runs, storing all losses in memory may be inefficient. You can log loss metrics directly to a CSV file using the `--log-loss` flag:

```bash
style-visualizer   --content input.jpg   --style style.jpg   --log-loss losses.csv   --log-every 50
```

- `--log-loss`: Path to the CSV file for logging losses.
- `--log-every`: Interval for logging losses (default: 10 steps).

When CSV logging is enabled:

- Loss metrics are written to disk.
- Loss plots are **automatically disabled**.
- Memory usage is reduced for long runs.

---

### CSV Format

The CSV file contains these columns:

| Column         | Description                                         |
| -------------- | --------------------------------------------------- |
| `step`         | The optimization step number                        |
| `style_loss`   | Style loss value at this step                       |
| `content_loss` | Content loss value at this step                     |
| `total_loss`   | Total loss (weighted sum of style and content loss) |

Example:

```csv
step,style_loss,content_loss,total_loss
10,1234.56,78.90,1313.46
20,1150.34,70.12,1220.46
30,1080.12,65.78,1145.90
```

---

## Project Structure

```text
style_transfer_visualizer/
    pyproject.toml                 # Project configuration
    config.toml                    # Example config
    uv.lock                        # Reproducible dependency lockfile
    src/
        style_transfer_visualizer/
            __init__.py
            cli.py                # CLI entry point
            config.py
            config_defaults.py
            constants.py
            core_model.py
            image_grid/
                __init__.py       # Compatibility re-exports for legacy imports
                core.py           # Frame + rendering primitives
                layouts.py        # Grid and gallery composition logic
                naming.py         # Path building and file saving helpers
            image_io.py
            logging_utils.py
            loss_logger.py        # Handles CSV loss logging
            main.py
            optimization.py       # OptimizationRunner orchestrating the training loop
            runtime/
                __init__.py
                device.py         # Device selection helpers
                output.py         # Final save/serialization logic
                validation.py     # Input validation utilities
                version.py        # Version resolution helpers
            type_defs.py
            utils.py
            video.py
            visualization/
                __init__.py
                metrics.py        # Loss plotting helpers
    tests/
        __init__.py
        conftest.py
        test_cli.py
        test_compare_grid.py
        test_config.py
        test_core_model.py
        test_image_grid.py
        test_image_grid_modules.py
        test_image_io.py
        test_logging_utils.py
        test_loss_logger.py
        test_main.py
        test_optimization.py
        test_utils.py
        test_video.py
        runtime/
            test_device.py
            test_output.py
            test_validation.py
            test_version.py
        visualization/
            test_metrics.py
    tools/
        compare_grid.py            # Standalone helper to build grid or gallery comparison images
```

### Image Grid Modules

The image comparison tooling now lives in a focused package:

- `style_transfer_visualizer.image_grid.core` contains reusable rendering primitives (`FrameParams`, panel framing, wall textures).
- `style_transfer_visualizer.image_grid.layouts` composes horizontal grids and gallery wall arrangements.
- `style_transfer_visualizer.image_grid.naming` handles filename generation and convenience save helpers.

Legacy imports that pointed to `style_transfer_visualizer.image_grid` continue to work via compatibility re-exports, but new code should prefer the dedicated submodules.

---

## License

MIT License

## Comparison Flags

The command line interface now supports saving comparison images after a run.

### Flags

- `--compare-inputs` saves a two panel gallery wall that compares Content and Style.
- `--compare-result` saves a three panel gallery wall that compares Content, Style, and the stylized result. If the expected result image is not found, a warning is logged and the comparison is skipped.

## Examples on Windows

```bat
uv run style-visualizer --content C:\img\cat.jpg --style C:\img\wave.jpg --steps 400 --compare-inputs
```

```bat
uv run style-visualizer --content C:\img\cat.jpg --style C:\img\wave.jpg --steps 800 --compare-result
```

Notes

- Output is saved as a PNG inside the directory provided by `--output`.
- File naming uses canonical stems derived from the content and style file names.
- For complete control over layout, wall color, and sizing, see the Tools section below.

## Tools

The `tools` directory contains utilities that work with the project image grid and gallery layouts without running the training pipeline.

### tools/compare_grid.py

A standalone runner for building comparison images.

### Modes

- Grid mode when no layout is provided. This requires a result image and produces a tight three image grid.
- Gallery mode when a layout is provided. This produces a gallery wall with either two or three framed panels.

### Arguments

- `--content PATH` path to the content image
- `--style PATH` path to the style image
- `--result PATH` path to an existing stylized result image. Required in grid mode. Ignored when layout is gallery two across
- `--out PATH` output path. If omitted, a default name is derived from inputs. The suffix is normalized to `.png`
- `--target-height N` grid mode scale height
- `--target-size WxH` gallery mode canvas size, for example `1920x1080`
- `--pad N` pixel padding for grid mode
- `--border-px N` optional border width for grid mode
- `--layout NAME` gallery layout. One of `gallery-two-across`, `gallery-stacked-left`
- `--wall #rrggbb` gallery wall color as a hex string, default `#3c434a`
- `--frame-style NAME` gallery frame tone preset. One of `gold`, `oak`, `black`
- `--show-labels` draw Content, Style, and Final labels on gallery frames

### Grid Examples on Windows

Grid mode with three images

```bat
uv run python tools\compare_grid.py ^
  --content C:\img\content.jpg ^
  --style C:\img\style.jpg ^
  --result C:
uns\stylized_content_x_style.png ^
  --target-height 384 ^
  --pad 8 ^
  --border-px 2 ^
  --out C:
uns\cmp_grid.png
```

Gallery mode two across for Content and Style

```bat
uv run python tools\compare_grid.py ^
  --content C:\img\content.jpg ^
  --style C:\img\style.jpg ^
  --layout gallery-two-across ^
  --wall #112233 ^
  --target-size 1920x1080 ^
  --out C:
uns\gallery_inputs.png
```

Gallery mode stacked left for Content, Style, and Result

```bat
uv run python tools\compare_grid.py ^
  --content C:\img\content.jpg ^
  --style C:\img\style.jpg ^
  --result C:
uns\stylized_content_x_style.png ^
  --layout gallery-stacked-left ^
  --frame-style black ^
  --show-labels ^
  --out C:
uns\gallery_result.png
```

### Behavior

- If `--out` is omitted, a default name is generated from the content and style file stems.
- If `--out` does not end with `.png`, it is rewritten to `.png`.
- In gallery mode with layout gallery two across, any `--result` value is ignored.

## Additional Tools

```csv
tools    compare_grid.py    Standalone helper to build grid or gallery comparison images
```
