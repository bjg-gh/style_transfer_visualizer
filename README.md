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
- Optional normalization and initialization methods
- Save intermediate steps or final image only
- Deterministic execution with `--seed`
- Unit and integration tests with full coverage
- Optional CSV loss logging for long runs

---

## Installation
Set up your environment with uv (https://docs.astral.sh/uv/) and Python 3.12:

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

### Common Options

- `--steps`, `--save-every`, `--style-w`, `--content-w`, `--lr`
- `--init-method {random,white,content}`
- `--fps`, `--quality`
- `--no-normalize`, `--no-video`, `--final-only`
- `--device cpu|cuda`
- `--seed`
- `--config config.toml`
- `--log-loss path/to/losses.csv` (log loss metrics to CSV)
- `--log-every N` (log every N steps, default: 10)

---

## Loss Logging Options

The Style Transfer Visualizer supports two methods for tracking loss metrics during optimization:

### In-Memory Loss Tracking (Default)

By default, all loss metrics (style, content, and total loss) are stored in memory. At the end of the run, a loss plot is generated using matplotlib and saved to the output directory:

```
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

```
style_transfer_visualizer/
├── pyproject.toml                 # Project configuration
├── config.toml                    # Example config
├── src/
│   └── style_transfer_visualizer/
│       ├── cli.py                # CLI entry point
│       ├── config.py
│       ├── config_defaults.py
│       ├── constants.py
│       ├── core_model.py
│       ├── image_io.py
│       ├── logging_utils.py
│       ├── loss_logger.py         # Handles CSV loss logging
│       ├── main.py
│       ├── optimization.py
│       ├── types.py
│       ├── utils.py
│       └── video.py
├── tests/
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_core_model.py
│   ├── test_image_io.py
│   ├── test_main.py
│   ├── test_optimization.py
│   ├── test_utils.py
│   ├── test_video.py
│   └── conftest.py
```

---

## License

MIT License
