# Style Transfer Visualizer
![Python CI](https://github.com/bjg-gh/style_transfer_visualizer/actions/workflows/python-ci.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![codecov](https://codecov.io/gh/bjg-gh/style_transfer_visualizer/branch/ci-test/graph/badge.svg)](https://codecov.io/gh/bjg-gh/style_transfer_visualizer)
[![GitHub release](https://img.shields.io/github/v/release/bjg-gh/style-transfer-visualizer?sort=semver)](https://github.com/bjg-gh/style-transfer-visualizer/releases)

**Author**: [@bjg-gh](https://github.com/bjg-gh)

A powerful and customizable neural style transfer tool implemented in PyTorch. This script allows you to stylize content images using the style of another image and generate a timelapse video and loss plot of the transfer process.

## 🚀 Features

- Content and style image support using VGG19-based feature extraction
- LBFGS optimizer with customizable initialization (random, white, content)
- Frame-by-frame image saving and loss tracking
- Optional timelapse MP4 video output and matplotlib loss plots
- Flexible CLI with extensive configuration options
- Comprehensive test suite with coverage and integration testing

## 🛠 Installation

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## 📸 Usage

```bash
python style_transfer_visualizer.py --content path/to/content.jpg --style path/to/style.jpg
```

### Example:

```bash
python style_transfer_visualizer.py --content cat.jpg --style starry_night.jpg --steps 500 --fps 30 --final-only
```

### Options:

| Option              | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `--content`         | Path to the content image (required)                                       |
| `--style`           | Path to the style image (required)                                         |
| `--output`          | Output directory (default: `out`)                                          |
| `--steps`           | Number of optimization steps (default: 300)                                |
| `--save-every`      | Save frame every N steps (default: 20)                                     |
| `--style-w`         | Style loss weight (default: 1e6)                                           |
| `--content-w`       | Content loss weight (default: 1.0)                                         |
| `--lr`              | Learning rate for optimizer (default: 1.0)                                 |
| `--fps`             | Frames per second for video (default: 10)                                  |
| `--height`          | Target output height (default: 1080)                                       |
| `--init-method`     | Initialization: `content`, `random`, or `white` (default: `random`)        |
| `--no-normalize`    | Disable ImageNet normalization                                             |
| `--no-video`        | Skip video generation                                                      |
| `--final-only`      | Save only the final image (implies `--no-video`)                           |
| `--quality`         | Video quality (1–10, default: 10)                                          |
| `--device`          | Device to run on: `cpu` or `cuda` (default: `cuda`)                        |
| `--seed`            | Random seed for reproducibility (default: 0)                               |

## 📊 Output

- `stylized_<content>_x_<style>.png`: Final stylized image
- `timelapse_<content>_x_<style>.mp4`: Optional video output
- `loss_plot.png`: Optional loss visualization

## ✅ Testing

This project uses `pytest` with coverage and visual/integration markers.

Run all tests with coverage:

```bash
pytest
```

View HTML coverage report:

```bash
pytest --cov=. --cov-report=html
```

Run only fast unit tests:

```bash
pytest -m "not slow"
```

## 🧪 Test Suite Highlights

- `test_core_model.py`: VGG feature slicing, Gram matrix, forward loss computation
- `test_cli.py`: Argument parsing, `main()` CLI entry
- `test_image_processing.py`: Loading, resizing, padding, normalization
- `test_utils.py`: Device setup, directory creation, seeding
- `test_video_output.py`: Timelapse writing, final image output

## 📁 Project Structure

```
.
├── style_transfer_visualizer.py    # Main neural style transfer implementation
├── requirements.txt                # Dependencies
├── pytest.ini                      # Pytest configuration
├── .coveragerc                     # Coverage configuration
├── tests/
    ├── test_cli.py
    ├── test_core_model.py
    ├── test_image_processing.py
    ├── test_utils.py
    └── test_video_output.py
```

## 📄 License

MIT License. See `LICENSE` file for details.
