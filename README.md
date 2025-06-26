# Style Transfer Visualizer: Adam Exploration

A command-line tool that applies neural style transfer to images using PyTorch. It supports timelapse video generation,
configuration via CLI or TOML, and flexible output control.

## Exploration and Abandonment of Adam Optimizer
After a full implementation, benchmarking, and visual inspection, I've decided to remove Adam optimizer support and
freeze the work in this branch.

Although Adam is popular for many deep learning tasks, the evaluation showed:

* LBFGS produces consistently better results for neural style transfer in fewer steps
* Adam required significantly more tuning (e.g., higher style weights, longer runs) and still underperformed
* The added complexity in CLI, config, and optimization logic was not justified by quality or speed gains

Since LBFGS works better out of the box for this specific task, I’ve decided to drop Adam and keep the pipeline
streamlined.

Thus, unit tests have not been updated to reflect the Adam code changes.

If future use cases emerge (e.g., batch transfer, multi-resolution training), this decision can be revisited.

## Features

- Neural style transfer using VGG19
- Configurable via CLI or TOML
- Timelapse video generation with adjustable FPS and quality
- Optional normalization and initialization methods
- Save intermediate steps or final image only
- Deterministic execution with `--seed`
- Unit and integration tests with full coverage

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For CUDA support (recommended), use:

```bash
pip install -r requirements-cuda.txt
```

---

## Usage

Run the tool from the command line:

```bash
python run_visualizer.py --content path/to/content.jpg --style path/to/style.jpg
```

### Common Options

- `--steps`, `--save-every`, `--style-w`, `--content-w`, `--lr`
- `--init-method {random,white,content}`
- `--optimizer`
- `--fps`, `--quality`
- `--no-normalize`, `--no-video`, `--final-only`
- `--device cpu|cuda`
- `--seed`
- `--config config.toml`

---

## Project Structure

```
style_transfer_visualizer/
├── run_visualizer.py              # Entry point
├── __version__.py                 # Version string
├── config.toml                    # Example config
├── src/
│   └── style_transfer_visualizer/
│       ├── cli.py
│       ├── config.py
│       ├── config_defaults.py
│       ├── constants.py
│       ├── core_model.py
│       ├── image_io.py
│       ├── logging_utils.py
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
│   ├── test_run_visualizer.py
│   ├── test_utils.py
│   ├── test_video.py
│   └── conftest.py
```

---

## Testing

Run all tests:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=src/style_transfer_visualizer --cov-report=term --cov-report=html
```

Open `htmlcov/index.html` in your browser for a visual coverage report.

---

## License

MIT License
