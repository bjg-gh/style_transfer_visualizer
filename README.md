# Style Transfer Visualizer

A command-line tool that applies neural style transfer to images using PyTorch. It supports timelapse video generation, configuration via CLI or TOML, and flexible output control.

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
