"""
This test dynamically loads the run_visualizer.py script using importlib.
"""
import importlib
import subprocess
import sys
from pathlib import Path

import pytest
from unittest import mock
from PIL import Image

# Why not a regular import for run_version?
#
#    We avoid using `import run_version` because run_version.py is a
#    top-level script, not part of the src/ package or installed module.
#    Pytest and static tools like PyCharm can't resolve it as a module,
#    which results in ModuleNotFoundError during collection.
#
#    Using importlib allows us to load and test the script without
#    requiring packaging or modifying PYTHONPATH. This also ensures
#    compatibility across CLI, IDE, and CI environments.

@pytest.mark.integration
def test_script_main_entry(tmp_path: Path):
    """Integration test: execute script via subprocess with real images."""
    script: Path = Path("run_visualizer.py").resolve()
    content: Path = tmp_path / "content.jpg"
    style: Path = tmp_path / "style.jpg"
    content.write_bytes(b"fake")
    style.write_bytes(b"fake")

    Image.new("RGB", (64, 64), color="blue").save(content)
    Image.new("RGB", (64, 64), color="green").save(style)

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--content", str(content),
            "--style", str(style),
            "--final-only",
            "--device", "cpu",
            "--steps", "2",
            "--save-every", "3",
            "--init-method", "white"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, (
        f"Script failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
    assert "Style transfer completed" in result.stdout or result.stderr


def test_run_visualizer_not_main():
    """Ensure run_visualizer.py does nothing when not executed as __main__."""
    # Force reload of the module to reset any state
    if "run_visualizer" in sys.modules:
        del sys.modules["run_visualizer"]

    with mock.patch("style_transfer_visualizer.cli.main") as mock_main:
        with mock.patch.object(sys, 'path', sys.path.copy()):
            module = importlib.import_module("run_visualizer")
            mock_main.assert_not_called()
            assert module.__name__ != "__main__"