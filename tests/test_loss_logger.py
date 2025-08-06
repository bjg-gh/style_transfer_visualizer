"""Unit tests for LossCSVLogger."""

import csv
from pathlib import Path
from typing import TextIO
from unittest import mock

import pytest
from pytest_mock import MockerFixture

from style_transfer_visualizer.loss_logger import LossCSVLogger


def test_csv_file_creation(tmp_path: Path) -> None:
    """Test that the CSV file is created and header row is written."""
    csv_path = tmp_path / "losses.csv"
    logger = LossCSVLogger(csv_path, log_every=1)
    logger.close()

    # Check file exists
    assert csv_path.exists()

    # Check header row
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["step", "style_loss", "content_loss", "total_loss"]


def test_log_writes_rows(tmp_path: Path) -> None:
    """Test that log writes rows at correct intervals."""
    csv_path = tmp_path / "losses.csv"
    logger = LossCSVLogger(csv_path, log_every=2)

    # Write steps 1 to 4
    for step in range(1, 5):
        logger.log(step, style_loss=1.0, content_loss=0.5, total_loss=1.5)

    logger.close()

    # Read CSV and verify rows
    with csv_path.open("r", newline="") as f:
        reader = list(csv.reader(f))
        assert reader[0] == ["step", "style_loss", "content_loss",
                             "total_loss"]
        # Should have rows for steps 2 and 4 only
        assert reader[1] == ["2", "1.0", "0.5", "1.5"]
        assert reader[2] == ["4", "1.0", "0.5", "1.5"]
        assert len(reader) == 3  # noqa: PLR2004


def test_close_closes_file(tmp_path: Path) -> None:
    """Test that close() closes the file handle."""
    csv_path = tmp_path / "losses.csv"
    logger = LossCSVLogger(csv_path, log_every=1)
    assert not logger.file.closed
    logger.close()
    assert logger.file.closed


def test_context_manager_closes_file(tmp_path: Path) -> None:
    """Test that context manager closes file properly."""
    csv_path = tmp_path / "losses.csv"
    with LossCSVLogger(csv_path, log_every=1) as logger:
        logger.log(1, 1.0, 0.5, 1.5)
    # After context, file should be closed
    assert logger.file.closed


def test_invalid_path_raises(tmp_path: Path, mocker: MockerFixture) -> None:
    """Simulate OSError when creating parent directory."""
    invalid_path = tmp_path / "losses.csv"
    mocker.patch("pathlib.Path.mkdir", side_effect=OSError("Mocked error"))

    with pytest.raises(OSError, match="Mocked error"):
        LossCSVLogger(invalid_path, log_every=1)


def test_flush_called_on_log(tmp_path: Path) -> None:
    """Test that flush() is called after each writerow()."""
    csv_path = tmp_path / "losses.csv"
    with mock.patch("builtins.open", mock.mock_open()):
        logger = LossCSVLogger(csv_path, log_every=1)
        logger.file.flush = mock.MagicMock()
        logger.log(1, 1.0, 0.5, 1.5)
        logger.file.flush.assert_called()


@pytest.mark.parametrize(
    "file_value",
    [
        (None, None),           # Simulate logger.file = None
        ("mock_file", True),    # Simulate logger.file.closed == True
    ],
)
def test_close_noop_when_file_missing_or_closed(
    tmp_path: Path,
    file_value: str | None,
) -> None:
    """Test that close() does nothing if file is None or already closed."""
    logger = LossCSVLogger(tmp_path / "losses.csv", log_every=1)

    if file_value is None:
        object.__setattr__(logger, "file", None)
    else:
        mock_file = mock.MagicMock(spec=TextIO)
        mock_file.closed = True
        object.__setattr__(logger, "file", mock_file)

    logger.close()  # Should not raise
