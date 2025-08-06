"""
CSV logging utility for recording loss values during optimization.

Defines the LossCSVLogger class, which writes style, content, and total
losses to disk at regular intervals.
"""


import csv
from pathlib import Path
from types import TracebackType


class LossCSVLogger:
    """
    Handles CSV logging of style transfer loss metrics.

    This class opens a CSV file, writes the header row, and appends
    loss values (step, style loss, content loss, total loss) every
    N steps. It ensures the file is closed cleanly when logging is
    finished.

    Supports use as a context manager for deterministic cleanup.

    Attributes:
        path: Path to the CSV file.
        log_every: Step interval for logging.
        file: File handle for the open CSV file.
        writer: CSV writer object used for writing rows.

    """

    def __init__(self, path: str | Path, log_every: int) -> None:
        """
        Initialize the CSV logger.

        Args:
            path: Path to the CSV file.
            log_every: Log losses every N steps.

        Raises:
            OSError: If the file cannot be created.

        """
        self.path = Path(path)
        self.log_every = log_every

        # Create parent directory if necessary
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Open CSV file and write header
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            ["step", "style_loss", "content_loss", "total_loss"],
        )
        self.file.flush()  # Flush header immediately

    def log(
        self,
        step: int,
        style_loss: float,
        content_loss: float,
        total_loss: float,
    ) -> None:
        """
        Write a row of loss metrics to a CSV file at step intervals.

        Appends the style, content, and total loss values for the
        current step if the step aligns with the log_every setting.

        Args:
            step: Current optimization step.
            style_loss: Style loss at this step.
            content_loss: Content loss at this step.
            total_loss: Combined total loss at this step.

        """
        if self.writer and step % self.log_every == 0:
            self.writer.writerow(
                [step, style_loss, content_loss, total_loss],
            )
            self.file.flush()  # Ensure row is flushed to disk

    def close(self) -> None:
        """
        Close the CSV file if it was opened.

        This should be called at the end of the optimization loop
        to ensure all data is flushed to disk.
        """
        if self.file and not self.file.closed:
            self.file.close()

    def __enter__(self) -> "LossCSVLogger":
        """
        Enter the runtime context related to this object.

        Returns:
            LossCSVLogger: The logger instance itself.

        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        """
        Exit the runtime context and close the CSV file.

        Ensures the file handle is closed even if an exception occurs.

        Args:
            exc_type: Exception type (unused).
            exc_value: Exception value (unused).
            traceback: Traceback object (unused).

        Returns:
            None: Allows any exception to propagate.

        """
        self.close()
        return None
