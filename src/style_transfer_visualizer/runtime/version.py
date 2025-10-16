"""Helpers for accessing the installed package version."""

from __future__ import annotations

import tomllib
from importlib import metadata as importlib_metadata
from pathlib import Path

from style_transfer_visualizer.logging_utils import logger


def resolve_project_version() -> str:
    """
    Return the best-guess project version without adding dependencies.

    Checks installed distributions first, then walks up the filesystem looking
    for a pyproject.toml with a project.version entry, and finally
    falls back to "0.0.0" for development contexts.
    """
    for distribution_name in (
        "style-transfer-visualizer",
        "style_transfer_visualizer",
    ):
        try:
            return importlib_metadata.version(distribution_name)
        except importlib_metadata.PackageNotFoundError:
            continue

    for parent in Path(__file__).resolve().parents:
        pyproject_path = parent / "pyproject.toml"
        if pyproject_path.is_file():
            try:
                with pyproject_path.open("rb") as handle:
                    data = tomllib.load(handle)
            except OSError as exc:
                logger.warning("Error reading %s: %s", pyproject_path, exc)
                break

            version = data.get("project", {}).get("version")
            if isinstance(version, str) and version.strip():
                return version.strip()

    return "0.0.0"
