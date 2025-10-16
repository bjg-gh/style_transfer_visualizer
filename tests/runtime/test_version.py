"\"\"\"Tests for runtime.version helpers.\"\"\""

from __future__ import annotations

from pathlib import Path

import pytest

from style_transfer_visualizer.runtime import version as runtime_version


def test_resolve_version_from_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime_version.importlib_metadata,
        "version",
        lambda _name: "9.9.9",
    )
    assert runtime_version.resolve_project_version() == "9.9.9"


def test_resolve_version_from_pyproject(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pkg").mkdir()
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(
        "[project]\nversion = '1.2.3'\n",
        encoding="utf-8",
    )

    def raise_missing(_: str) -> None:
        raise runtime_version.importlib_metadata.PackageNotFoundError

    monkeypatch.setattr(
        runtime_version.importlib_metadata,
        "version",
        raise_missing,
    )
    monkeypatch.setattr(runtime_version, "__file__", str(tmp_path / "pkg" / "version.py"))

    assert runtime_version.resolve_project_version() == "1.2.3"


def test_resolve_version_fallback_to_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    (tmp_path / "pkg").mkdir()
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text("[project]\n", encoding="utf-8")

    def raise_missing(_: str) -> None:
        raise runtime_version.importlib_metadata.PackageNotFoundError

    monkeypatch.setattr(
        runtime_version.importlib_metadata,
        "version",
        raise_missing,
    )

    def raise_os_error(handle: object) -> None:
        raise OSError("boom")

    monkeypatch.setattr(runtime_version.tomllib, "load", raise_os_error)
    monkeypatch.setattr(runtime_version, "__file__", str(tmp_path / "pkg" / "version.py"))

    with caplog.at_level("WARNING"):
        assert runtime_version.resolve_project_version() == "0.0.0"

    assert any("Error reading" in rec.message for rec in caplog.records)


def test_resolve_version_missing_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nested = tmp_path / "pkg" / "inner"
    nested.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")

    def raise_missing(_: str) -> None:
        raise runtime_version.importlib_metadata.PackageNotFoundError

    monkeypatch.setattr(
        runtime_version.importlib_metadata,
        "version",
        raise_missing,
    )
    monkeypatch.setattr(runtime_version, "__file__", str(nested / "version.py"))

    assert runtime_version.resolve_project_version() == "0.0.0"
