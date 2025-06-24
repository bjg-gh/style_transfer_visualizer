"""
Unit test for __version__.py.

This test verifies that the __version__ variable is defined and
is a non-empty string. It ensures that version tracking is enforced
as part of the project structure and remains visible to automated
tooling and documentation systems.
"""
from __version__ import __version__

def test_version_exists():
    assert isinstance(__version__, str)
    assert __version__  # Non-empty