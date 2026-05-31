"""Pytest configuration and fixtures."""

import pytest
import sys
from pathlib import Path

# Add the code directory to path
code_dir = Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))


@pytest.fixture(autouse=True)
def reset_storage():
    """Reset storage before each test."""
    from alpha_mining.tools.storage_tools import reset_storage
    reset_storage()
    yield
    reset_storage()
