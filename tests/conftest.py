"""Pytest configuration and fixtures."""

import pytest
import sys
from pathlib import Path

# Add the code directory to path
code_dir = Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))


@pytest.fixture(autouse=True)
def reset_storage():
    """Reset storage before each test.

    Soft-import so the data-API test module — which doesn't need the
    alpha-mining LangGraph stack — runs even when langchain isn't installed.
    """
    try:
        from alpha_mining.tools.storage_tools import reset_storage as _reset
    except ImportError:
        yield
        return
    _reset()
    yield
    _reset()
