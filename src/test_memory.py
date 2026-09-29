"""Tests for src/memory.py, with its files redirected to a temporary directory."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from memory import save_preference, get_preference, save_environment_setup, get_environment_setup  # noqa: E402


@pytest.fixture
def tmp_dir(tmp_path):
    # Create a temporary directory for test files
    temp_dir = tmp_path / "test_memory"
    temp_dir.mkdir(exist_ok=True)
    return temp_dir

@pytest.fixture
def monkeypatch_env(tmp_dir, monkeypatch):
    # Use monkeypatch to set the environment variables for the test
    monkeypatch.setattr("memory.PREFERENCES_FILE", str(tmp_dir / "preferences.json"))
    monkeypatch.setattr("memory.ENV_SETUP_FILE", str(tmp_dir / "env_setup.json"))


def test_save_and_get_preference(tmp_dir, monkeypatch_env):
    # Test saving and retrieving preferences
    save_preference("theme", "dark")
    save_preference("language", "en")
    save_preference("font_size", 14)

    assert get_preference("theme", "default") == "dark"
    assert get_preference("language", "default") == "en"
    assert get_preference("font_size", 12) == 14
    assert get_preference("nonexistent", "default") == "default"


def test_save_and_get_environment_setup(tmp_dir, monkeypatch_env):
    # Test saving and retrieving environment setup
    env_setup = {
        "python_version": "3.9.7",
        "dependencies": ["requests", "numpy"]
    }
    save_environment_setup(env_setup)

    assert get_environment_setup() == env_setup


def test_get_environment_setup_empty(tmp_dir, monkeypatch_env):
    # Test get_environment_setup returns empty dict when nothing is saved
    assert get_environment_setup() == {}