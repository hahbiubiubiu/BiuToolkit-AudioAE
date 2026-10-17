"""Shared package configuration."""

from __future__ import annotations

import os
from pathlib import Path


def _default_model_dir() -> Path:
    """Return the package-local model cache directory without creating it."""
    return Path(__file__).resolve().parent / "model_cache"


DEFAULT_MODEL_DIR = str(
    Path(os.getenv("AUDIO_TOOLKIT_MODEL_DIR", _default_model_dir())).expanduser()
)
os.environ.setdefault("AUDIO_TOOLKIT_MODEL_DIR", DEFAULT_MODEL_DIR)
os.environ.setdefault("NEMO_LOG_LEVEL", "ERROR")
os.environ.setdefault("NEMO_CACHE_DIR", str(Path(DEFAULT_MODEL_DIR) / "nemo"))
os.environ.setdefault("HF_HOME", str(Path(DEFAULT_MODEL_DIR) / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(Path(DEFAULT_MODEL_DIR) / "torch"))


def get_model_path(model_name: str) -> Path:
    """Resolve a model directory below the configured model cache."""
    model_root = os.environ.get("AUDIO_TOOLKIT_MODEL_DIR", DEFAULT_MODEL_DIR)
    return Path(model_root).expanduser() / model_name


__all__ = ["DEFAULT_MODEL_DIR", "get_model_path"]
