from __future__ import annotations

from pathlib import Path

import torch
from qwen_asr import Qwen3ASRModel

from audio_toolkit import get_model_path


DEFAULT_MODEL_NAME = "Qwen3-ASR-1.7B"


def load_model(model_dir=None, device="cuda:0"):
    model_path = (
        Path(model_dir).expanduser()
        if model_dir
        else get_model_path(DEFAULT_MODEL_NAME)
    )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory '{model_path}' not found. "
            "Set AUDIO_TOOLKIT_MODEL_DIR or pass model_dir explicitly."
        )
    return Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
        max_inference_batch_size=32,
        max_new_tokens=256,
    )


def get_func(model_dir=None, device="cuda:0", by_file=True):
    model = load_model(model_dir=model_dir, device=device)

    if by_file:

        def transcribe(audio_path):
            results = model.transcribe(
                audio=[str(audio_path)],
                language=["English"],
            )
            return _first_text(results)

        return transcribe

    def transcribe(waveform):
        with torch.inference_mode():
            results = model.transcribe(
                audio=waveform,
                language=["English"],
                max_new_tokens=256,
            )
        return _first_text(results)

    return transcribe


def _first_text(results):
    if not results:
        raise RuntimeError("Qwen3-ASR returned no transcription result")
    return results[0].text.strip()
