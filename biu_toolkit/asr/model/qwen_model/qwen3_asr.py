from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import torch
import transformers
from qwen_asr import Qwen3ASRModel

from biu_toolkit import get_model_path

transformers.logging.set_verbosity_error()

DEFAULT_MODEL_NAME = "Qwen3-ASR-1.7B"


def asr_model(model_dir=None, device="cuda:0"):
    model_path = Path(model_dir).expanduser() if model_dir else get_model_path(DEFAULT_MODEL_NAME)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory '{model_path}' not found. "
            "Set BIU_TOOLKIT_MODEL_DIR or pass model_dir explicitly."
        )
    return Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
        max_inference_batch_size=32,
        max_new_tokens=256,
    )

def asr_func(model_dir=None, device="cuda:0", by_file=True):
    model = asr_model(model_dir=model_dir, device=device)
    if by_file:
        def transcribe(wav_path):
            results = model.transcribe(
                audio=[str(wav_path)],
                language=["English"],
            )
            return results[0].text.strip() if results else ""

        return transcribe

    def transcribe(waveform):
        with torch.inference_mode():
            result = model.transcribe(
                audio=waveform,
                language=["English"],
                max_new_tokens=256,
            )
        return result[0].text.strip() if result else ""

    return transcribe
