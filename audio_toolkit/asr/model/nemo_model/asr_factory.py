from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from nemo.collections.asr import models as asr_models
from nemo.utils import logging as nemo_logging

os.environ["NEMO_LOG_LEVEL"] = "ERROR"
nemo_logging.setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


class ModelLoadError(Exception):
    """Exception raised when model loading fails."""


@dataclass(frozen=True)
class ModelSpec:
    model_class_name: str
    default_pretrained: Optional[str]
    model_type: str
    best_pretrained: Optional[str] = None


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "jasper": ModelSpec(
        model_type="ctc",
        model_class_name="EncDecCTCModel",
        default_pretrained="stt_en_jasper10x5dr",
        best_pretrained="stt_en_jasper10x5dr",
    ),
    "quartznet": ModelSpec(
        model_type="ctc",
        model_class_name="EncDecCTCModel",
        default_pretrained="stt_en_quartznet15x5",
        best_pretrained="stt_en_quartznet15x5",
    ),
    "citrinet": ModelSpec(
        model_type="ctc",
        model_class_name="EncDecCTCModelBPE",
        default_pretrained="stt_en_citrinet_256",
        best_pretrained="stt_en_citrinet_1024",
    ),
    "contextnet": ModelSpec(
        model_type="rnnt",
        model_class_name="EncDecRNNTBPEModel",
        default_pretrained="stt_en_contextnet_256",
        best_pretrained="stt_en_contextnet_1024",
    ),
    "squeezeformer": ModelSpec(
        model_type="ctc",
        model_class_name="EncDecCTCModelBPE",
        default_pretrained="stt_en_squeezeformer_ctc_xsmall_ls",
        best_pretrained="stt_en_squeezeformer_ctc_large_ls",
    ),
    "conformer_ctc": ModelSpec(
        model_type="ctc",
        model_class_name="EncDecCTCModelBPE",
        default_pretrained="stt_en_conformer_ctc_small_ls",
        best_pretrained="stt_en_conformer_ctc_xlarge",
    ),
    "conformer_rnnt": ModelSpec(
        model_type="rnnt",
        model_class_name="EncDecRNNTBPEModel",
        default_pretrained="stt_en_conformer_transducer_small",
        best_pretrained="stt_en_conformer_transducer_xxlarge",
    ),
    "fastconformer_ctc": ModelSpec(
        model_type="ctc",
        model_class_name="EncDecCTCModelBPE",
        default_pretrained="stt_en_fastconformer_ctc_large",
        best_pretrained="stt_en_fastconformer_ctc_xxlarge",
    ),
    "fastconformer_rnnt": ModelSpec(
        model_type="rnnt",
        model_class_name="EncDecRNNTBPEModel",
        default_pretrained="stt_en_fastconformer_transducer_large",
        best_pretrained="stt_en_fastconformer_transducer_xxlarge",
    ),
    "fastconformer_hybrid": ModelSpec(
        model_type="hybrid",
        model_class_name="EncDecHybridRNNTCTCBPEModel",
        default_pretrained="stt_en_fastconformer_hybrid_large_pc",
        best_pretrained="stt_en_fastconformer_hybrid_large_pc",
    ),
    "parakeet_ctc": ModelSpec(
        model_type="ctc",
        model_class_name="EncDecCTCModelBPE",
        default_pretrained="nvidia/parakeet-ctc-0.6b",
        best_pretrained="nvidia/parakeet-ctc-1.1b",
    ),
    "parakeet_rnnt": ModelSpec(
        model_type="rnnt",
        model_class_name="EncDecRNNTBPEModel",
        default_pretrained="nvidia/parakeet-rnnt-1.1b",
        best_pretrained="nvidia/parakeet-rnnt-1.1b",
    ),
    "parakeet_tdt": ModelSpec(
        model_type="tdt",
        model_class_name="ASRModel",
        default_pretrained="nvidia/parakeet-tdt-0.6b-v3",
        best_pretrained="nvidia/parakeet-tdt-1.1b",
    ),
}


def get_model_class(name: str):
    try:
        if name in MODEL_REGISTRY:
            return getattr(asr_models, MODEL_REGISTRY[name].model_class_name)
        return getattr(asr_models, name)
    except AttributeError as exc:
        raise ModelLoadError(f"Unsupported model class: {name}") from exc


def list_available_pretrained(class_name: str) -> List[str]:
    """List available pretrained names for a given NeMo model class."""
    model_class = get_model_class(class_name)
    if hasattr(model_class, "list_available_models"):
        return [
            model.pretrained_model_name
            for model in model_class.list_available_models()
        ]
    return []


def load_asr_model(
    name: str,
    device: str = "cuda",
    pretrained_name: Optional[str] = None,
    model_path: Optional[str] = None,
    load_best: bool = False,
):
    """Load a NeMo ASR model by registry name.

    Args:
        name: registry key (e.g., "quartznet", "conformer_rnnt").
        device: torch device string.
        pretrained_name: override pretrained model name.
        model_path: path to a local .nemo file.
    """
    if name not in MODEL_REGISTRY:
        raise ModelLoadError(f"Unknown model name: {name}")

    spec = MODEL_REGISTRY[name]
    model_class = get_model_class(spec.model_class_name)
    resolved_pretrained = (
        spec.best_pretrained
        if load_best and spec.best_pretrained
        else pretrained_name or spec.default_pretrained
    )

    try:
        if model_path:
            model = model_class.restore_from(model_path)
        elif resolved_pretrained:
            model = model_class.from_pretrained(resolved_pretrained)
        else:
            raise ModelLoadError("No pretrained name or model path provided")
    except Exception as exc:
        available = list_available_pretrained(spec.model_class_name)
        msg = (
            f"Failed to load model '{name}' with "
            f"pretrained='{resolved_pretrained}' or path='{model_path}'.\n"
            f"Error: {exc} \n"
        )
        if available:
            msg += f"Available: {available}"
        raise ModelLoadError(msg) from exc

    model = model.to(device)
    model.eval()
    return model


def get_model_type(name: str) -> str:
    if name not in MODEL_REGISTRY:
        raise ModelLoadError(f"Unknown model name: {name}")
    return MODEL_REGISTRY[name].model_type


def get_model_list() -> List[str]:
    """Get a list of available model names in the registry."""
    return list(MODEL_REGISTRY.keys())
