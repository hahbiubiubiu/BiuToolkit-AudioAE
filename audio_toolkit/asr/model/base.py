from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple, Union

import torch


AudioPath = Union[str, Path]


class ASRModel(ABC):
    """Common interface for local differentiable ASR backends."""

    sample_rate = 16_000

    def __init__(self, model: torch.nn.Module, device: Union[str, torch.device]):
        self.model = model
        self.device = torch.device(device)

    @abstractmethod
    def transcribe(self, audio: torch.Tensor, sample_rate: int = 16_000) -> str:
        """Transcribe one waveform tensor."""

    @abstractmethod
    def transcribe_file(self, audio_path: AudioPath) -> str:
        """Transcribe one audio file."""

    @abstractmethod
    def compute_loss(
        self,
        audio: torch.Tensor,
        target: str,
        sample_rate: int = 16_000,
    ) -> torch.Tensor:
        """Compute the ASR objective for a target transcription."""

    def compute_audio_gradient(
        self,
        audio: torch.Tensor,
        target: str,
        sample_rate: int = 16_000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the loss gradient with respect to the input waveform."""
        if not audio.requires_grad:
            audio.requires_grad_(True)
        loss = self.compute_loss(audio, target, sample_rate)
        gradient = torch.autograd.grad(loss, audio)[0]
        return gradient, loss

    def compute_model_gradients(
        self,
        audio: torch.Tensor,
        target: str,
        sample_rate: int = 16_000,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Return gradients for trainable parameters connected to the loss."""
        named_parameters = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        loss = self.compute_loss(audio, target, sample_rate)
        gradients = torch.autograd.grad(
            loss,
            tuple(parameter for _, parameter in named_parameters),
            allow_unused=True,
        )
        return {
            name: gradient
            for (name, _), gradient in zip(named_parameters, gradients)
            if gradient is not None
        }, loss

    def _prepare_audio(
        self,
        audio: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        elif audio.ndim != 2:
            raise ValueError("audio must have shape [time] or [batch, time]")

        if sample_rate != self.sample_rate:
            from torchaudio.functional import resample

            audio = resample(audio, sample_rate, self.sample_rate)
        return audio.to(self.device)
