from __future__ import annotations

import io
import os
import wave
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ASRAPI(ABC):
    """Shared interface for remote speech-recognition providers."""

    def __init__(
        self,
        *,
        env_file: str | os.PathLike[str] | None = None,
        load_env: bool = True,
    ):
        self.env_file = (
            Path.cwd() / ".env"
            if env_file is None
            else Path(env_file).expanduser()
        )
        self._load_env = load_env
        if load_env:
            self._load_environment()

    def _load_environment(self) -> None:
        from dotenv import load_dotenv

        load_dotenv(self.env_file, override=False)

    def transcribe_file(self, audio_path: str | os.PathLike[str]) -> str:
        path = Path(audio_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        audio_data = path.read_bytes()
        if not audio_data:
            raise ValueError(f"Audio file is empty: {path}")
        return self._transcribe_bytes(audio_data, sample_rate=None)

    def transcribe_data(self, audio: Any, sample_rate: int = 16_000) -> str:
        audio_data = audio_to_wav_bytes(audio, sample_rate)
        return self._transcribe_bytes(audio_data, sample_rate=sample_rate)

    def transcribe(self, audio: Any, sample_rate: int = 16_000) -> str:
        if isinstance(audio, (str, os.PathLike)):
            return self.transcribe_file(audio)
        return self.transcribe_data(audio, sample_rate=sample_rate)

    @abstractmethod
    def _transcribe_bytes(
        self,
        audio_data: bytes,
        sample_rate: int | None,
    ) -> str:
        """Send encoded audio bytes to the provider."""


def audio_to_wav_bytes(audio: Any, sample_rate: int) -> bytes:
    if isinstance(audio, bytes):
        if not audio:
            raise ValueError("Audio data is empty")
        return audio
    if isinstance(audio, bytearray):
        return audio_to_wav_bytes(bytes(audio), sample_rate)
    if isinstance(audio, io.BytesIO):
        return audio_to_wav_bytes(audio.getvalue(), sample_rate)

    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()

    import numpy as np

    waveform = np.asarray(audio).squeeze()
    if waveform.ndim != 1:
        raise ValueError(
            f"Audio data must contain one waveform, got shape {waveform.shape}"
        )
    if waveform.size == 0:
        raise ValueError("Audio data is empty")

    if np.issubdtype(waveform.dtype, np.floating):
        waveform = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
    elif waveform.dtype != np.int16:
        waveform = waveform.astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(waveform.tobytes())
    return buffer.getvalue()
