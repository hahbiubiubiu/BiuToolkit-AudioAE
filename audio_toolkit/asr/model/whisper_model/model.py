from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn.functional as F

from audio_toolkit import get_model_path
from audio_toolkit.asr.model.base import ASRModel, AudioPath


class WhisperASRModel(ASRModel):
    """OpenAI Whisper implementation of the shared ASR interface."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: Union[str, torch.device] = "cuda",
        language: str = "en",
        model: Optional[torch.nn.Module] = None,
    ):
        if model is None:
            import whisper

            model = whisper.load_model(
                model_size,
                device=device,
                download_root=str(get_model_path("whisper")),
            )
        super().__init__(model, device)
        self.language = language
        self.model.eval()

    def transcribe(self, audio: torch.Tensor, sample_rate: int = 16_000) -> str:
        waveform = self._single_waveform(audio, sample_rate)
        with torch.no_grad():
            result = self.model.transcribe(
                waveform.detach(),
                language=self.language,
                temperature=0.0,
            )
        return result["text"]

    def transcribe_file(self, audio_path: AudioPath) -> str:
        with torch.no_grad():
            result = self.model.transcribe(
                str(audio_path),
                language=self.language,
                temperature=0.0,
            )
        return result["text"]

    def compute_loss(
        self,
        audio: torch.Tensor,
        target: str,
        sample_rate: int = 16_000,
    ) -> torch.Tensor:
        import whisper
        from whisper.tokenizer import get_tokenizer

        waveform = self._single_waveform(audio, sample_rate)
        waveform = whisper.pad_or_trim(waveform)
        mel = whisper.log_mel_spectrogram(
            waveform,
            n_mels=self.model.dims.n_mels,
        )
        mel = mel.to(dtype=next(self.model.parameters()).dtype).unsqueeze(0)

        tokenizer = get_tokenizer(
            self.model.is_multilingual,
            num_languages=self.model.num_languages,
            language=self.language,
            task="transcribe",
        )
        prefix = list(tokenizer.sot_sequence_including_notimestamps)
        target_tokens = tokenizer.encode(target) + [tokenizer.eot]
        decoder_tokens = torch.tensor(
            [prefix + target_tokens[:-1]],
            dtype=torch.long,
            device=self.device,
        )

        logits = self.model(mel, decoder_tokens)
        logits = logits[:, len(prefix) - 1 :]
        labels = torch.tensor(target_tokens, dtype=torch.long, device=self.device)
        return F.cross_entropy(logits[0], labels)

    def transcribe_by_file(self, file_path: AudioPath) -> str:
        """Compatibility alias for the previous public method name."""
        return self.transcribe_file(file_path)

    def _single_waveform(
        self,
        audio: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        audio = self._prepare_audio(audio, sample_rate)
        if audio.shape[0] != 1:
            raise ValueError("WhisperASRModel accepts one waveform at a time")
        return audio[0]


def asr_model(
    model_size: str = "large-v3",
    device: Union[str, torch.device] = "cuda",
) -> WhisperASRModel:
    return WhisperASRModel(model_size=model_size, device=device)


def asr_func(model_size="large-v3", device="cuda", by_file=True):
    model = asr_model(model_size=model_size, device=device)
    return model.transcribe_file if by_file else model.transcribe


def get_model_list() -> list[str]:
    return [
        "tiny",
        "base",
        "small",
        "medium",
        "large-v3",
    ]
