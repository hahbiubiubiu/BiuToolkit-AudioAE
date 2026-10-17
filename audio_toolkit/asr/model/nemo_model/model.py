import torch
import torch.nn.functional as F
import torchaudio

from audio_toolkit.asr.model.base import ASRModel, AudioPath

from .asr_factory import get_model_list, get_model_type, load_asr_model
from .processor import DifferentiableAudioPreprocessor


class NemoASRModel(ASRModel):
    """Differentiable NeMo implementation of the shared ASR interface."""

    def __init__(self, model_name, device="cuda", load_best=False):
        model = load_asr_model(
            model_name,
            device=device,
            load_best=load_best,
        )
        super().__init__(model, device)
        self.model_type = get_model_type(model_name)
        self.preprocessor = DifferentiableAudioPreprocessor(
            self.model.preprocessor
        ).to(device)
        self.preprocessor.eval()

    def compute_loss(
        self,
        audio: torch.Tensor,
        target: str,
        sample_rate: int = 16_000,
    ) -> torch.Tensor:
        audio = self._prepare_audio(audio, sample_rate)
        label, label_len = self._tokenize(target)
        audio_len = torch.tensor(
            [audio.size(1)],
            dtype=torch.int32,
            device=audio.device,
        )
        feature, feature_len = self.preprocessor(x=audio, seq_len=audio_len)
        encoded, encoded_len = self.model.encoder(
            audio_signal=feature,
            length=feature_len,
        )

        if self.model_type == "ctc":
            log_probs = self.model.decoder(encoder_output=encoded)
            log_probs = F.log_softmax(log_probs, dim=2).transpose(0, 1)
            return F.ctc_loss(
                log_probs=log_probs,
                targets=label,
                input_lengths=encoded_len,
                target_lengths=label_len,
                blank=self.model.decoder.num_classes_with_blank - 1,
                reduction="mean",
            )

        encoded_len = encoded_len.to(dtype=torch.int32)
        decoded_out, _, _ = self.model.decoder(
            targets=label,
            target_length=label_len,
        )
        self.model.joint._fuse_loss_wer = False
        joint_out = self.model.joint(
            encoder_outputs=encoded,
            decoder_outputs=decoded_out,
        )

        if self.model_type == "tdt":
            return self.model.loss(
                log_probs=joint_out,
                targets=label,
                input_lengths=encoded_len,
                target_lengths=label_len,
            )

        max_encoded_len = encoded_len.max().item()
        joint_out = joint_out[:, :max_encoded_len, :, :]
        rnnt_loss = torchaudio.functional.rnnt_loss(
            logits=joint_out,
            targets=label,
            logit_lengths=encoded_len,
            target_lengths=label_len,
            blank=self.model.decoder.blank_idx,
            reduction="mean",
        )
        if self.model_type == "rnnt":
            return rnnt_loss

        ctc_logits = self.model.ctc_decoder(encoder_output=encoded)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
        ctc_loss = F.ctc_loss(
            log_probs=ctc_log_probs,
            targets=label,
            input_lengths=encoded_len,
            target_lengths=label_len,
            blank=self.model.ctc_decoder.num_classes_with_blank - 1,
            reduction="mean",
            zero_infinity=True,
        )
        ctc_weight = self.model.cfg.aux_ctc.ctc_loss_weight
        return rnnt_loss + ctc_weight * ctc_loss

    @torch.inference_mode()
    def transcribe(self, audio: torch.Tensor, sample_rate: int = 16_000) -> str:
        audio = self._prepare_audio(audio, sample_rate)
        audio_len = torch.tensor(
            [audio.size(1)],
            dtype=torch.int32,
            device=audio.device,
        )
        if self.model_type == "ctc":
            _, encoded_len, predictions = self.model(
                input_signal=audio,
                input_signal_length=audio_len,
            )
            return self.model.decoding.ctc_decoder_predictions_tensor(
                predictions,
                encoded_len,
            )[0].text

        processed, processed_len = self.model.preprocessor(
            input_signal=audio,
            length=audio_len,
        )
        encoded, encoded_len = self.model.encoder(
            audio_signal=processed,
            length=processed_len,
        )
        return self.model.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded,
            encoded_lengths=encoded_len,
        )[0].text

    def transcribe_file(self, audio_path: AudioPath) -> str:
        audio, sample_rate = torchaudio.load(audio_path)
        return self.transcribe(audio, sample_rate=sample_rate)

    def _tokenize(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model_type == "ctc":
            if hasattr(self.model, "tokenizer"):
                token_ids = self.model.tokenizer.text_to_ids(text)
            elif hasattr(self.model.decoder, "vocabulary"):
                vocabulary = self.model.decoder.vocabulary
                token_ids = [
                    vocabulary.index(character) for character in text.lower()
                ]
            else:
                raise AttributeError("Cannot find tokenizer or vocabulary in the model")
        else:
            token_ids = self.model.tokenizer.text_to_ids(text)

        labels = torch.tensor(
            [token_ids],
            dtype=torch.int32,
            device=self.device,
        )
        label_len = torch.tensor(
            [len(token_ids)],
            dtype=torch.int32,
            device=self.device,
        )
        return labels, label_len


def asr_model(model_name, device="cuda", eval_mode=True):
    return NemoASRModel(model_name, device=device, load_best=eval_mode)


def asr_func(model_name, device="cuda", by_file=True, eval_mode=True):
    model = asr_model(model_name=model_name, device=device, eval_mode=eval_mode)
    return model.transcribe_file if by_file else model.transcribe
