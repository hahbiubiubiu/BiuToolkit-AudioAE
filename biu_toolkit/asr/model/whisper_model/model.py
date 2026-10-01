import whisper
import torch

from biu_toolkit import get_model_path

class WhisperModel():
    def __init__(self, model_size="large-v3", device='cuda'):
        self.device = device
        self.model = whisper.load_model(
            model_size,
            device=device,
            download_root=str(get_model_path("whisper")),
        )

    def transcribe(self, waveform, sample_rate=16000):
        with torch.inference_mode():
            # print(waveform.shape)
            result = self.model.transcribe(
                waveform.squeeze().detach().to(self.device),
                language="en", temperature=0.0
            )
        return result["text"]
    
    def transcribe_by_file(self, file_path):
        with torch.inference_mode():
            result = self.model.transcribe(file_path, language="en", temperature=0.0)
        return result["text"]

def asr_model(model_size="large-v3", device='cuda'):
    return whisper.load_model(
        model_size,
        device=device,
        download_root=str(get_model_path("whisper")),
    )

def asr_func(model_size="large-v3", device='cuda', by_file=True):
    model = asr_model(model_size=model_size, device=device)
    if by_file:
        return lambda wav_path: model.transcribe(wav_path, language="en", temperature=0.0)["text"]
    else:
        def transcribe(waveform):
            with torch.inference_mode():
                result = model.transcribe(
                    waveform.squeeze().detach().to(device),
                    language="en", temperature=0.0
                )
            return result["text"]
        return transcribe
