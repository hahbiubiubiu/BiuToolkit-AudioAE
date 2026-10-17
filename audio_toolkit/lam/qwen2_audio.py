from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from audio_toolkit import get_model_path


DEFAULT_MODEL_NAME = "Qwen2-Audio-7B-Instruct"


def load_model(model_dir=None, device=None):
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

    processor = AutoProcessor.from_pretrained(model_path, sampling_rate=16_000)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device or "auto",
    )
    return model, processor


def get_func(model_dir=None, device="cuda:0", by_file=True):
    if not by_file:
        raise NotImplementedError(
            "Qwen2-Audio only supports file-based transcription."
        )

    model, processor = load_model(model_dir=model_dir, device=device)

    def transcribe(audio_path):
        return _transcribe(model, processor, audio_path, device)

    return transcribe


def _transcribe(model, processor, audio_path, device):
    prompt = (
        "Transcribe the audio content directly. "
        "Do not add a prefix or wrap the output in quotes. "
        "Output only the transcribed text."
    )
    conversations = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": str(audio_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = _parse_input(processor, conversations, device)
    generated_ids = model.generate(**inputs, max_length=256)
    generated_ids = generated_ids[:, inputs.input_ids.size(1) :]
    responses = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return responses[0]


def _parse_input(processor, conversations, device):
    text = [
        processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        for conversation in conversations
    ]
    audios = []
    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue
            for element in message["content"]:
                if element["type"] != "audio":
                    continue
                audio_path_or_url = element["audio_url"]
                if audio_path_or_url.startswith(("http://", "https://")):
                    audio_data = BytesIO(urlopen(audio_path_or_url).read())
                else:
                    audio_data = Path(audio_path_or_url).expanduser()
                    if not audio_data.exists():
                        raise FileNotFoundError(f"Audio file not found: {audio_data}")

                waveform, _ = librosa.load(
                    audio_data,
                    sr=processor.feature_extractor.sampling_rate,
                )
                audios.append(waveform)

    inputs = processor(text=text, audio=audios, return_tensors="pt", padding=True)
    return inputs.to(device)
