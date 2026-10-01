from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from biu_toolkit import get_model_path


DEFAULT_MODEL_NAME = "Qwen2-Audio-7B-Instruct"


def asr_model(model_dir=None, device=None):
    model_path = Path(model_dir).expanduser() if model_dir else get_model_path(DEFAULT_MODEL_NAME)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory '{model_path}' not found. "
            "Set BIU_TOOLKIT_MODEL_DIR or pass model_dir explicitly."
        )

    processor = AutoProcessor.from_pretrained(model_path, sampling_rate=16000)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device or "auto",
    )
    return model, processor


def parse_input(processor, conversations, device):
    """Convert conversation audio and prompts to model inputs."""
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
                        raise FileNotFoundError(f"找不到本地文件: {audio_data}")

                waveform, _ = librosa.load(
                    audio_data,
                    sr=processor.feature_extractor.sampling_rate,
                )
                audios.append(waveform)

    inputs = processor(text=text, audio=audios, return_tensors="pt", padding=True)
    return inputs.to(device)


def qwen_transcribe(model, processor, audio_path, device):
    """Transcribe one audio file."""
    prompt = (
        "Transcribe the audio content directly. "
        "Do NOT output prefixes like 'The transcription is'. "
        "Do NOT wrap the output in quotes. "
        "Output ONLY the transcribed text."
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

    try:
        inputs = parse_input(processor, conversations, device)
        generate_ids = model.generate(**inputs, max_length=256)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        response = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return response[0]
    except Exception as exc:
        print(f"Transcription failed for {audio_path}: {exc}")
        return ""


def asr_func(model_dir=None, device="cuda:0", by_file=True):
    model, processor = asr_model(model_dir=model_dir, device=device)
    if not by_file:
        raise NotImplementedError("Qwen2-Audio only supports file-based transcription for now.")
    return lambda wav_path: qwen_transcribe(model, processor, wav_path, device)
