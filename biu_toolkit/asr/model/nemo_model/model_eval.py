
import logging
from nemo.collections.asr import models as asr_models
from nemo.utils import logging as nemo_logging
nemo_logging.setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

from nemo.collections.speechlm2.models import SALM
from nemo.collections.asr.models import EncDecMultiTaskModel

def asr_model(model_name, device="cuda"):
    if model_name == 'canary-qwen':
        model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
    elif model_name == 'canary':
        model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')
        decode_cfg = model.cfg.decoding
        decode_cfg.beam.beam_size = 1
        model.change_decoding_strategy(decode_cfg)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    return model.to(device)


def transcribe_with_file(model_name, model, audio_path):
    if model_name == 'canary-qwen':
        prompts = [[{
            "role": "user", 
            "content": f"Transcribe the following: {model.audio_locator_tag}", 
            "audio": [audio_path]
        }]]
        answer_ids = model.generate(prompts=prompts, max_new_tokens=128)
        return model.tokenizer.ids_to_text(answer_ids[0].cpu())
    
    elif model_name == 'canary':
        predicted_text = model.transcribe(
            audio=[audio_path],
            batch_size=1,
        )[0].text
        return predicted_text
    
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def asr_func(model_name, device='cuda', by_file=True):
    model = asr_model(model_name=model_name, device=device)
    if by_file:
        return lambda wav_path: transcribe_with_file(model_name, model, wav_path)
    else:
        raise NotImplementedError("Transcription from audio tensor is not implemented yet.")

def get_model_list():
    return ['canary-qwen', 'canary']