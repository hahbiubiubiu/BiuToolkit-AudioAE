def asr_model(model_name, device='cuda', eval_mode=False, model_dir=None):
    if model_name == "qwen2_audio":
        from . import qwen2_audio
        return qwen2_audio.asr_model(model_dir=model_dir, device=device)
    elif model_name == "qwen3_asr":
        from . import qwen3_asr
        return qwen3_asr.asr_model(model_dir=model_dir, device=device)
    else:
        raise ValueError(f"Unsupported Qwen ASR model: {model_name}")

def asr_func(model_name, device='cuda', by_file=False, eval_mode=True, model_dir=None):
    if model_name == "qwen2_audio":
        from . import qwen2_audio
        return qwen2_audio.asr_func(model_dir=model_dir, device=device, by_file=by_file)
    elif model_name == "qwen3_asr":
        from . import qwen3_asr
        return qwen3_asr.asr_func(model_dir=model_dir, device=device, by_file=by_file)
    else:
        raise ValueError(f"Unsupported Qwen ASR model: {model_name}")

def get_model_list():
    return [
        "qwen2_audio",
        "qwen3_asr",
    ]
