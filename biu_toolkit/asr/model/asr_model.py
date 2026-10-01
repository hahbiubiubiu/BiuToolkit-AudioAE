
def get_model(model_name, device='cuda', eval_mode=False, model_dir=None):
    print(f"Loading ASR model: {model_name} on device: {device}")
    if model_name.startswith("whisper"):
        model_size = model_name.split('_', 1)[1]
        from .whisper_model import asr_model as get_whisper_model
        return get_whisper_model(model_size=model_size, device=device)
    elif model_name.startswith("nemo"):
        model_name = model_name.split("_", 1)[1]
        from .nemo_model import asr_model as get_nemo_model
        return get_nemo_model(model_name=model_name, device=device, eval_mode=eval_mode)
    elif model_name.startswith("qwen"):
        model_name = model_name.split("_", 1)[1]
        from .qwen_model import asr_model as get_qwen_model
        return get_qwen_model(model_name=model_name, device=device, eval_mode=eval_mode, model_dir=model_dir)
    else:
        raise ValueError(f"Unsupported ASR model: {model_name}")

def get_func(model_name, device='cuda', by_file=False, eval_mode=True, model_dir=None):
    print(f"Loading ASR function: {model_name} on device: {device}")
    if model_name.startswith("whisper"):
        model_size = model_name.split('_', 1)[1]
        from .whisper_model import asr_func as get_whisper_asr_func
        return get_whisper_asr_func(model_size=model_size, device=device, by_file=by_file)
    elif model_name.startswith("nemo"):
        model_name = model_name.split("_", 1)[1]
        from .nemo_model import asr_func as get_nemo_asr_func
        return get_nemo_asr_func(model_name=model_name, device=device, by_file=by_file, eval_mode=eval_mode)
    elif model_name.startswith("qwen"):
        model_name = model_name.split("_", 1)[1]
        from .qwen_model import asr_func as get_qwen_asr_func
        return get_qwen_asr_func(model_name=model_name, device=device, by_file=by_file, eval_mode=eval_mode, model_dir=model_dir)
    else:
        raise ValueError(f"Unsupported ASR model: {model_name}")

def get_model_list(model_type=None):
    if model_type == "whisper":
        return [
            "whisper_tiny",
            "whisper_base",
            "whisper_small",
            "whisper_medium",
            "whisper_large-v3",
        ]
    elif model_type == "nemo":
        from .nemo_model import get_model_list as get_nemo_model_list
        return [f"nemo_{name}" for name in get_nemo_model_list()]
    elif model_type == "qwen":
        from .qwen_model import get_model_list as get_qwen_model_list
        return [f"qwen_{name}" for name in get_qwen_model_list()]
    else:
        return get_model_list("whisper") + get_model_list("nemo")


__all__ = [
    "get_func",
    "get_model",
    "get_model_list",
]