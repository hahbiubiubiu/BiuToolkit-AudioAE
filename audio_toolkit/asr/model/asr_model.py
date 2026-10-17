
def get_model(model_name, device="cuda", eval_mode=False):
    if model_name.startswith("whisper"):
        model_size = model_name.split("_", 1)[1]
        from .whisper_model import asr_model as get_whisper_model

        return get_whisper_model(model_size=model_size, device=device)
    if model_name.startswith("nemo"):
        model_name = model_name.split("_", 1)[1]
        from .nemo_model import asr_model as get_nemo_model

        return get_nemo_model(
            model_name=model_name,
            device=device,
            eval_mode=eval_mode,
        )
    raise ValueError(f"Unsupported ASR model: {model_name}")


def get_func(model_name, device="cuda", by_file=False, eval_mode=True):
    if model_name.startswith("whisper"):
        model_size = model_name.split("_", 1)[1]
        from .whisper_model import asr_func as get_whisper_asr_func

        return get_whisper_asr_func(
            model_size=model_size,
            device=device,
            by_file=by_file,
        )
    if model_name.startswith("nemo"):
        model_name = model_name.split("_", 1)[1]
        from .nemo_model import asr_func as get_nemo_asr_func

        return get_nemo_asr_func(
            model_name=model_name,
            device=device,
            by_file=by_file,
            eval_mode=eval_mode,
        )
    raise ValueError(f"Unsupported ASR model: {model_name}")


def get_model_list(model_type=None):
    if model_type == "whisper":
        from .whisper_model import get_model_list as get_whisper_model_list
        return [f"whisper_{name}" for name in get_whisper_model_list()]
    if model_type == "nemo":
        from .nemo_model import get_model_list as get_nemo_model_list
        return [f"nemo_{name}" for name in get_nemo_model_list()]
    if model_type is None:
        return get_model_list("whisper") + get_model_list("nemo")
    raise ValueError(f"Unsupported ASR model type: {model_type}")


__all__ = [
    "get_func",
    "get_model",
    "get_model_list",
]
