def get_model(model_name, device="cuda", model_dir=None):
    if model_name == "qwen2_audio":
        from .qwen2_audio import load_model

        return load_model(model_dir=model_dir, device=device)
    if model_name == "qwen3_asr":
        from .qwen3_asr import load_model

        return load_model(model_dir=model_dir, device=device)
    raise ValueError(f"Unsupported large audio model: {model_name}")


def get_func(model_name, device="cuda", by_file=False, model_dir=None):
    if model_name == "qwen2_audio":
        from .qwen2_audio import get_func as get_qwen2_func

        return get_qwen2_func(
            model_dir=model_dir,
            device=device,
            by_file=by_file,
        )
    if model_name == "qwen3_asr":
        from .qwen3_asr import get_func as get_qwen3_func

        return get_qwen3_func(
            model_dir=model_dir,
            device=device,
            by_file=by_file,
        )
    raise ValueError(f"Unsupported large audio model: {model_name}")


def get_model_list():
    return [
        "qwen2_audio",
        "qwen3_asr",
    ]
