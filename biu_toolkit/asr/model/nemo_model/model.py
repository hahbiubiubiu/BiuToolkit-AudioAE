from .model_train import get_model_list as get_train_model_list
from .model_eval import get_model_list as get_eval_model_list


def asr_model(model_name, device='cuda', eval_mode=True):
    if model_name in get_eval_model_list():
        from . import model_eval as model_target
        model = model_target.asr_model(model_name=model_name, device=device)
    else:
        from . import model_train as model_target
        model = model_target.asr_model(model_name=model_name, device=device, eval_mode=eval_mode)
    return model

def asr_func(model_name, device='cuda', by_file=True, eval_mode=True):
    if model_name in get_eval_model_list():
        from . import model_eval as model_target
        func = model_target.asr_func(model_name=model_name, device=device, by_file=by_file)
    else:
        from . import model_train as model_target
        func = model_target.asr_func(model_name=model_name, device=device,  by_file=by_file, eval_mode=eval_mode)
    return func

def get_model_list(model_type=None):
    train_model_list = get_train_model_list()
    if model_type == "all":
        eval_model_list = get_eval_model_list()
        return train_model_list + eval_model_list
    else:
        return train_model_list
