_config = {}

_config["rid"] = {
    "noisy_ball_xy": {
        "batch_size": 512,
        "hidden_size": 256,
        "layer_num": 8,
        "lr": 1e-3,
        "adam_betas": (0.5, 0.999),
        "weight_decay": 1e-5,
        "lrs_patience": 10,
        "max_epoch": 400,
        "tau": 1
    },
    "noisy_ball_y": {
        "batch_size": 512,
        "hidden_size": 256,
        "layer_num": 8,
        "lr": 1e-3,
        "adam_betas": (0.5, 0.999),
        "weight_decay": 1e-5,
        "lrs_patience": 10,
        "max_epoch": 400,
        "tau": 1
    },
    "noisy_ball_x": {
        "batch_size": 512,
        "hidden_size": 256,
        "layer_num": 8,
        "lr": 1e-3,
        "adam_betas": (0.5, 0.999),
        "weight_decay": 1e-5,
        "lrs_patience": 10,
        "max_epoch": 400,
        "tau": 1
    },
    "noisy_kine_x": {
        "batch_size": 512,
        "hidden_size": 256,
        "layer_num": 8,
        "lr": 1e-3,
        "adam_betas": (0.5, 0.999),
        "weight_decay": 1e-5,
        "lrs_patience": 10,
        "max_epoch": 200,
        "tau": 1,
    },
    "noisy_kine_y": {
        "batch_size": 512,
        "hidden_size": 256,
        "layer_num": 8,
        "lr": 1e-3,
        "adam_betas": (0.5, 0.999),
        "weight_decay": 1e-5,
        "lrs_patience": 10,
        "max_epoch": 200,
        "tau": 1,
    },
    "noisy_kine_xy": {
        "batch_size": 512,
        "hidden_size": 256,
        "layer_num": 8,
        "lr": 1e-3,
        "adam_betas": (0.5, 0.999),
        "weight_decay": 1e-5,
        "lrs_patience": 10,
        "max_epoch": 200,
        "tau": 1,
    },
    "noisy_mm_x": {
        "batch_size": 512,
        "hidden_size": 256,
        "layer_num": 8,
        "lr": 1e-3,
        "adam_betas": (0.5, 0.999),
        "weight_decay": 1e-5,
        "lrs_patience": 10,
        "max_epoch": 200,
        "tau": 1,
    },
    "noisy_mm_y": {
        "batch_size": 512,
        "hidden_size": 256,
        "layer_num": 8,
        "lr": 1e-3,
        "adam_betas": (0.5, 0.999),
        "weight_decay": 1e-5,
        "lrs_patience": 10,
        "max_epoch": 200,
        "tau": 2,
    },
    "noisy_mm_xy": {
        "batch_size": 512,
        "hidden_size": 256,
        "layer_num": 8,
        "lr": 1e-3,
        "adam_betas": (0.5, 0.999),
        "weight_decay": 1e-5,
        "lrs_patience": 10,
        "max_epoch": 200,
        "tau": 5,
    },
}


def get_config(method, task):
    if method not in _config:
        raise NotImplementedError()
    else:
        if task not in _config[method]:
            return _config[method]["default"]
        else:
            return _config[method][task]
