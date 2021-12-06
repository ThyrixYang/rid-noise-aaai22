##################################################
# @author Thyrix (Jia-Qi) Yang
# @email thyrixyang@gmail.com
# @create date 2021-12-06 16:17:02
# 
# RID-Noise: Towards Robust Inverse Design under Noisy Environments, AAAI'22
##################################################

import argparse
import random
import os

import torch
import numpy as np

import rid
from configs import get_config
from utils import get_noise_data

_method_selector = {
    "rid": rid,
}


def main(args):
    print("args: {}".format(args))
    method = args.method
    task = args.task
    device = args.device
    config = get_config(method=method, task=task)
    config.update({
        "device": device,
        "method": method,
        "task": task,
        "seed": 0,
        "query_num": 1,
    })
    prefix = "{}_{}".format(method, task)
    config["prefix"] = prefix

    # ensure reproducibility
    seed = config["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    config = get_noise_data(config)
    method_fn = _method_selector[method].train_and_inference
    print("config: {}".format(
        {k: v for k, v in config.items() if not isinstance(v, np.ndarray)}))
    output = method_fn(config)
    pred_xs = output["pred_xs"]
    test_y = output["test_y"]
    query_num, test_num, x_dim = pred_xs.shape
    env = config["env"]
    eval_y = env.forward(np.reshape(pred_xs, (-1, x_dim)))
    eval_y = np.reshape(eval_y, (test_num, -1))
    err = np.mean((eval_y - test_y)**2, axis=-1)
    mean_err = np.mean(err)
    std_err = np.std(err)
    print("config: {}".format(
        {k: v for k, v in config.items() if not isinstance(v, np.ndarray)}))
    print("resim_err_mean: {}, resim_err_std: {}".format(mean_err, std_err))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        choices=["rid"])
    parser.add_argument("--task", type=str, required=True,
                        choices=["noisy_kine_x", "noisy_ball_x",
                                 "noisy_kine_y", "noisy_ball_y",
                                 "noisy_kine_xy", "noisy_ball_xy",
                                 "noisy_mm_x", "noisy_mm_y", "noisy_mm_xy"])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    main(args)
