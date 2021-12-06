##################################################
# @author Thyrix (Jia-Qi) Yang
# @email thyrixyang@gmail.com
# @create date 2021-12-06 16:17:02
# 
# RID-Noise: Towards Robust Inverse Design under Noisy Environments, AAAI'22
##################################################

from envs.noisy_envs import get_noisy_ball_x, get_noisy_kine_x
from envs.noisy_envs import get_noisy_kine_y, get_noisy_kine_xy
from envs.noisy_envs import get_noisy_ball_y, get_noisy_ball_xy
from envs.noisy_envs import get_noisy_mm_x, get_noisy_mm_y, get_noisy_mm_xy

env_dict = {
    "noisy_kine_x": get_noisy_kine_x,
    "noisy_ball_x": get_noisy_ball_x,
    "noisy_kine_y": get_noisy_kine_y,
    "noisy_ball_y": get_noisy_ball_y,
    "noisy_kine_xy": get_noisy_kine_xy,
    "noisy_ball_xy": get_noisy_ball_xy,
    "noisy_mm_x": get_noisy_mm_x,
    "noisy_mm_y": get_noisy_mm_y,
    "noisy_mm_xy": get_noisy_mm_xy
}


def get_noise_data(config):
    name = config["task"]
    if name not in env_dict.keys():
        raise NotImplementedError()
    env = env_dict[name]()
    if "mm" in name:
        train_num = 20000
    else:
        train_num = 10000
    x, y = env.generate_data(train_num, seed=0)  # 10000 training point
    train_n = int(train_num*(0.8))
    train_x = x[:train_n]
    train_y = y[:train_n]
    vali_x = x[train_n:]
    vali_y = y[train_n:]

    # generate 10000 testing point, use different seed to ensure they are different from training set.
    test_x, test_y = env.generate_data(10000, seed=1000)
    data = {
        "train_x": train_x,
        "train_y": train_y,
        "vali_x": vali_x,
        "vali_y": vali_y,
        "test_x": test_x,
        "test_y": test_y
    }
    data["env"] = env
    config.update(data)
    return config
