##################################################
# @author Thyrix (Jia-Qi) Yang
# @email thyrixyang@gmail.com
# @create date 2021-12-06 16:17:02
# 
# RID-Noise: Towards Robust Inverse Design under Noisy Environments, AAAI'22
##################################################

import numpy as np

from .ballistics import BallisticsEnv
from .kinematics import KinematicsEnv
from .meta_material import MetaMaterialEnv
from .base import NoisyEnv


def get_noisy_ball_x():
    noise = 0.5
    base_env = BallisticsEnv(normalize=True)

    def x_noise_fn(x):
        x = np.copy(x)
        x3 = np.copy(x[:, 3])
        s = np.logical_and(x3 < 0.3, x3 > -0.3)
        x3[s] = x3[s] + \
            np.random.randn(*x3[s].shape) * noise
        x[:, 3] = x3
        return x

    def y_noise_fn(y):
        y = np.copy(y)
        return y
    noisy_env = NoisyEnv(base_env, x_noise_fn, y_noise_fn)
    return noisy_env


def get_noisy_ball_y():
    noise = 0.3
    base_env = BallisticsEnv(normalize=True)

    def x_noise_fn(x):
        return x

    def y_noise_fn(y):
        y = np.copy(y)
        s = y < 0.3
        _noise = np.random.randn(*y.shape) * noise
        y[s] = y[s] + _noise[s]
        return y
    noisy_env = NoisyEnv(base_env, x_noise_fn, y_noise_fn)
    return noisy_env


def get_noisy_ball_xy():
    base_env = BallisticsEnv(normalize=True)

    def x_noise_fn(x):
        x = np.copy(x)
        x3 = np.copy(x[:, 3])
        s = np.logical_and(x3 < 0.3, x3 > -0.3)
        x3[s] = x3[s] + \
            np.random.randn(*x3[s].shape) * 0.5
        x[:, 3] = x3
        return x

    def y_noise_fn(y):
        y = np.copy(y)
        s = y < 0.3
        _noise = np.random.randn(*y.shape) * 0.3
        y[s] = y[s] + _noise[s]
        return y
    noisy_env = NoisyEnv(base_env, x_noise_fn, y_noise_fn)
    return noisy_env


def get_noisy_kine_x():
    base_env = KinematicsEnv(normalize=True)

    def x_noise_fn(x):
        x = np.copy(x)
        x1 = np.copy(x[:, 1])
        s = np.copy(x1 < 0)
        x1[s] = x1[s] + np.random.randn(*x1[s].shape) * 0.5
        x[:, 1] = x1
        return x

    def y_noise_fn(y):
        y = np.copy(y)
        return y
    noisy_env = NoisyEnv(base_env, x_noise_fn, y_noise_fn)
    return noisy_env


def get_noisy_kine_y():
    base_env = KinematicsEnv(normalize=True)

    def x_noise_fn(x):
        x = np.copy(x)
        return x

    def y_noise_fn(y):
        y = np.copy(y)
        y1 = np.copy(y[:, 1])
        s = y1 < 0
        _noise = np.random.randn(*y1.shape) * 0.5
        y1[s] = y1[s] + _noise[s]
        y[:, 1] = y1
        return y
    noisy_env = NoisyEnv(base_env, x_noise_fn, y_noise_fn)
    return noisy_env


def get_noisy_kine_xy():
    base_env = KinematicsEnv(normalize=True)

    def x_noise_fn(x):
        x = np.copy(x)
        x1 = np.copy(x[:, 1])
        s = np.copy(x1 < 0)
        x1[s] = x1[s] + np.random.randn(*x1[s].shape) * 0.5
        x[:, 1] = x1
        return x

    def y_noise_fn(y):
        y = np.copy(y)
        y1 = np.copy(y[:, 1])
        s = y1 < 0
        _noise = np.random.randn(*y1.shape) * 0.5
        y1[s] = y1[s] + _noise[s]
        y[:, 1] = y1
        return y
    noisy_env = NoisyEnv(base_env, x_noise_fn, y_noise_fn)
    return noisy_env


def get_noisy_mm_x():
    base_env = MetaMaterialEnv()

    def x_noise_fn(x):
        x = np.copy(x)
        s = x[:, 0] < 0
        x[s] = x[s] + np.random.randn(*x.shape)[s] * 0.5
        return x

    def y_noise_fn(y):
        y = np.copy(y)
        return y
    noisy_env = NoisyEnv(base_env, x_noise_fn, y_noise_fn)
    return noisy_env


def get_noisy_mm_y():
    base_env = MetaMaterialEnv()

    def x_noise_fn(x):
        x = np.copy(x)
        return x

    def y_noise_fn(y):
        y = np.copy(y)
        s = y[:, 0] < 0.6
        y[s] = y[s] + np.random.randn(*y.shape)[s] * 0.3
        return y
    noisy_env = NoisyEnv(base_env, x_noise_fn, y_noise_fn)
    return noisy_env


def get_noisy_mm_xy():
    base_env = MetaMaterialEnv()

    def x_noise_fn(x):
        x = np.copy(x)
        s = x[:, 0] < 0
        x[s] = x[s] + np.random.randn(*x.shape)[s] * 0.5
        return x

    def y_noise_fn(y):
        y = np.copy(y)
        s = y[:, 0] < 0.6
        y[s] = y[s] + np.random.randn(*y.shape)[s] * 0.3
        return y

    noisy_env = NoisyEnv(base_env, x_noise_fn, y_noise_fn)
    return noisy_env
