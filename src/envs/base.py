##################################################
# @author Thyrix (Jia-Qi) Yang
# @email thyrixyang@gmail.com
# @create date 2021-12-06 16:17:02
# 
# RID-Noise: Towards Robust Inverse Design under Noisy Environments, AAAI'22
##################################################

import abc

import numpy as np


class ForwardEnv(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def generate_random_x(self, num):
        pass

    def get_input_shape(self):
        return self.input_shape

    def get_output_shape(self):
        return self.output_shape

    def generate_data(self, num, seed):
        current_random_state = np.random.get_state()
        np.random.seed(seed)

        x = self.generate_random_x(num)
        y = self.forward(x)

        np.random.set_state(current_random_state)
        return x, y


class NoisyEnv(ForwardEnv):

    def __init__(self, env, noise_fn_x, noise_fn_y):
        self.env = env
        self.input_shape = env.input_shape
        self.output_shape = env.output_shape
        self.noise_fn_x = noise_fn_x
        self.noise_fn_y = noise_fn_y

    def forward(self, x):
        _x = self.noise_fn_x(x)
        _y = self.env.forward(_x)
        y = self.noise_fn_y(_y)
        return y

    def generate_random_x(self, num):
        return self.env.generate_random_x(num)
