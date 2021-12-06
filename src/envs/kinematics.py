##################################################
# @author Thyrix (Jia-Qi) Yang
# @email thyrixyang@gmail.com
# @create date 2021-12-06 16:17:02
# 
# RID-Noise: Towards Robust Inverse Design under Noisy Environments, AAAI'22
##################################################

import numpy as np

from .base import ForwardEnv


class KinematicsEnv(ForwardEnv):

    def __init__(self,
                 normalize=True,
                 l1=0.5,
                 l2=0.5,
                 l3=1,
                 sigma1=0.25,
                 sigma2=0.5,
                 sigma3=0.5,
                 sigma4=0.5):
        self.normalize = normalize
        self.input_shape = (4,)
        self.output_shape = (2,)
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.sigma4 = sigma4
        self.x_mean = np.array([[0, 0, 0, 0]])
        self.y_mean = np.array([[1.519, 0]])
        self.x_std = np.array([[0.25, 0.5, 0.5, 0.5]])
        self.y_std = np.array([[0.437, 0.582]])
        self.name = "kinematics"

    def clip_x(self, x):
        x[:, 0] = np.clip(x[:, 0], -3*self.sigma1, 3*self.sigma1)
        x[:, 1] = np.clip(x[:, 1], -3*self.sigma2, 3*self.sigma2)
        x[:, 2] = np.clip(x[:, 2], -3*self.sigma3, 3*self.sigma3)
        x[:, 3] = np.clip(x[:, 3], -3*self.sigma4, 3*self.sigma4)
        return x

    def forward(self, x):
        assert len(x.shape) == 2 and x.shape[1] == 4
        if self.normalize:
            x = x * self.x_std + self.x_mean
        x = self.clip_x(x)
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        y1 = self.l1*np.sin(x2) \
            + self.l2*np.sin(x3 - x2) \
            + self.l3*np.sin(x4 - x3 - x2) + x1
        y2 = self.l1*np.cos(x2) \
            + self.l2*np.cos(x3 - x2) \
            + self.l3*np.cos(x4 - x2 - x3)
        y1 = np.expand_dims(y1, axis=1)
        y2 = np.expand_dims(y2, axis=1)
        y = np.concatenate([y2, y1], axis=1)
        if self.normalize:
            y = (y - self.y_mean) / self.y_std
        return y

    def generate_random_x(self, num=8):
        output = np.zeros([num, 4])
        output[:, 0] = np.random.normal(0, self.sigma1, size=num)
        output[:, 1] = np.random.normal(0, self.sigma2, size=num)
        output[:, 2] = np.random.normal(0, self.sigma3, size=num)
        output[:, 3] = np.random.normal(0, self.sigma4, size=num)
        if self.normalize:
            output = (output - self.x_mean) / self.x_std
        return output
