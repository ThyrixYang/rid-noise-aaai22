##################################################
# @author Thyrix (Jia-Qi) Yang
# @email thyrixyang@gmail.com
# @create date 2021-12-06 16:17:02
#
# RID-Noise: Towards Robust Inverse Design under Noisy Environments, AAAI'22
##################################################

import numpy as np

from .base import ForwardEnv


class BallisticsEnv(ForwardEnv):

    def __init__(self,
                 normalize=True,
                 g: float = 9.81,
                 m: float = 0.2,
                 k: float = 0.25,
                 mean1=0,
                 sigma1=0.25,
                 mean2=1.5,
                 sigma2=0.25,
                 low1=9,
                 high1=72,
                 lamda=15):
        self.normalize = normalize
        self.input_shape = (4,)
        self.output_shape = (1,)
        self.g = g
        self.m = m
        self.k = k
        self.name = "ballistics"
        self.x_mean = np.array([[0, 1.5, 0.706, 15]])
        self.y_mean = np.array([[7.38]])
        self.x_std = np.array([[0.25, 0.25, 0.3173, 3.89]])
        self.y_std = np.array([[2.79]])
        self.mean1 = mean1
        self.sigma1 = sigma1
        self.mean2 = mean2
        self.sigma2 = sigma2
        self.low1 = low1
        self.high1 = high1
        self.lamda = lamda

    def forward(self, x):
        if self.normalize:
            x = x * self.x_std + self.x_mean
        x = self.clip_x(x)
        xs, ys = self.trajectories_from_parameters(x)
        y_final, valid_index = self.impact_from_trajectories(xs, ys)
        y = y_final[:, None]
        if self.normalize:
            y = (y - self.y_mean) / self.y_std
        return y

    def clip_x(self, x):
        x[:, 0] = np.clip(x[:, 0], a_min=self.mean1 - 3 *
                          self.sigma1, a_max=self.mean1 + 3 * self.sigma1)
        x[:, 1] = np.clip(x[:, 1], a_min=self.mean2 - 3 *
                          self.sigma2, a_max=self.mean2 + 3 * self.sigma2)
        x[:, 2] = np.clip(x[:, 2], a_min=np.radians(
            self.low1), a_max=np.radians(self.high1))
        x[:, 3] = np.clip(x[:, 3], a_min=1e-4, a_max=90)
        return x

    def generate_random_x(self, num=8):
        output = np.zeros([num, 4])
        output[:, 0] = np.random.normal(self.mean1, self.sigma1, size=num)
        output[:, 1] = np.random.normal(self.mean2, self.sigma2, size=num)
        output[:, 2] = np.radians(np.random.uniform(
            self.low1, self.high1, size=num))
        output[:, 3] = np.random.poisson(
            self.lamda, size=num) + np.random.uniform(-0.5, 0.5, size=num)
        if self.normalize:
            output = (output - self.x_mean) / self.x_std
        return output

    def trajectories_from_parameters(self, x):
        x0, y0, angle, v0 = np.split(x, 4, axis=1)

        v0 = np.repeat(v0, 1500, axis=-1)
        angle = np.repeat(angle, 1500, axis=-1)
        t = np.repeat(np.linspace(0, 6, 1500)[None, :], x.shape[0], axis=0)
        vx = v0 * np.cos(angle)
        vy = v0 * np.sin(angle)

        expterm = np.exp(-self.k * t / self.m) - 1
        xt = x0 - (vx * self.m / self.k) * expterm
        yt = y0 - (self.m / (self.k * self.k)) * \
            ((self.g * self.m + vy * self.k) * expterm + self.g * t * self.k)
        return xt, yt

    def impact_from_trajectories(self, xs, ys):
        bs = xs.shape[0]
        ys_peak = np.argmax(ys, axis=1)
        ys_after_peak = np.where(
            xs < xs[np.arange(xs.shape[0]), ys_peak][:, None], 1, ys)
        ys_after_peak[:, -1] = -10000
        xs_impact_index = np.diff(np.signbit(ys_after_peak))
        xs_impact_index = xs_impact_index.nonzero()
        xs_impact = xs[xs_impact_index]
        assert xs_impact.shape[0] == bs
        return xs_impact, xs_impact_index[0]
