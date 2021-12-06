##################################################
# @author Thyrix (Jia-Qi) Yang
# @email thyrixyang@gmail.com
# @create date 2021-12-06 16:17:02
# 
# RID-Noise: Towards Robust Inverse Design under Noisy Environments, AAAI'22
##################################################

import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ForwardEnv


class MMNetwork(nn.Module):
    def __init__(self):
        super(MMNetwork, self).__init__()
        self.bp = False
        self.linears = nn.ModuleList([])
        self.bn_linears = nn.ModuleList([])

        # The parameters and the network architecture can be found in
        # https://github.com/BensonRen/BDIMNNA/blob/main/NA/models/meta_material/parameters.txt
        linear_size = [8, 1000, 1000, 1000, 1000, 150]
        conv_out_channel = [4, 4, 4]
        conv_kernel_size = [8, 5, 5]
        conv_stride = [2, 1, 1]
        # Excluding the last one as we need intervals
        for ind, fc_num in enumerate(linear_size[0:-1]):
            self.linears.append(nn.Linear(fc_num, linear_size[ind + 1]))
            self.bn_linears.append(nn.BatchNorm1d(linear_size[ind + 1]))

        # Conv Layer definitions here
        self.convs = nn.ModuleList([])
        # Initialize the in_channel number
        in_channel = 1
        for ind, (out_channel, kernel_size, stride) in enumerate(zip(conv_out_channel,
                                                                     conv_kernel_size,
                                                                     conv_stride)):
            if stride == 2:     # We want to double the number
                pad = int(kernel_size/2 - 1)
            elif stride == 1:   # We want to keep the number unchanged
                pad = int((kernel_size - 1)/2)
            else:
                Exception("Now only support stride = 1 or 2, contact Ben")

            self.convs.append(nn.ConvTranspose1d(in_channel, out_channel, kernel_size,
                                                 stride=stride, padding=pad))  # To make sure L_out double each time
            in_channel = out_channel  # Update the out_channel
        # If there are upconvolutions, do the convolution back to single channel
        if len(self.convs):
            self.convs.append(
                nn.Conv1d(in_channel, out_channels=1, kernel_size=1, stride=1, padding=0))

    def forward(self, G):
        """
        The forward function which defines how the network is connected
        :param G: The input geometry (Since this is a forward network)
        :return: S: The 300 dimension spectra
        """
        out = G                                                         # initialize the out
        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears, self.bn_linears)):
            if ind != len(self.linears) - 1:
                # ReLU + BN + Linear
                out = F.relu(bn(fc(out)))
            else:
                # For last layer, no activation function
                out = fc(out)

        # Add 1 dimension to get N,L_in, H
        out = out.unsqueeze(1)
        # For the conv part
        for ind, conv in enumerate(self.convs):
            out = conv(out)
        S = out.squeeze(1)
        return S


this_file_path = pathlib.Path(__file__).parent.absolute()
_pretrained_model_path = os.path.join(this_file_path, "mm_pts")


class MetaMaterialEnv(ForwardEnv):

    def __init__(self, model_path=_pretrained_model_path, device="cuda:0"):
        self.input_shape = (8,)
        self.output_shape = (300,)
        self.name = "metamaterial"
        self.model_path = model_path
        self.device = device
        self.ckpt_paths = [os.path.join(
            self.model_path, "mm{}.pth".format(i)) for i in range(1, 6)]
        self.models = [MMNetwork() for i in range(5)]
        for model, ckpt_path in zip(self.models, self.ckpt_paths):
            model.load_state_dict(torch.load(ckpt_path))
            model.eval()
        self.models = [model.to(device) for model in self.models]

    def forward(self, x):
        assert len(x.shape) == 2 and x.shape[1] == 8
        x = torch.Tensor(x).to(self.device)
        with torch.no_grad():
            pred_list = [model(x).detach().cpu().numpy()
                         for model in self.models]
        pred = np.mean(np.array(pred_list), axis=0)
        return pred

    def generate_random_x(self,
                          num=8,
                          bounds=((-1, 1.273), (-1, 1.273), (-1, 1.273), (-1, 1.273),
                                  (-1, 1), (-1, 1), (-1, 1), (-1, 1))):
        output = np.zeros([num, 8])
        for i in range(8):
            output[:, i] = np.random.uniform(
                bounds[i][0], bounds[i][1], size=num)
        return output

    def visualize(self, x, save_path):
        assert len(x.shape) == 2 and x.shape[1] == 8 and x.shape[0] == 1
        pred = self.forward(x)
        pred = np.reshape(pred, (-1))
        plt.plot(pred)
        plt.savefig(save_path)
