# -*- coding: utf-8 -*-
"""
# @Time : 2024/2/18 21:18
# @Author : ganli
# @File : singleTask.py
# @Project : GraduationProjects
"""
import numpy as np
import torch
from torch import nn
from model.en_de import Encoder, Decoder, ReconstractionDecoder


class UNet(nn.Module):
    def __init__(self, n_classes=3, scale_factor=2):
        super(UNet, self).__init__()
        filters = np.array([16, 32, 64, 128, 256])
        filters = filters // scale_factor
        self.delineationEncoder = Encoder(filters)
        self.delineationDecoder = Decoder(filters)
    def forward(self,x):
        x1, x2, x3, x4, x5 = self.delineationEncoder(x)
        loc = self.delineationDecoder(x1, x2, x3, x4, x5)
        return loc

if __name__ == '__main__':
    net = UNet()
    x = torch.rand(1, 1, 128)
    out1 = net(x)
    print(out1.shape)