# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn

from model.blocks.SC import FuseInfo
from model.en_de import Encoder, ReconstractionDecoder, DelineationDecoder


class Muti_UNet(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        filters = np.array([16, 32, 64, 128, 256])
        # filters = np.array([64, 128, 256, 512, 1024])
        filters = filters // scale_factor
        self.reconstructionEncoder = Encoder(filters)
        self.delineationEncoder = Encoder(filters)
        self.reconstructionDecoder = ReconstractionDecoder(filters, hermite_order=8, hermite_k=31)
        self.delineationDecoder = DelineationDecoder(filters)
        # self.GPA = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Sequential(nn.Linear(128, 3), nn.ReLU())
        self.fuse = FuseInfo(dim=filters[4])


    def forward(self,x):
        x1, x2, x3, x4, x5 = self.reconstructionEncoder(x)
        g1, g2, g3, g4, g5 = self.delineationEncoder(x)

        g5 = self.fuse(x5, g5)

        rec, coe = self.reconstructionDecoder(x1, x2, x3, x4, x5)
        loc = self.delineationDecoder(g1, g2, g3, g4, g5, coe)
        # return rec, loc, coe
        return loc


if __name__ == '__main__':
    net = Muti_UNet()
    print(net)
    x = torch.rand(1, 1, 512)
    out = net(x)
    for i in range(0, 3):
        print(out[i].shape)
    # print(out[1])
