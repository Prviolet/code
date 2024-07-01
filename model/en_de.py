# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
from model.blocks.hermite import HermiteConvolution
from model.blocks.SC import ConvMod, FuseInfo


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class Encoder(nn.Module):
    def __init__(self, filters, n_channels=1):
        super().__init__()
        self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=n_channels, ch_out=filters[0])
        self.Conv2 = conv_block(ch_in=filters[0], ch_out=filters[1])
        self.Conv3 = conv_block(ch_in=filters[1], ch_out=filters[2])
        self.Conv4 = conv_block(ch_in=filters[2], ch_out=filters[3])
        self.Conv5 = conv_block(ch_in=filters[3], ch_out=filters[4])

    def forward(self,x):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        return x1, x2, x3, x4, x5


class ReconstractionDecoder(nn.Module):
    def __init__(self, filters, n_classes=1, hermite_order=16, hermite_k=31):
        super().__init__()
        self.Up5 = up_conv(ch_in=filters[4], ch_out=filters[3])
        self.ConvMod5 = FuseInfo(dim=filters[3])
        self.Up_conv5 = conv_block(ch_in=filters[4], ch_out=filters[3])

        self.Up4 = up_conv(ch_in=filters[3], ch_out=filters[2])
        self.ConvMod4 = FuseInfo(dim=filters[2])
        self.Up_conv4 = conv_block(ch_in=filters[3], ch_out=filters[2])

        self.Up3 = up_conv(ch_in=filters[2], ch_out=filters[1])
        self.ConvMod3 = FuseInfo(dim=filters[1])
        self.Up_conv3 = conv_block(ch_in=filters[2], ch_out=filters[1])

        self.Up2 = up_conv(ch_in=filters[1], ch_out=filters[0])
        self.ConvMod2 = FuseInfo(dim=filters[0])
        self.Up_conv2 = conv_block(ch_in=filters[1], ch_out=filters[0])

        self.HermiteConv = HermiteConvolution(in_channel=filters[0], order_channel=hermite_order, out_channel=n_classes,
                                              k=hermite_k)
        # self.Conv1x1 = nn.Conv1d(filters[0], n_classes, 1)


    def forward(self,x1, x2, x3, x4, x5):
        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.ConvMod5(d5, x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.ConvMod4(d4, x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.ConvMod3(d3, x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.ConvMod2(d2, x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1, coe = self.HermiteConv(d2)
        # d1 = self.Conv1x1(d2)

        return d1, coe
        # return d1



class DelineationDecoder(nn.Module):
    def __init__(self, filters, n_classes=3):
        super().__init__()
        self.Up5 = up_conv(ch_in=filters[4], ch_out=filters[3])
        self.ConvMod5 = FuseInfo(dim=filters[3])
        self.Up_conv5 = conv_block(ch_in=filters[4], ch_out=filters[3])

        self.Up4 = up_conv(ch_in=filters[3], ch_out=filters[2])
        self.ConvMod4 = FuseInfo(dim=filters[2])
        self.Up_conv4 = conv_block(ch_in=filters[3], ch_out=filters[2])

        self.Up3 = up_conv(ch_in=filters[2], ch_out=filters[1])
        self.ConvMod3 = FuseInfo(dim=filters[1])
        self.Up_conv3 = conv_block(ch_in=filters[2], ch_out=filters[1])

        self.Up2 = up_conv(ch_in=filters[1], ch_out=filters[0])
        self.ConvMod2 = FuseInfo(dim=filters[0])
        self.Up_conv2 = conv_block(ch_in=filters[1], ch_out=filters[0])
        self.Conv1x1 = nn.Conv1d(filters[0], n_classes, 1)


    def forward(self,x1, x2, x3, x4, x5, coe):
        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.ConvMod5(d5, x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.ConvMod4(d4, x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.ConvMod3(d3, x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.ConvMod2(d2, x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d2 = d2 + coe

        d1 = self.Conv1x1(d2)

        return torch.sigmoid(d1)





class Decoder(nn.Module):
    def __init__(self, filters, n_classes=3):
        super().__init__()
        self.Up5 = up_conv(ch_in=filters[4], ch_out=filters[3])
        self.Up_conv5 = conv_block(ch_in=filters[4], ch_out=filters[3])

        self.Up4 = up_conv(ch_in=filters[3], ch_out=filters[2])
        self.Up_conv4 = conv_block(ch_in=filters[3], ch_out=filters[2])

        self.Up3 = up_conv(ch_in=filters[2], ch_out=filters[1])
        self.Up_conv3 = conv_block(ch_in=filters[2], ch_out=filters[1])

        self.Up2 = up_conv(ch_in=filters[1], ch_out=filters[0])
        self.Up_conv2 = conv_block(ch_in=filters[1], ch_out=filters[0])
        self.Conv1x1 = nn.Conv1d(filters[0], n_classes, 1)


    def forward(self,x1, x2, x3, x4, x5):
        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv1x1(d2)

        return torch.sigmoid(d1)


class ReconstractionNormalDecoder(nn.Module):
    def __init__(self, filters, n_classes=1, hermite_order=8, hermite_k=31):
        super().__init__()
        self.Up5 = up_conv(ch_in=filters[4], ch_out=filters[3])
        self.Up_conv5 = conv_block(ch_in=filters[4], ch_out=filters[3])

        self.Up4 = up_conv(ch_in=filters[3], ch_out=filters[2])
        self.Up_conv4 = conv_block(ch_in=filters[3], ch_out=filters[2])

        self.Up3 = up_conv(ch_in=filters[2], ch_out=filters[1])
        self.Up_conv3 = conv_block(ch_in=filters[2], ch_out=filters[1])

        self.Up2 = up_conv(ch_in=filters[1], ch_out=filters[0])
        self.Up_conv2 = conv_block(ch_in=filters[1], ch_out=filters[0])

        self.HermiteConv = HermiteConvolution(in_channel=filters[0], order_channel=hermite_order, out_channel=n_classes,
                                              k=hermite_k)
        # self.Conv1x1 = nn.Conv1d(filters[0], n_classes, 1)


    def forward(self,x1, x2, x3, x4, x5):
        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1, coe = self.HermiteConv(d2)
        # d1 = self.Conv1x1(d2)

        return d1, coe
        # return d1

