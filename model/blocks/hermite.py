# -*- coding: utf-8 -*-
"""
# @Time : 2024/2/11 11:49
# @Author : ganli
# @File : hermite.py
# @Project : GraduationProjects
"""
from functools import lru_cache
import numpy as np
import torch
from torch import nn


class Hermite():
    def __init__(self, max_degree):
        self.pi = np.pi
        # 因为要算阶乘，提前指定最大阶数，将所有结果提前计算并存储在 self.factorial_cache 里面
        self.factorial_cache = self.precompute_factorials(max_degree)
    @staticmethod
    def cal_factorial(n):
        if n == 0:
            return 1
        else:
            return n * Hermite.cal_factorial(n - 1)
    def precompute_factorials(self, max_degree):
        factorials = [self.cal_factorial(i) for i in range(max_degree + 1)]
        return np.array(factorials)
    @lru_cache(maxsize=100)
    def H_polynomial(self, t, n):
        if n == 0:
            return torch.ones_like(t)
        elif n == 1:
            return 2 * t
        else:
            h_prev = torch.ones_like(t)
            h_curr = 2 * t
            for i in range(2, n + 1):
                h_next = 2 * t * h_curr - 2 * (i - 1) * h_prev
                h_prev, h_curr = h_curr, h_next
            return h_curr

    def phi(self, t, i):
        factorial = self.factorial_cache[i]
        h_polynomial = self.H_polynomial(t, i)
        exp_term = torch.exp((-torch.pow(t, 2)) / 2)
        const = (2 ** (i / 2)) / np.sqrt(self.pi * factorial)
        f = exp_term * const * h_polynomial
        norm = torch.norm(f, dim=1, keepdim=True)
        f = torch.div(f, torch.clamp(norm, min=1e-12))
        return f

class Adaptive_r(nn.Module):
    def __init__(self,in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channel, in_channel, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv1d(in_channel, in_channel, kernel_size=3, dilation=2, padding=2)
        self.bn1 = nn.BatchNorm1d(in_channel)
        self.bn2 = nn.BatchNorm1d(in_channel)
        self.bn3 = nn.BatchNorm1d(in_channel)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.conv_1x1 = nn.Conv1d(in_channel, in_channel, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channel, out_channel)
    def forward(self,x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)

        x3 = x1 + x2
        x3 = self.conv_1x1(x3)
        x3 = self.bn3(x3)
        x3 = self.relu3(x3)
        x3 = self.global_pool(x3).flatten(1)
        r = self.fc(x3)
        return r

class HermiteConvolution(nn.Module):
    def __init__(self, in_channel, order_channel, out_channel, k, requires_grad=False):
        super(HermiteConvolution, self).__init__()
        self.order = order_channel
        self.requires_grad = requires_grad
        self.k = k
        # Define Hermite kernel
        self.hermite = Hermite(order_channel)
        # Convolution layers
        self.conv1 = nn.Conv1d(in_channel, order_channel, kernel_size=1)
        self.conv2 = nn.Conv1d(order_channel, out_channel, kernel_size=k, padding=k // 2)
        self.width = Adaptive_r(order_channel, 1)
        self.center = Adaptive_r(order_channel, 1)

    def forward(self, x):
        B, _, _ = x.shape
        # Apply first convolution
        x1 = self.conv1(x)
        # Generate Hermite kernel
        width = self.width(x1)
        center = self.center(x1)
        center = torch.clamp(center, min=1, max=128)
        T = torch.vstack([torch.arange(self.k) for _ in range(B)]).to(device=x.device)
        t = width * (T - center)
        kernel = [self.hermite.phi(t, i).unsqueeze(1) for i in range(self.order-1)]
        kernel.append(1 / (1+torch.exp(-2 * t)).unsqueeze(1))
        kernel = torch.cat(kernel, dim=1)

        # Apply second convolution with adaptive Hermite kernel
        output = []
        for j in range(B):
            k = kernel[j, :, :].unsqueeze(0)
            self.conv2.weight.data = k
            self.conv2.weight.requires_grad_(self.requires_grad)
            o = self.conv2(x1[j, :, :].unsqueeze(0))
            output.append(o)
        output = torch.cat(output, dim=0)
        return output, x1



if __name__ == '__main__':
    net = HermiteConvolution(16, 8, 1, k=51)
    x = torch.rand(64, 16, 128)
    out = net(x)
