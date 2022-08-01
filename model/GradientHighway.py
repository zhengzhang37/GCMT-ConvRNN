import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from model.ConvRNN import *
from torch.nn.parameter import Parameter
from torch.autograd import Variable

class GHU(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride):
        super(GHU, self).__init__()
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 2, width, width]))
        self.conv_z = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 2, width, width]))

    def forward(self, x, z):
        
        z_concat = self.conv_z(z)
        x_concat = self.conv_x(x)
        gates = x_concat + z_concat
        p, u = torch.split(gates, self.num_hidden, dim=1)
        p = torch.tanh(p)
        u = torch.sigmoid(u)
        z_new = u * p + (1 - u) * z

        return z_new





if __name__ == '__main__':
    pass