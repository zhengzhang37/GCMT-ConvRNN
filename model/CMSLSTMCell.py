import torch.nn as nn
import torch
from torch.autograd import Variable
from  model.SE_Block import *
from torch.nn import Module, Sequential, Conv2d

class CMSLSTM_cell(Module):
    def __init__(self, 
                 in_channel, 
                 num_hidden, 
                 width, 
                 filter_size, 
                 stride,
                 layer_norm, 
                 ce_iterations=2):
        super(CMSLSTM_cell, self).__init__()

        # attention module
        self.ceiter = ce_iterations
        self.SEBlock = SE_Block(num_hidden, width)
        self.attentions = None
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 3, width, width])
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, width, width])
        )
        self.convQ = Sequential(
            nn.Conv2d(num_hidden, in_channel, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            nn.LayerNorm([in_channel, width, width])
        )

        self.convR = Sequential(
            Conv2d(in_channel, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            nn.LayerNorm([num_hidden, width, width])
        )
        self.norm_cell = nn.LayerNorm([num_hidden, width, width])
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)
    def CEBlock(self, xt, ht):
        for i in range(1, self.ceiter + 1):
            if i % 2 == 0:
                ht = (2 * torch.sigmoid(self.convR(xt))) * ht
            else:
                xt = (2 * torch.sigmoid(self.convQ(ht))) * xt

        return xt, ht
        
    def forward(self, x_t, h_t, c_t):
        x_t, h_t = self.CEBlock(x_t, h_t)
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)
        o_t = torch.sigmoid(o_x + o_h)
        c_new = f_t * c_t + i_t * g_t
        c_new = self.norm_cell(c_new)
        h_new = o_t * torch.tanh(c_new)
        next_h, next_c, self.attentions = self.SEBlock(h_new, c_new)
        return next_h, next_c