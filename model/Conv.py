import torch.nn as nn
import torch


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch,output_padding=1):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(

            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=True,output_padding = output_padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Conv2D(nn.Module):
    def __init__(self,
                 cell_param
                 ):
        super(Conv2D, self).__init__()
        self.cell_param = cell_param
        self.net = nn.Sequential()
        self.act_conv2d = nn.Conv2d(
            in_channels=self.cell_param['in_channel'],
            out_channels=self.cell_param['out_channel'],
            kernel_size=self.cell_param['kernel_size'],
            stride=self.cell_param['stride'],
            padding=self.cell_param['padding']
        )
        torch.nn.init.xavier_uniform_(self.act_conv2d.weight)
        torch.nn.init.constant_(self.act_conv2d.bias, 0)
        self.net.add_module('conv', self.act_conv2d)
        if self.cell_param['activate'] == None:
            pass
        elif self.cell_param['activate'] == 'relu':
            self.net.add_module('activate', nn.ReLU())
        elif self.cell_param['activate'] == 'tanh':
            self.net.add_module('activate', nn.Tanh())

    def forward(self, input):

        output=self.net(input)

        return output


class DeConv2D(nn.Module):
    def __init__(self,
                 cell_param
                 ):
        super(DeConv2D, self).__init__()
        self.cell_param = cell_param
        self.net = nn.Sequential()
        self.act_de_conv2d = nn.ConvTranspose2d(
            in_channels=self.cell_param['in_channel'],
            out_channels=self.cell_param['out_channel'],
            kernel_size=self.cell_param['kernel_size'],
            stride=self.cell_param['stride'],
            padding=self.cell_param['padding'],
            output_padding=self.cell_param['output_padding']
        )
        torch.nn.init.xavier_uniform_(self.act_de_conv2d.weight)
        torch.nn.init.constant_(self.act_de_conv2d.bias, 0)
        self.net.add_module('de_conv', self.act_de_conv2d)
        if self.cell_param['activate'] == None:
            pass
        elif self.cell_param['activate'] == 'relu':
            self.net.add_module('activate',nn.ReLU())
        elif self.cell_param['activate'] == 'tanh':
            self.net.add_module('activate', nn.Tanh())

    def forward(self, input):

        output = self.net(input)

        return output