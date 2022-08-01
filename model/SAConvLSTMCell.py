__author__ = 'yunbo'

import torch
import torch.nn as nn


class SAConvLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(SAConvLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.width = width
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )
        # self.conv_sa = nn.Conv2d(num_hidden, num_hidden * 5, kernel_size=1, stride=1, padding=0)
        self.conv_kh = nn.Conv2d(num_hidden, 1, kernel_size=1, stride=1, padding=0)
        self.conv_qh = nn.Conv2d(num_hidden, 1, kernel_size=1, stride=1, padding=0)
        self.conv_km = nn.Conv2d(num_hidden, 1, kernel_size=1, stride=1, padding=0)
        self.conv_vh = nn.Conv2d(num_hidden, num_hidden, kernel_size=1, stride=1, padding=0)
        self.conv_vm = nn.Conv2d(num_hidden, num_hidden, kernel_size=1, stride=1, padding=0)

        self.conv_hm = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)
        self.conv_z = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,groups=num_hidden),
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=1),
            nn.LayerNorm([num_hidden * 3, width, width])
        )
        self.conv_h_new = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,groups=num_hidden),
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=1),
            nn.LayerNorm([num_hidden * 3, width, width])
        )
        # self.conv_zh=nn.Sequential(
        #     nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=1, padding=self.padding),
        #     nn.LayerNorm([num_hidden, width, width])
        # )
        # self.conv_zm=nn.Sequential(
        #     nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=1, padding=self.padding),
        #     nn.LayerNorm([num_hidden, width, width])
        # )

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        o_t = torch.sigmoid(o_x + o_h)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t
        h_new = o_t * torch.tanh(c_new)
        
        # self-attention memory module
        # sah_concat = self.conv_sah(h_new).view(x_t.shape[0] ,self.num_hidden*3, self.width*self.width)  # [8,64*5,16*16]
        # sam_concat = self.conv_sam(m_t).view(x_t.shape[0] ,self.num_hidden*2, self.width*self.width)  # [8,64*5,16*16]
        # q_h, k_h, v_h = torch.split(sah_concat, self.num_hidden, dim=1) # [8,64,16*16]
        # k_m, v_m = torch.split(sam_concat, self.num_hidden, dim=1) # [8,64,16*16]
        q_h = self.conv_qh(h_new).view(x_t.shape[0],1,self.width*self.width)
        k_h = self.conv_kh(h_new).view(x_t.shape[0],1,self.width*self.width)
        v_h = self.conv_vh(h_new).view(x_t.shape[0],self.num_hidden,self.width*self.width)
        k_m = self.conv_km(m_t).view(x_t.shape[0],1,self.width*self.width)
        v_m = self.conv_vm(m_t).view(x_t.shape[0],self.num_hidden,self.width*self.width)



        # similarity scores
        e_h = torch.matmul(q_h.transpose(1,2), k_h)   # [8,256,64]*[8,64,256]->[8,256,256]
        alpha_h = torch.softmax(e_h, dim=2) # [8,256,$256$] softmax
        e_m = torch.matmul(q_h.transpose(1,2), k_m)
        alpha_m = torch.softmax(e_m, dim=2)

        # [8,64,256]*[8,256,256]->[8,64,256]->[8,64,16,16]
        z_h = torch.matmul(v_h, alpha_h.transpose(1,2)).view(x_t.shape[0] ,self.num_hidden, self.width, self.width)
        z_m = torch.matmul(v_m, alpha_m.transpose(1,2)).view(x_t.shape[0] ,self.num_hidden, self.width, self.width)
        # z_h = self.conv_zh(z_h)+h_t
        # z_m = self.conv_zm(z_m)+h_t
        z = self.conv_hm(torch.cat((z_h, z_m), 1))


        # After attention
        z_concat = self.conv_z(z)
        h_new_concat = self.conv_h_new(h_new)

        i_z_prime, g_z_prime, o_z_prime = torch.split(z_concat, self.num_hidden, dim=1)
        i_h_new_prime, g_h_new_prime, o_h_new_prime = torch.split(h_new_concat, self.num_hidden, dim=1)

        i_t_prime = torch.sigmoid(i_z_prime + i_h_new_prime)
        g_t_prime = torch.tanh(g_z_prime + g_h_new_prime)
        o_t_prime = torch.sigmoid(o_z_prime + o_h_new_prime)

        m_new = (torch.ones_like(i_t_prime) - i_t_prime) * m_t + i_t_prime * g_t_prime
        h_new = o_t_prime * m_new

        return h_new, c_new, m_new

