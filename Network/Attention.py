import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy


class confidence_attention(nn.Module):

    def __init__(self, f_dm, kernel_size = (9, 9),  dilation = 1, causal_conv = True):
        super().__init__()

        self.sp_avg_pool = nn.AvgPool2d((1, f_dm))
        self.sp_max_pool = nn.MaxPool2d((1, f_dm))
        self.sp_attention = nn.Sequential(DilatedResidualLayer2d(dilation =  dilation,
                                     in_channels = 1,
                                     out_channels = 1,
                                     causal_conv=causal_conv,
                                     kernel_size = kernel_size),
                                     nn.Sigmoid(),)
    def forward(self, x):
        B, T, V, C = x.size()

        # sp attention
        # avg
        confidence_index = self.sp_avg_pool(x)
        confidence_index  = confidence_index.permute(0, 3, 1, 2).contiguous()  # to b, c, t, v
        avg_confidence_index = self.sp_attention(confidence_index)

        # max
        confidence_index = self.sp_max_pool(x)
        confidence_index  = confidence_index.permute(0, 3, 1, 2).contiguous()  # to b, c, t, v
        max_confidence_index = self.sp_attention(confidence_index)

        # product
        confidence_index = avg_confidence_index + max_confidence_index
        x = x.permute(0, 3, 1, 2).contiguous()  # to b, c, t, v
        x = torch.mul(confidence_index, x)
        x = x.permute(0, 2, 3, 1).contiguous()  # to b, t, v, c

        return x




class DilatedResidualLayer(nn.Module):
    def __init__(self,
                 dilation,
                 in_channels,
                 out_channels,
                 causal_conv=False,
                 kernel_size=3):
        super(DilatedResidualLayer, self).__init__()
        self.causal_conv = causal_conv
        self.dilation = dilation
        self.kernel_size = kernel_size
        if self.causal_conv:
            self.conv_dilated = nn.Conv1d(in_channels,
                                          out_channels,
                                          kernel_size,
                                          padding=(dilation *
                                                   (kernel_size - 1)),
                                          dilation=dilation)
        else:
            self.conv_dilated = nn.Conv1d(in_channels,
                                          out_channels,
                                          kernel_size,
                                          padding=dilation,
                                          dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        if self.causal_conv:
            out = out[:, :, :-(self.dilation * (self.kernel_size-1))]
        out = self.conv_1x1(out)
        return (x + out)


class DilatedResidualLayer2d(nn.Module):
    def __init__(self,
                 dilation,
                 in_channels,
                 out_channels,
                 causal_conv=False,
                 kernel_size=(3, 1)):
        super(DilatedResidualLayer2d, self).__init__()
        self.causal_conv = causal_conv
        self.dilation = dilation
        self.kernel_size = kernel_size
        if self.causal_conv:
            self.conv_dilated = nn.Conv2d(in_channels,
                                          out_channels,
                                          self.kernel_size,
                                          padding=(dilation * (kernel_size[0] - 1),
                                                   dilation * (kernel_size[1] - 1)//2),
                                          dilation=(dilation,dilation))
        else:
            self.conv_dilated = nn.Conv1d(in_channels,
                                          out_channels,
                                          kernel_size,
                                          padding=(dilation * (kernel_size[0] - 1)//2,
                                                   dilation * (kernel_size[1] - 1)//2),
                                          dilation=(dilation,dilation))
        self.conv_1x1 = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))

        if self.causal_conv and (self.dilation * (self.kernel_size[0]-1)) > 0:
            out = out[:, :, :-(self.dilation * (self.kernel_size[0]-1)),:]

        if self.causal_conv and (self.dilation * (self.kernel_size[0]-1)) == 0:
            out = out

        out = self.conv_1x1(out)
        return (x + out)



class trans(nn.Module):

    def __init__(self, d_k):
        super().__init__()

        self.d_k = d_k

        self.q = nn.Linear(d_k, d_k, bias=False)
        self.k = nn.Linear(d_k, d_k, bias=False)
        self.v = nn.Linear(d_k, d_k, bias=False)

        self.LN = nn.LayerNorm(self.d_k)

    def forward(self, x):

        residual = x

        q = self.q(x) # b, t, f
        k = self.k(x)
        v = self.v(x)

        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k) # b, t, t

        attn = nn.Softmax(dim=-1)(scores) # b, t, t
        context = torch.matmul(attn, v) # b, t, f

        x = self.LN(residual + context)

        return x



class embed(nn.Module):

    def __init__(self, n_channel, rep_channel):
        super().__init__()

        in_channel = n_channel // 2 + 1
        out_channel = rep_channel // 2

        self.loc_conv = nn.Conv2d(in_channel, out_channel, kernel_size = 1)
        self.size_conv = nn.Conv2d(in_channel, out_channel, kernel_size= 1)


    def forward(self, x):

        x_confidence = x[:,4:,:,:]

        x_loc = torch.cat((x[:,0:2,:,:] , x_confidence), dim= 1)
        x_size = torch.cat((x[:,2:4,:,:] , x_confidence), dim= 1)

        x_loc = self.loc_conv(x_loc)
        x_size = self.size_conv(x_size)

        x = torch.cat((x_loc , x_size), dim= 1)

        return x

class smoothing(nn.Module):

    def __init__(self, rep_channel, kernel_size = (9, 1)):
        super().__init__()
        padding = int((kernel_size[0] - 1) / 2)
        padding = (padding, 0)
        self.conv2d = nn.Conv2d(rep_channel, rep_channel, kernel_size, padding = padding, groups = rep_channel)

    def forward(self, x):

        x = self.conv2d(x)

        return x
