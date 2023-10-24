# implementation adapted from:
# https://github.com/yabufarha/ms-tcn/blob/master/model.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from Network.Attention import DilatedResidualLayer2d



class SingleStageModel(nn.Module):
    def __init__(self,
                 num_layers,
                 num_f_maps,
                 dim,
                 num_f_maps2,
                 causal_conv=False):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv2d(dim, num_f_maps, 1)
        
        self.layers_up = nn.ModuleList([
            copy.deepcopy(
                DilatedResidualLayer2d(2**i,
                                     num_f_maps,
                                     num_f_maps,
                                     causal_conv=causal_conv))
            for i in range(num_layers)
        ])
        self.layers_down = nn.ModuleList([
            copy.deepcopy(
                DilatedResidualLayer2d(2**(num_layers-i),
                                     num_f_maps,
                                     num_f_maps,
                                     causal_conv=causal_conv))
            for i in range(num_layers)
        ])
        self.conv_out_classes = nn.Conv1d(num_f_maps, num_f_maps2, 1)
        self.fusion = nn.Sequential(nn.Linear(num_layers, 1),
                                          nn.ReLU(inplace=True),)

    def forward(self, x):
        out = self.conv_1x1(x)
        skip_res = []
        # up-sample
        for l in range(len(self.layers_up)):
            out = self.layers_up[l](out)
            skip_res.append(out)
        
        return out


