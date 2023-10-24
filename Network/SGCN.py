import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from Network.TGCN import ConvTemporalGraphical
from Network.Graph import Graph

class SGCN(nn.Module):
    r"""Spatial temporal graph convolutional networks.
    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, out_channels, graph_args,
                 edge_importance_weighting, gcn_layer, graph_mode, **kwargs):
        super().__init__()



        # load graph
        self.graph = Graph(mode = graph_mode, **graph_args)

        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False) # in cholec 1 8 8

        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        # create the stgcn to learn
        self.channel_n_1 = out_channels

        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, self.channel_n_1, kernel_size, 1, residual=True, **kwargs0),
            st_gcn(self.channel_n_1, self.channel_n_1, kernel_size, 1, residual=True, **kwargs0),
        ))


        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)






    def forward(self, x):

        x = x.unsqueeze(-1)

        N, T, V, C, M = x.size()
        x = x.permute(0, 3, 1, 2, 4).contiguous() # N C T V M

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)



        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):

            x, _ = gcn(x, self.A * importance )


        # fcn
        N, C, T, V = x.size()

        out_stem = x.permute(0, 2, 3, 1).contiguous() # N, T, V, C
        feature = out_stem.clone()

        return feature

class attention_m(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.m = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid(),)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 2, 3, 4, 1).contiguous() # N, T, V, M, C
        x = self.m(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous() # N, C, T, V, M

        return x

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x, A):

        x, A = self.gcn(x, A)

        return self.relu(x), A

