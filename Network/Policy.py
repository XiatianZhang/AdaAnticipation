import torch
import torch.nn as nn
import torch.nn.functional as F
from Network.MSTCN import SingleStageModel

from Network.SGCN import SGCN


class Policy(nn.Module):

    def __init__(self, num_layers=8, num_f_maps=16, action_size = 9, target_action_size = 3, causal_conv=True, tau = 1e-5):
        super(Policy, self).__init__()
        
        self.Action_Embed = nn.Sequential(nn.Linear(num_f_maps, action_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(action_size, action_size),
                                        nn.ReLU(inplace=True),)

        self.TCN = SingleStageModel(num_layers,action_size,action_size,action_size,
                                     causal_conv=causal_conv)
        
        self.node_encode = nn.AdaptiveAvgPool2d((None, 1))

        self.tau = tau

        self.target_action_size = target_action_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        
        x = self.Action_Embed(x)

        x = x.permute(0, 3, 1, 2).contiguous() # from B, T, V, A To B, A, T, V
        x = self.TCN(x)
        x = self.node_encode(x) # from B, A, T, V to B, A, T, 1
        x = x.squeeze(-1) # from B, A, T, 1 to B, A, T        
        x = x.permute(0, 2, 1).contiguous() # from B, A, T to B, T, A

        # Reshape x from (B, T, A) to (B, T, A, A) 
        x = x.unsqueeze(2)
        x = x.repeat(1, 1, x.shape[-1], 1)

        x = self.gumbel_softmax_sinkhorn(x, self.tau)

        x = x[:, :, :, :self.target_action_size] # from B, T, A, A to B, T, A, T_A
        
        return x

    def gumbel_softmax_sinkhorn(self, logits, temperature):
        B, T, A, _ = logits.shape
        # Gumbel-Softmax trick
        if self.training:
            gumbel_noise = torch.zeros_like(logits).uniform_(0, 1)
            gumbel_noise = -torch.log(-torch.log(gumbel_noise + 1e-8) + 1e-8)
        else:
            gumbel_noise =  torch.zeros((B, int(1e4), A, A)).to(self.device).uniform_(0, 1)
            gumbel_noise = gumbel_noise[:, :T, :, :]
            gumbel_noise = -torch.log(-torch.log(gumbel_noise + 1e-8) + 1e-8)
        
        gumbel_logits = logits + gumbel_noise
        y = self.sinkhorn(gumbel_logits)  # using the provided Sinkhorn function
        
        return y
    
    def sinkhorn(self, log_alpha, n_iters=10, t= 1e-6):
        n = log_alpha.shape[-1]
        log_alpha = log_alpha
        for _ in range(n_iters):
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=-1, keepdim=True))
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=-2, keepdim=True))
        return F.softmax(log_alpha/t, dim=-1)


class GCN_List(nn.Module):

    def __init__(self, in_channels, num_f_maps, graph_args, edge_importance_weighting, graph_mode , gcn_layer,  node_n):
        super(GCN_List, self).__init__()
        
        self.action_list = graph_mode
        self.GCN_List = nn.ModuleList([
            SGCN(in_channels, num_f_maps, graph_args, edge_importance_weighting, graph_mode = graph_mode[i], gcn_layer = gcn_layer)
              for i in range(len(graph_mode))])    

        
    
    def forward(self, x):

        x = [self.GCN_List[i](x) for i in range(len(self.GCN_List))]
        x = torch.stack(x, dim = -1)
        return x
    
class Policy_GCN(nn.Module):

    def __init__(self, num_layers, in_channels, num_f_maps, graph_args, edge_importance_weighting, graph_mode , gcn_layer, node_n, k = 3):
        super().__init__()

        self.action_list = graph_mode
        self.action_size = len(graph_mode)

        self.Policy = Policy(num_layers, in_channels, self.action_size, causal_conv=True, target_action_size= k)
        
        self.GCN_List = GCN_List(in_channels, num_f_maps, graph_args, edge_importance_weighting, graph_mode , gcn_layer,  node_n)

        self.channel_attention = nn.Sequential(nn.Linear(k, 2*k),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(2*k, k),
                                        nn.Sigmoid(),)

    def forward(self, x):

        x_policy = self.Policy(x)
        x = self.GCN_List(x)
        x = torch.einsum('ijklm,ijmn->ijkln', x, x_policy)
        x_attention = x.mean(dim = -2, keepdim = True).mean(dim = -3, keepdim = True)
        x = x * self.channel_attention(x_attention)
        x = x.sum(dim = -1)
        return x, x_policy
    