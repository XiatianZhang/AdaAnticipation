import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
from Network.MSTCN import SingleStageModel
from Network.Attention import confidence_attention, embed
from Network.Policy import Policy_GCN


class GCNProgressNet(nn.Module):

    def __init__(self,
                 num_stages_t=2,
                 num_layers_t=9,
                 num_f_maps_t=16,

                 num_stages_p=2,
                 num_layers_p=8,
                 num_f_maps_p=16,

                 horrizon_n = 3,

                 dim=2048,
                 causal_conv =True,
                 in_channels=5,
                 rep_channels = 2,
                 
                 num_exp = 2,
                 num_classes_phase=11,
                 num_class=5,
                 node_n = 8,

                 graph_args={'max_hop': 1, 'strategy': 'uniform', 'dilation': 1},
                 edge_importance_weighting=False,
                 fc_output_channel=256,
                 feature_extractor_pretrain=False,
                 stream = 'Graph',
                 GCN_mode = 'TCNGCN',
                 FPS = 2.5,
                 oFps = 10,
                 FPM = 25*60,
                 graph_list = ['10010000', '11010000', '11000000', '10000000', '00000000', '01000000', '11000001', '10000100', '10100000', '10000010'],
                 horizon_list=[2, 3, 5],
                 only_RSD = False,
                 t_k = 9,
                 s_k = 9,
                 q = 6000,
                 dropout_r = 0.5,
                 **kwargs,
                 ):

        super(GCNProgressNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.stream = stream

        self.only_RSD = only_RSD

        self.FPM = FPM

        if GCN_mode == 'TCNGCN':

            self.attention = confidence_attention(in_channels, (t_k, s_k))

            self.embed = embed(in_channels, rep_channels)

            self.tool_grpah_modes = graph_list


            self.tool_gcn = Policy_GCN(num_layers_t,rep_channels, num_f_maps_t, graph_args, edge_importance_weighting,
                                        graph_mode = graph_list , gcn_layer = 0,  node_n = node_n)
            
            self.tool_encode = nn.AdaptiveAvgPool2d((None, 1))


            self.TCN_tool = nn.ModuleList([
            copy.deepcopy(
                SingleStageModel(num_layers_t,num_f_maps_t,num_f_maps_t,num_f_maps_t,
                                 causal_conv=causal_conv))
            for s in range(num_stages_t - 1)])

            if self.only_RSD == False:

                self.phase_grpah_modes = graph_list
                self.phase_gcn = Policy_GCN(num_layers_p,rep_channels, num_f_maps_p, graph_args, edge_importance_weighting,
                                            graph_mode = graph_list , gcn_layer = 0,  node_n = node_n)
                self.phase_encode = nn.AdaptiveAvgPool2d((None, 1))
                self.TCN_phase = nn.ModuleList([
                copy.deepcopy(
                    SingleStageModel(num_layers_p,num_f_maps_p,num_f_maps_p,num_f_maps_p,
                                     causal_conv=causal_conv))
                for s in range(num_stages_p - 1)])


        self.GCN_mode = GCN_mode




        self.fc_h_channel = fc_output_channel # the channel before FC layers

        self.FPS = FPS
        self.oFPS = oFps
        self.step =self.oFPS/self.FPS

        self.fc_input_channel = num_f_maps_t
        if self.only_RSD == False:

            self.fc_input_channel = num_f_maps_t + num_f_maps_p

            self.fc_stage_RSD = nn.Sequential(nn.Linear(self.fc_input_channel,
                                                        self.fc_h_channel),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(dropout_r),
                                              nn.Linear(self.fc_h_channel, num_classes_phase),
                                              nn.ReLU(inplace=True),)

            self.fc_tool_stage_RSD = nn.Sequential(nn.Linear(self.fc_input_channel,self.fc_h_channel),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(dropout_r),
                                              nn.Linear(self.fc_h_channel, num_class),
                                              nn.ReLU(inplace=True),)
        else:
            self.fc_RSD = nn.Sequential(nn.Linear(self.fc_input_channel, self.fc_h_channel),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(dropout_r),
                                        nn.Linear(self.fc_h_channel, 1),
                                        nn.ReLU(inplace=True))


            self.fc_phase = nn.Sequential(nn.Linear(self.fc_input_channel,self.fc_h_channel),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.fc_h_channel, num_classes_phase),
                                            nn.ReLU(inplace=True),)
            
            self.fc_exp = nn.Sequential(nn.Linear(self.fc_input_channel,self.fc_h_channel),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.fc_h_channel, num_exp),
                                            nn.ReLU(inplace=True),)

        self.horizon_list = horizon_list
        self.horizon_max = horizon_list[-1]

        self.phase_class = num_classes_phase
        self.tool_class = num_class


    def forward(self, x, bb):

        N, T, V, C = bb.size()

        bb = self.attention(bb)



        bb = bb.permute(0, 3, 1, 2).contiguous() # from(B, T, V, C) to (B, C, T, V)
        bb = self.embed(bb)
        bb = bb.permute(0, 2, 3, 1).contiguous() # from (B, C, T, V) to (B, T, V, C)



        bb_feature = bb

        out_stem_tool = bb
        out_stem_phase = bb



        out_stem_tool, out_stem_tool_policy = self.tool_gcn(out_stem_tool)
        out_stem_tool = out_stem_tool.permute(0, 3, 1, 2).contiguous()  # from(B,T,V,C) to (B,C,T,V)

        for s in self.TCN_tool:
            out_stem_tool = s(out_stem_tool)
        out_stem_tool = out_stem_tool.permute(0, 2, 1, 3).contiguous()  # from(B, C, T, V) to (B, T, C, V)
        out_stem_tool = self.tool_encode(out_stem_tool).squeeze(-1) # (B, T, C, V) to (B, T, C)

        out_stem_phase_policy = None
        if self.only_RSD == False:
            out_stem_phase, out_stem_phase_policy = self.phase_gcn(out_stem_phase)
            out_stem_phase = out_stem_phase.permute(0, 3, 1, 2).contiguous()  # from(B,T,V,C) to (B,C,T,V)
            for s in self.TCN_phase:
                out_stem_phase  = s(out_stem_phase)
            out_stem_phase = out_stem_phase.permute(0, 2, 1, 3).contiguous()  # from(B, C, T, V) to (B, T, C, V)
            out_stem_phase = self.phase_encode(out_stem_phase).squeeze(-1) # (B, T, C, V) to (B, T, C)

            out_stem = torch.cat((out_stem_tool, out_stem_phase), dim=-1)

            stage_RSD = self.fc_stage_RSD(out_stem)
            tool_stage_RSD = self.fc_tool_stage_RSD(out_stem)

            stage_RSD = stage_RSD * self.FPM
            tool_stage_RSD = tool_stage_RSD * self.FPM

            RSD = None
            phase = None
            exp = None

        else:
            out_stem = out_stem_tool
            stage_RSD = 0
            tool_stage_RSD = 0


            N, T, C = out_stem.size()
            # fc to the cat output
            feature = out_stem

            RSD = self.fc_RSD(out_stem)
            RSD = RSD * self.FPM
            phase = self.fc_phase(out_stem)
            exp = self.fc_exp(out_stem)
        

        


        return bb_feature, stage_RSD, tool_stage_RSD, RSD, phase, exp, out_stem_tool_policy, out_stem_phase_policy
