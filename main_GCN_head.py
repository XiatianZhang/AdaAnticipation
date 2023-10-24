import random
import logging
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader
from Utility.C80_video_dataloader import  SurgeryVideoDataset
from Network.GCN_RSDNet import GCNProgressNet
from Utility.Loss import anticipation_mae, consistency_loss, seq_class_loss
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import wandb

class val_GCN(nn.Module):

    def __init__(self, random_seed,
                 train_dir=r'C:\Users\Xiatian\Desktop\PHD\Anticipation2023\Test\Data\Cholec80Tensor\All_Train',
                 val_dir = r'C:\Users\Xiatian\Desktop\PHD\Anticipation2023\Test\Data\Cholec80Tensor\Test',

                 ):

        super(val_GCN, self).__init__()

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.seed = random_seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




    def val(self, config = None):




        with wandb.init(config=config):


            config = wandb.config

            # set seed
            self.seed = config.seed
            self.setup_seed(self.seed)

            epochs = config.epochs
            val_step = 1
            # load model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(torch.cuda.get_device_name(0))

            model = GCNProgressNet(num_classes_phase=config.num_classes_phase,
                                   stream='Graph',
                                   rep_channels=config.rep_channels,
                                   t_k=config.t_k,
                                   num_stages_t=config.stage_t,
                                   num_layers_t=config.layer_t,
                                   num_f_maps_t=config.out_channel_t,
                                   num_stages_p=config.stage_p,
                                   num_layers_p=config.layer_p,
                                   num_f_maps_p=config.out_channel_p, 
                                   dropout_r= config.dropout,).to(device)
            model_path = 'last_PNet_epoch.pth'
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            # set fps
            FPS = 25
            self.fpm = FPS * 60

            # train setting
            horizon_set = [2, 3, 5]
            weight_set = [config.weight1, config.weight2, config.weight3, config.weight4, 0, 0]
            val_weight_set = [1, 1, 1, 0, 0, 0]
            tool_n = 5
            phase_n = 6
            self.set_loss(horizon_set)
            self.set_temp_weight()
            optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': self.temp_weight, 'lr': config.learning_rate}  # Include tmp_weight as a separate parameter with a specific learning rate
            ], lr=config.learning_rate, weight_decay=config.weight_decay*config.learning_rate)
            loss_RSD = nn.L1Loss()
            loss_stage = seq_class_loss()

            # set epoch
            epochs = epochs
            val_step = val_step

            # set metrics
            RSD_train = []
            stage_RSD_loss_values_train = []
            tool_RSD_loss_values_train = []

            RSD_val = []
            stage_RSD_loss_values_val = []
            tool_RSD_loss_values_val = []

            total_loss_train = []
            total_loss_val = []

            min_RSD_MAE = 100000



            tmp_RSD_val = []
            tmp_tool_stageRSD_loss_values_val = []
            tmp_stageRSD_loss_values_val = []
            tmp_total_loss_val = []

            val_data_loader = DataLoader(SurgeryVideoDataset(self.val_dir), shuffle=False)
            i = 0
            for v, v_data in enumerate(val_data_loader):
                logging.info('epoch {} batch {}'.format(i, v))
                model.eval()
                v_data = v_data

                # define and squeeze the dim from 1 b n n n to b n n n n
                data = v_data['video'].to(device)
                bb = v_data['boxes'].to(device)
                tmp_stage_RSD = v_data['stage_RSD'][:, :, 1:].to(device)
                tmp_tool_RSD = v_data['tool_RSD'].to(device)
                tmp_RSD = v_data['RSD'].to(device)
                phase = v_data['Phase'].to(device)

                with torch.no_grad():
                    # feed data

                    tmp_output_feature, tmp_output_stg_RSD, tmp_output_tool_stg_RSD, tmp_output_RSD, _, _, out_stem_tool_policy, out_stem_phase_policy = model(
                        data, bb)
                    
                
                    # save out_stem_policy
                    pth_name = 'out_stem_policy_' + str(v) + '.pth'
                    torch.save(out_stem_tool_policy, pth_name)
                    # save out_stem_policy
                    pth_name = 'out_stem_phase_policy_' + str(v) + '.pth'
                    torch.save(out_stem_phase_policy, pth_name)
                    # save the output
                    save_dir = 'output_stg_RSD_' + str(v) + '.npy'
                    np.save(save_dir, tmp_output_stg_RSD.cpu().detach().numpy())
                    # save the RSD
                    save_dir = 'stage_RSD_' + str(v) + '.npy'
                    np.save(save_dir, tmp_stage_RSD.cpu().detach().numpy())

                    # save the output
                    save_dir = 'output_tool_stg_RSD_' + str(v) + '.npy'
                    np.save(save_dir, tmp_output_tool_stg_RSD.cpu().detach().numpy())
                    # save the RSD
                    save_dir = 'tool_RSD_' + str(v) + '.npy'
                    np.save(save_dir, tmp_tool_RSD.cpu().detach().numpy())

                    loss_tool_stg_all, tool_MAE, tool_MAE_horizon = self.loss_sum_cal(tmp_output_tool_stg_RSD, tmp_tool_RSD,
                                                            horizon_set,
                                                            tool_n,
                                                            val_weight_set,
                                                        'Tool',
                                                            True,
                                                            inference=True)
                    
                    print('tool_MAE', [i[0] for i in tool_MAE] )

                    # stage rsd
                    loss_stg_rsd_all, stage_MAE, stage_MAE_horizon = self.loss_sum_cal(tmp_output_stg_RSD, tmp_stage_RSD,
                                                            horizon_set,
                                                            phase_n,
                                                            val_weight_set,
                                                        'Phase',
                                                            True,
                                                            inference=True)

                    print('stage_MAE', [i[0] for i in stage_MAE] )


                if v == 0:

                    output_feature = tmp_output_feature

                    output_stg_RSD = tmp_output_stg_RSD
                    output_tool_stg_RSD = tmp_output_tool_stg_RSD


                    stage_RSD = tmp_stage_RSD
                    tool_RSD = tmp_tool_RSD

                    



                else:

                    output_feature = torch.cat((output_feature, tmp_output_feature), dim=1)

                    output_stg_RSD = torch.cat((output_stg_RSD, tmp_output_stg_RSD), dim=1)
                    output_tool_stg_RSD = torch.cat((output_tool_stg_RSD, tmp_output_tool_stg_RSD), dim=1)

                    stage_RSD = torch.cat((stage_RSD, tmp_stage_RSD), dim=1)
                    tool_RSD = torch.cat((tool_RSD, tmp_tool_RSD), dim=1)



            # RSD

            # tool stage rsd
            loss_tool_stg_all, tool_MAE, tool_MAE_horizon = self.loss_sum_cal(output_tool_stg_RSD, tool_RSD,
                                                    horizon_set,
                                                    tool_n,
                                                    val_weight_set,
                                                'Tool',
                                                    True,
                                                    inference=True)
            loss_tool_stg_minute = loss_tool_stg_all
            tmp_tool_stageRSD_loss_values_val.append(loss_tool_stg_minute.item())

            # stage rsd
            loss_stg_rsd_all, stage_MAE, stage_MAE_horizon = self.loss_sum_cal(output_stg_RSD, stage_RSD,
                                                    horizon_set,
                                                    phase_n,
                                                    val_weight_set,
                                                'Phase',
                                                    True,
                                                    inference=True)
            loss_stg_minute = loss_stg_rsd_all
            tmp_stageRSD_loss_values_val.append(loss_stg_minute.item())

            # total loss
            loss_total = loss_tool_stg_minute + loss_stg_minute # + loss_stg #+ loss_sim #+ loss_stg + loss_Elapsed_minute #att_loss+ loss_stg_minute + att_loss
            tmp_total_loss_val.append(loss_total.item())


            # update log
            logging.debug('gpu {}'.format(torch.cuda.memory_reserved(device) / 1024 / 1024 / 1024))
            logging.debug('val loss {}'.format(loss_total.item()))

            if loss_total.item() < min_RSD_MAE:
                logging.info('save model for epoch {}'.format(i))
                min_RSD_MAE = loss_total
                best_i = i

                wandb.log({'best_epoch': i,
                            'best_t_w': tool_MAE_horizon[0].item(),
                            'best_t_i': tool_MAE_horizon[1].item(),
                            'best_t_o': tool_MAE_horizon[2].item(),
                            'best_t_e': tool_MAE_horizon[3].item(),

                            'best_p_w': stage_MAE_horizon[0].item(),
                            'best_p_i': stage_MAE_horizon[1].item(),
                            'best_p_o': stage_MAE_horizon[2].item(),
                            'best_p_e': stage_MAE_horizon[3].item(),

                            'best_loss': np.mean(tmp_total_loss_val)})
                
                wandb.log({
                    'best_t_w_2min': tool_MAE[0][0],
                    'best_t_i_2min': tool_MAE[0][1],
                    'best_t_o_2min': tool_MAE[0][2],
                    'best_t_e_2min': tool_MAE[0][3],
                    'best_t_w_3min': tool_MAE[1][0],
                    'best_t_i_3min': tool_MAE[1][1],
                    'best_t_o_3min': tool_MAE[1][2],
                    'best_t_e_3min': tool_MAE[1][3],
                    'best_t_w_5min': tool_MAE[2][0],
                    'best_t_i_5min': tool_MAE[2][1],
                    'best_t_o_5min': tool_MAE[2][2],
                    'best_t_e_5min': tool_MAE[2][3],
                })

                wandb.log({
                    'best_p_w_2min': stage_MAE[0][0],
                    'best_p_i_2min': stage_MAE[0][1],
                    'best_p_o_2min': stage_MAE[0][2],
                    'best_p_e_2min': stage_MAE[0][3],
                    'best_p_w_3min': stage_MAE[1][0],
                    'best_p_i_3min': stage_MAE[1][1],
                    'best_p_o_3min': stage_MAE[1][2],
                    'best_p_e_3min': stage_MAE[1][3],
                    'best_p_w_5min': stage_MAE[2][0],
                    'best_p_i_5min': stage_MAE[2][1],
                    'best_p_o_5min': stage_MAE[2][2],
                    'best_p_e_5min': stage_MAE[2][3],
                })
                    


            if (i + 1)  == epochs:
                wandb.log({'final_epoch': i,

                        't_w': tool_MAE_horizon[0].item(),
                        't_i': tool_MAE_horizon[1].item(),
                        't_o': tool_MAE_horizon[2].item(),
                        't_e': tool_MAE_horizon[3].item(),

                        'p_w': stage_MAE_horizon[0].item(),
                        'p_i': stage_MAE_horizon[1].item(),
                        'p_o': stage_MAE_horizon[2].item(),
                        'p_e': stage_MAE_horizon[3].item(),

                        'loss': np.mean(tmp_total_loss_val)})
                
                wandb.log({
                    't_w_2min': tool_MAE[0][0],
                    't_i_2min': tool_MAE[0][1],
                    't_o_2min': tool_MAE[0][2],
                    't_e_2min': tool_MAE[0][3],
                    't_w_3min': tool_MAE[1][0],
                    't_i_3min': tool_MAE[1][1],
                    't_o_3min': tool_MAE[1][2],
                    't_e_3min': tool_MAE[1][3],
                    't_w_5min': tool_MAE[2][0],
                    't_i_5min': tool_MAE[2][1],
                    't_o_5min': tool_MAE[2][2],
                    't_e_5min': tool_MAE[2][3],
                    
                })

                wandb.log({
                    'p_w_2min': stage_MAE[0][0],
                    'p_i_2min': stage_MAE[0][1],
                    'p_o_2min': stage_MAE[0][2],
                    'p_e_2min': stage_MAE[0][3],
                    'p_w_3min': stage_MAE[1][0],
                    'p_i_3min': stage_MAE[1][1],
                    'p_o_3min': stage_MAE[1][2],
                    'p_e_3min': stage_MAE[1][3],
                    'p_w_5min': stage_MAE[2][0],
                    'p_i_5min': stage_MAE[2][1],
                    'p_o_5min': stage_MAE[2][2],
                    'p_e_5min': stage_MAE[2][3],
                    
                })

            # stat
            RSD_val.append(np.nanmean(tmp_RSD_val))
            stage_RSD_loss_values_val.append(np.nanmean(tmp_stageRSD_loss_values_val))
            tool_RSD_loss_values_val.append(np.nanmean(tmp_tool_stageRSD_loss_values_val))
            total_loss_val.append(np.nanmean(tmp_total_loss_val))


            plt.figure("train", (32, 12))

            plt.subplot(2, 4, 1)
            plt.title("Train RSD MAE Loss")
            x = [(i + 1) for i in range(len(RSD_train))]
            plt.xlabel("epoch")
            plt.plot(x, RSD_train)

            plt.subplot(2, 4, 2)
            plt.title("Train Stage RSD MAE Loss")
            x = [(i + 1) for i in range(len(stage_RSD_loss_values_train))]
            plt.xlabel("epoch")
            plt.plot(x, stage_RSD_loss_values_train)

            plt.subplot(2, 4, 3)
            plt.title("Train Tool RSD Time MAE Loss")
            x = [(i + 1) for i in range(len(tool_RSD_loss_values_train))]
            plt.xlabel("epoch")
            plt.plot(x, tool_RSD_loss_values_train)

            plt.subplot(2, 4, 4)
            plt.title("Train Total Loss")
            x = [(i + 1) for i in range(len(total_loss_train))]
            plt.xlabel("epoch")
            plt.plot(x, total_loss_train)

            plt.subplot(2, 4, 5)
            plt.title("Val RSD MAE Loss")
            x = [(i + 1) for i in range(len(RSD_val))]
            plt.xlabel("epoch")
            plt.plot(x, RSD_val)

            plt.subplot(2, 4, 6)
            plt.title("Val Stage RSD MAE Loss")
            x = [(i + 1) for i in range(len(stage_RSD_loss_values_val))]
            plt.xlabel("epoch")
            plt.plot(x, stage_RSD_loss_values_val)

            plt.subplot(2, 4, 7)
            plt.title("Val Tool RSD Time MAE Loss")
            x = [(i + 1) for i in range(len(tool_RSD_loss_values_val))]
            plt.xlabel("epoch")
            plt.plot(x, tool_RSD_loss_values_val)

            plt.subplot(2, 4, 8)
            plt.title("Val Total Loss")
            plt.xlabel("epoch")
            x = [(i + 1) * 1 for i in range(len(total_loss_val))]
            plt.plot(x, total_loss_val)

            save_dir = r'PNet_loss.png'
            plt.savefig(save_dir)
            plt.close("train")





    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def set_loss(self, horizon_list,
                     fpm = 60*25,):

        anticipation_loss_list = []
        consistency_loss_list = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for h in range(len(horizon_list)):

            tmp_h_frame = horizon_list[h] * self.fpm

            anticipation_loss_list.append(anticipation_mae(h = tmp_h_frame).to(device))
            consistency_loss_list.append(consistency_loss(h = tmp_h_frame).to(device))


        self.anticipation_loss_list = anticipation_loss_list
        self.consistency_loss_list  = consistency_loss_list

 
    def loss_sum_cal(self, output, target,
                     horizon_list,
                     output_n,
                     loss_weight = [1, 1, 1, 1],
                     type = 'tool',
                     log = False,
                     fpm = 60*25,
                     inference = False,):


        wMAE = 0
        inMAE = 0
        oMAE = 0
        eMAE = 0

        consistency_loss = 0

        MAE_list = []

        for  h in range(len(horizon_list)):

            tmp_wMAE, tmp_inMAE, tmp_oMAE, tmp_eMAE = self.anticipation_loss_list[h](output, target)
            tmp_consistency_loss = self.consistency_loss_list[h](output, target)


            if log == True:

                logging.info('{} wMAE {} min {}'.format(type, horizon_list[h], tmp_wMAE.item()/self.fpm))
                logging.info('{} inMAE {} min {}'.format(type, horizon_list[h], tmp_inMAE.item()/self.fpm))
                logging.info('{} oMAE {} min {}'.format(type, horizon_list[h], tmp_oMAE.item()/self.fpm))
                logging.info('{} eMAE {} min {}'.format(type, horizon_list[h], tmp_eMAE.item()/self.fpm))


            MAE_list.append([tmp_wMAE/self.fpm, tmp_inMAE/self.fpm, tmp_oMAE/self.fpm, tmp_eMAE/self.fpm])
            if inference == False:
                tmp_temp_precision = torch.exp(-self.temp_weight[h])
                print('temp precision {}'.format(tmp_temp_precision))
                wMAE += tmp_wMAE * tmp_temp_precision  + self.temp_weight[h]
                inMAE += tmp_inMAE * tmp_temp_precision  + self.temp_weight[h]
                oMAE += tmp_oMAE * tmp_temp_precision  + self.temp_weight[h]
                eMAE += tmp_eMAE * tmp_temp_precision + self.temp_weight[h]
                consistency_loss += tmp_consistency_loss 
            else:
                wMAE += tmp_wMAE / self.fpm 
                inMAE += tmp_inMAE / self.fpm
                oMAE += tmp_oMAE / self.fpm
                eMAE += tmp_eMAE / self.fpm
                consistency_loss += tmp_consistency_loss


        MAE_list_horizon_total = [wMAE, inMAE, oMAE, eMAE, consistency_loss]


        loss_total = loss_weight[0] * wMAE + loss_weight[1] * inMAE + loss_weight[2] * oMAE + loss_weight[3] * eMAE

        return loss_total, MAE_list, MAE_list_horizon_total


    def set_temp_weight(self, horizon_n=3):
        self.temp_weight = torch.nn.Parameter(torch.tensor(horizon_n * [0], dtype=torch.float32, device=self.device), requires_grad=True)



