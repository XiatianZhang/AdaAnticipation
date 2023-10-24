import torch
import torch.nn as nn

class anticipation_mae_train(nn.Module):

    def __init__(self, h=7500):
        super(anticipation_mae_train, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.h  = torch.tensor(h).float().to(self.device)
        self.lossfunc = nn.L1Loss()

    def forward(self, output_rsd, target_rsd):

        wMAE = []
        inMAE = []
        oMAE = []
        eMAE = []

        if len(target_rsd.size()) < 3:
            target_rsd = target_rsd.unsqueeze(-1)

        for n in range(output_rsd.size(0)):
            for c in range(output_rsd.size(-1)):
                tmp_output = output_rsd[n,:,c]
                tmp_target = target_rsd[n,:,c]

                # tmp_output = torch.where((tmp_output > self.h)|(tmp_output <0), self.h, tmp_output)
                tmp_target = torch.where((tmp_target > self.h)|(tmp_target <0), self.h, tmp_target)


                cond_oMAE = torch.where((tmp_target == self.h))
                cond_inMAE = torch.where((tmp_target<self.h) & (tmp_target>0))
                cond_uMAE = torch.where((tmp_target == 0))
                cond_eMAE = torch.where((tmp_target < 0.1 * self.h) & (tmp_target > 0))


                output_oMAE = torch.abs(tmp_output[cond_oMAE] - tmp_target[cond_oMAE])
                output_inMAE = torch.abs(tmp_output[cond_inMAE] - tmp_target[cond_inMAE])
                output_uMAE = torch.abs(tmp_output[cond_uMAE] - tmp_target[cond_uMAE])
                output_inMAE = torch.nanmean(torch.stack((torch.nanmean(output_uMAE),torch.nanmean(output_inMAE))))
                output_wMAE = torch.nanmean(torch.stack((torch.nanmean(output_oMAE),torch.nanmean(output_inMAE))))
                output_eMAE = torch.abs(tmp_output[cond_eMAE] - tmp_target[cond_eMAE])



                wMAE.append(output_wMAE)
                inMAE.append(torch.mean(output_inMAE))
                oMAE.append(torch.mean(output_oMAE))
                eMAE.append(torch.mean(output_eMAE))

        # use the mean over instrument types in minutes per metric
        # use the nan mean in case there is no corresponding instrument in the current sequence
        wMAE = torch.nanmean(torch.stack(wMAE))
        inMAE = torch.nanmean(torch.stack(inMAE))
        oMAE = torch.nanmean(torch.stack(oMAE))
        eMAE = torch.nanmean(torch.stack(eMAE))


        return wMAE, inMAE, oMAE, eMAE

class anticipation_mae(nn.Module):

    def __init__(self, h=7500):
        super(anticipation_mae, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.h  = torch.tensor(h).float().to(self.device)
        self.lossfunc = nn.L1Loss()

    def forward(self, output_rsd, target_rsd):

        wMAE = []
        inMAE = []
        oMAE = []
        eMAE = []

        if len(target_rsd.size()) < 3:
            target_rsd = target_rsd.unsqueeze(-1)

        for n in range(output_rsd.size(0)):
            for c in range(output_rsd.size(-1)):
                tmp_output = output_rsd[n,:,c]
                tmp_target = target_rsd[n,:,c]

                tmp_output = torch.where((tmp_output > self.h)|(tmp_output <0), self.h, tmp_output)
                tmp_target = torch.where((tmp_target > self.h)|(tmp_target <0), self.h, tmp_target)


                cond_oMAE = torch.where((tmp_target == self.h))
                cond_inMAE = torch.where((tmp_target<self.h) & (tmp_target>0))
                cond_eMAE = torch.where((tmp_target < 0.1 * self.h) & (tmp_target > 0))


                output_oMAE = torch.abs(tmp_output[cond_oMAE] - tmp_target[cond_oMAE])
                output_inMAE = torch.abs(tmp_output[cond_inMAE] - tmp_target[cond_inMAE])
                output_wMAE = torch.nanmean(torch.stack((torch.nanmean(output_oMAE),torch.nanmean(output_inMAE))))
                output_eMAE = torch.abs(tmp_output[cond_eMAE] - tmp_target[cond_eMAE])



                wMAE.append(output_wMAE)
                inMAE.append(torch.mean(output_inMAE))
                oMAE.append(torch.mean(output_oMAE))
                eMAE.append(torch.mean(output_eMAE))

        # use the mean over instrument types in minutes per metric
        # use the nan mean in case there is no corresponding instrument in the current sequence
        wMAE = torch.nanmean(torch.stack(wMAE))
        inMAE = torch.nanmean(torch.stack(inMAE))
        oMAE = torch.nanmean(torch.stack(oMAE))
        eMAE = torch.nanmean(torch.stack(eMAE))


        return wMAE, inMAE, oMAE, eMAE

class attention_loss(nn.Module):
    def __init__(self, sigma = 0.05):
        super(attention_loss, self).__init__()


        self.sigma = sigma



    def forward(self, a ):

        loss  = self.sigma*torch.sum(a)

        return loss


class framewise_ce(nn.Module):
    def __init__(self, sigma = 0.05):
        super(framewise_ce, self).__init__()


        self.ce = nn.CrossEntropyLoss(size_average=True)



    def forward(self, x, target):

        loss = 0
        for b in range(x.size(0)):

            loss+= self.ce(x[b,:,:],target[b,:])


        return loss


class similarity_loss(nn.Module):
    def __init__(self, sigma = 0.05):
        super(similarity_loss, self).__init__()


        self.similarity_loss = nn.CosineSimilarity(dim=2)



    def forward(self, output,target ):

        loss  = torch.mean(self.similarity_loss(output,target))

        return loss

class class_wise_anticipation_mae(nn.Module):

    def __init__(self, h=7500):
        super(class_wise_anticipation_mae, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.h = torch.tensor(h).float().to(self.device)
        self.lossfunc = anticipation_mae(h)
        self.base = [0]

    def forward(self, output_rsd, target_rsd):

        c = output_rsd.size(-1)
        wMAE_loss_list  = self.base*c
        inMAE_loss_list = self.base*c
        oMAE_loss_list = self.base*c
        eMAE_loss_list = self.base*c

        for i in range(c):

            tmp_output_rsd = output_rsd[:,:,i:i+1]
            tmp_target_rsd = target_rsd[:,:,i:i+1]
            wMAE_loss_list[i] , inMAE_loss_list[i],oMAE_loss_list[i], eMAE_loss_list[i] = self.lossfunc(tmp_output_rsd,tmp_target_rsd)

        output_loss_list = [wMAE_loss_list, inMAE_loss_list,oMAE_loss_list,eMAE_loss_list]
        return output_loss_list


class consistency_loss(nn.Module):

    def __init__(self, h=7500):
        super(consistency_loss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.h  = torch.tensor(h).float().to(self.device)
        self.lossfunc = nn.CosineSimilarity(dim=-1, eps=1e-6)


    def forward(self, output_rsd, target_rsd):

        if len(target_rsd.size()) < 3:
            target_rsd = target_rsd.unsqueeze(-1)

        for n in range(output_rsd.size(0)):
            for c in range(output_rsd.size(-1)):
                tmp_output = output_rsd[n,:,c]
                tmp_target = target_rsd[n,:,c]

                tmp_target = torch.where((tmp_target > self.h)|(tmp_target <0), self.h, tmp_target)


                cond_o = torch.where((tmp_target == self.h))
                cond_in = torch.where((tmp_target<self.h) & (tmp_target>0))

                output_o = tmp_output[cond_o]
                output_o_loss = self.compute_loss(output_o)

                output_in = tmp_output[cond_in]
                output_in_loss = self.compute_loss(output_in)

                return output_o_loss + output_in_loss

    def compute_loss(self, x):

        x_p_step = x[:-1:2]
        x_n_step = x[1::2]
        loss = self.lossfunc(x_p_step, x_n_step)

        return - loss

class seq_class_loss(nn.Module):

    def __init__(self):
        super(seq_class_loss, self).__init__()

        self.ce =  nn.CrossEntropyLoss()

    def forward(self, output, target):

        b = output.size(0)

        loss = 0
        for i in range(b):

            tmp_output = output[i, :, :]
            tmp_target = target[i, :]

            tmp_loss = self.ce(tmp_output, tmp_target)

            loss += tmp_loss

        return loss
