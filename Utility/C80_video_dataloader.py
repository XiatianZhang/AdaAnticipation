import os
from natsort import natsorted
import numpy as np
import pandas as pd
import json
import glob
from torch.utils.data import Dataset
import torch
import logging
from PIL import Image
from torchvision import transforms


class SurgeryVideoDataset(Dataset):
    def __init__(self,
                 tensor_dir
                 ):
        logging.basicConfig(level=logging.DEBUG)
        # tensor dir
        self.tensor_dir = tensor_dir
        self.tensors = natsorted(glob.glob(os.path.join(self.tensor_dir, '*pt')))


    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):

        sample = torch.load(self.tensors[idx])

        return sample
    

class Train_SurgeryVideoDataset(Dataset):
    def __init__(self,
                 tensor_dir
                 ):
        logging.basicConfig(level=logging.DEBUG)
        # tensor dir
        self.tensor_dir = tensor_dir
        self.tensors = natsorted(glob.glob(os.path.join(self.tensor_dir, '*pt')))


    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx, max_frame=5995):

        sample = torch.load(self.tensors[idx])

        video = sample['video']
        video = torch.cat([video, torch.zeros([max_frame - video.shape[0], video.shape[-2], video.shape[-1]])], dim=0)
        boxes = sample['boxes']
        boxes = torch.cat([boxes, torch.zeros([max_frame - boxes.shape[0], boxes.shape[-2], boxes.shape[-1]])], dim=0)
        stage_RSD = sample['stage_RSD']
        stage_RSD = torch.cat([stage_RSD, torch.zeros([max_frame - stage_RSD.shape[0], stage_RSD.shape[-1]])], dim=0)
        tool_RSD = sample['tool_RSD']
        tool_RSD = torch.cat([tool_RSD, torch.zeros([max_frame - tool_RSD.shape[0], tool_RSD.shape[-1]])], dim=0)
        RSD = sample['RSD']
        RSD = torch.cat([RSD, torch.zeros([max_frame - RSD.shape[0]])], dim=0)
        Phase = sample['Phase']
        Phase = torch.cat([Phase, torch.zeros([max_frame - Phase.shape[0]])], dim=0)
        Phase = Phase.long()

        sample = {'video': video, 'boxes': boxes, 'stage_RSD': stage_RSD, 'tool_RSD': tool_RSD, 'RSD': RSD, 'Phase': Phase}

        return sample



