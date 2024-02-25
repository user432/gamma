# https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_utils.py

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F




class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True,  channel=3):
        super(PointNetEncoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.global_feat = global_feat


    def forward(self, x):
        B, D, N = x.size()

        x = F.relu(self.conv1(x))

        pointfeat = x
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)
        if self.global_feat:
            return x
        else:
            x = x.view(-1, 512, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1)



