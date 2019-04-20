# -*- coding: utf-8 -*-
from torch import nn
import torch as t
from torch.nn import functional as F
from ..config import Config

class LeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, (5, 5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        # self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.use_gpu = Config.use_gpu

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.size(0), -1)
        if self.use_gpu:
            with t.cuda.device(0):
                x = x.cpu()
        fc1 = nn.Linear(x.size(1), 120)
        x = F.relu(fc1(x))
        x = F.dropout(x)
        if self.use_gpu:
            with t.cuda.device(0):
                x = x.cuda()
        x = F.relu(self.fc2(x))
        x = F.dropout(x)
        x = self.fc3(x)
        return x
