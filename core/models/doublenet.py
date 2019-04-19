# -*- coding: utf-8 -*-
import torchvision.models as models
import torch.nn as nn
from ..models import lenet
from ..config import Config

class DoubleNet(nn.Module):
    def __init__(self, config: Config):
        super(DoubleNet, self).__init__()
        self.input_shape = (-1, 3, config.image_resize[0], config.image_resize[1])
        self.use_gpu = config.use_gpu
        self.loss_type = config.loss_type
        if self.loss_type == 'cross_entropy':
            self.num_classes = config.num_classes
        elif self.loss_type == 'mseloss':
            self.num_classes = 1

        if config.module == 'alexnet':
            self.model1 = models.alexnet(num_classes=self.num_classes)
            self.model2 = models.alexnet(num_classes=self.num_classes)
        elif config.module == 'resnet':
            self.model1 = models.resnet18(num_classes=self.num_classes)
            self.model2 = models.resnet18(num_classes=self.num_classes)
        elif config.module == 'lenet':
            self.model1 = lenet.LeNet(num_classes=self.num_classes)
            self.model2 = lenet.LeNet(num_classes=self.num_classes)

    def forward(self, tensor_s, tensor_p):
        tensor_s = tensor_s.view(self.input_shape)
        tensor_p = tensor_p.view(self.input_shape)
        sub_probs = self.model1(tensor_s)
        pare_probs = self.model2(tensor_p)
        return sub_probs, pare_probs
