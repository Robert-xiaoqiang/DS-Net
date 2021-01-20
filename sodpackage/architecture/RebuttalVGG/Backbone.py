import os
import sys
import logging
from pprint import pprint
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from ..Component.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d
BN_MOMENTUM = 0.01

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

def make_layers(cfg, initial_inplanes, batch_norm = False):
    layers = []
    in_channels = initial_inplanes
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16_bn': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19_bn': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512,
          512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, config, initial_inplanes):
        super().__init__()
        extra = config.MODEL.EXTRA
        self.initial_inplanes = initial_inplanes
        self.arch_name = extra['ARCH_NAME']
        self.features = make_layers(cfgs[self.arch_name], self.initial_inplanes, True)

        self.last_stage_channels = [ 128, 256, 512, 512 ]
    
    def forward(self, x):
        # extract the feature maps just before max pooling layer
        if self.arch_name == 'vgg16_bn':
            div1 = nn.Sequential(*tuple(self.features.children())[0:6])
            div2 = nn.Sequential(*tuple(self.features.children())[6:13])
            div4 = nn.Sequential(*tuple(self.features.children())[13:23])
            div8 = nn.Sequential(*tuple(self.features.children())[23:33])
            div16 = nn.Sequential(*tuple(self.features.children())[33:43])
        elif self.arch_name == 'vgg19_bn':
            div1 = nn.Sequential(*tuple(self.features.children())[0:6])
            div2 = nn.Sequential(*tuple(self.features.children())[6:13])
            div4 = nn.Sequential(*tuple(self.features.children())[13:26])
            div8 = nn.Sequential(*tuple(self.features.children())[26:39])
            div16 = nn.Sequential(*tuple(self.features.children())[39:52])
        else:
            raise NotImplementedError
        
        x1 = div1(x)
        x2 = div2(x1)
        x3 = div4(x2)
        x4 = div8(x3)
        x5 = div16(x4)

        # every feature map is larger than the one from ResNet or HRNet
        ret = [ x2, x3, x4, x5 ]
        
        return ret
    
    def init_weights(self, dummy_dirname):
        pretrained_dict = model_zoo.load_url(model_urls[self.arch_name])
        pprint('=> loading ImageNet scratch pretrained model model_zoo.{}'.format(self.arch_name))
        model_dict = self.state_dict()

        filtered_dict = { }
        for k, v in pretrained_dict.items():
            if k in model_dict.keys():
                if k.startswith('features.0') and self.initial_inplanes == 1:
                    continue
                filtered_dict[k] = v

        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict)
        pprint('=> loaded ImageNet scratch pretrained model model_zoo.{}'.format(self.arch_name))

Backbone = VGG