import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import numpy as np
from pprint import pprint

from .Backbone import Backbone

from ..Component.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d
BN_MOMENTUM = 0.01

class RGB2DepthNetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        extra = self.config.MODEL.EXTRA
        self.encoder = Backbone(self.config, 3)
        self.last_inp_channels = self.encoder.last_stage_channels

    def forward(self, rgb):
        ori_h, ori_w = rgb.shape[2], rgb.shape[3]

        x = self.encoder(rgb)
        encoder_output = x

        return encoder_output

    def init_weights(self, pretrained = ''):
        pprint('=> init weights for encoder(RGB2DepthNet)')
        self.encoder.init_weights(pretrained)