import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import numpy as np
from pprint import pprint

from .DAM import DAM
from .Backbone import Backbone

from ..Component.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d
BN_MOMENTUM = 0.01
       
class D2DNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        extra = config.MODEL.EXTRA
        self.r2d = RGB2DepthNet(config)

    def forward(self, rbg, depth):
        pass

    def init_weights(self, pretrained = ''):
        pprint('=> init weights for rgb stream')
        self.rgb_net.init_weights(pretrained)
        pprint('=> init weights for depth stream')
        self.depth_net.init_weights(pretrained)
