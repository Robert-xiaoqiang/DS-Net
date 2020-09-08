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

class MSDAM(nn.Module):
    def __init__(self, channels_list):
        self.n = len(channels_list)
        self.dams = nn.ModuleList([ DAM(p, p) for p in channels_list ])

    def forward(self, rgb_feature, depth_feature):
        assert type(rgb_feature) == type(depth_feature) == list and \
               len(rgb_feature) == self.n, 'backbone features should be multi-scale'
        target_size = rgb_feature[0].shape[2:]

        target_list = [ ]
        for i in range(self.n):
            f = self.dams(rgb_feature[i], depth_feature[i])
            # maybe identity
            f = F.interpolate(f, size = target_size, mode = 'bilinear', align_corners = True)
            target_list.append(f)
        target = torch.cat(target_list, dim = 1)

        return target

class MSDecoder(nn.Module):
    def __init__(self, inplanes):
        self.inplanes = inplanes
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.inplanes,
                out_channels=self.inplanes,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(self.inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=self.inplanes,
                out_channels=1,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.last_layer(x)

        return y

class MSMRFNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        extra = config.MODEL.EXTRA
        self.rgb_net = Backbone(config)
        self.depth_net = Backbone(config, 1)
        self.stage_channels = self.rgb_net.stage_channels # list of list
        self.inplanes = self.rgb_net.last_stage_channels # self.stage_channels[-1]

        self.msdams = nn.ModuleList([ MSDAM(channels_list) for channels_list in self.stage_channels ])

        last_inp_channels = np.int(np.sum(self.inplanes))

        self.decoders = nn.ModuleList([ MSDecoder(np.int(np.sum(l))) for l in self.stage_channels ])

    def forward(self, rgb, depth):
        ori_h, ori_w = rgb.shape[2], rgb.shape[3]

        # list of list
        rgb_feature = self.rgb_net(rgb)
        depth_feature = self.depth_net(depth)
        
        num_stages = len(self.msdams)

        # list
        fused_feature = [ self.msdams[i](rgb_feature[i], depth_feature[i]) for i in range(num_stages) ]
        x = fused_feature

        x = [ self.decoders[i](x[i]) for i in range(num_stages) ]

        # Upsampling 4 times
        x = [ F.interpolate(x, size=(ori_h, ori_w), mode='bilinear', align_corners=True) for i in range(num_stages) ]

        return x

    def init_weights(self, pretrained = ''):
        pprint('=> init weights for rgb stream')
        self.rgb_net.init_weights(pretrained)
        pprint('=> init weights for depth stream')
        self.depth_net.init_weights(pretrained)
