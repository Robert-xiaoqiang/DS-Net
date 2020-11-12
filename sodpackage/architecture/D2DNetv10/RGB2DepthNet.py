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

class DecoderSubnet(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        self.inplanes = inplanes
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.inplanes,
                out_channels=self.inplanes,
                kernel_size=3,
                stride=1,
                padding=1),
            BatchNorm2d(self.inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=self.inplanes,
                out_channels=self.inplanes,
                kernel_size=3,
                stride=1,
                padding=1)
        )
    def forward(self, x):
        y = self.layer(x)

        return y

class MSDecoderSubnet(nn.Module):
    def __init__(self, muitl_scale_inplanes):
        super().__init__()
        # list channels
        self.muitl_scale_inplanes = muitl_scale_inplanes
        self.ns = len(self.muitl_scale_inplanes)
        self.layers = nn.ModuleList([ DecoderSubnet(p) for p in self.muitl_scale_inplanes ])

    def forward(self, x):
        assert len(x) == self.ns, 'please make sure multi-scale output'
        return [ self.layers[i](x[i]) for i in range(self.ns) ]

class RGB2DepthNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        extra = config.MODEL.EXTRA

        self.encoder = Backbone(config, 3)
        self.last_stage_channels = self.encoder.last_stage_channels
        self.sum_last_stage_channels = np.int(np.sum(np.asarray(self.last_stage_channels)))
        
        self.depth_estimation_subnet = MSDecoderSubnet(self.last_stage_channels)
        # TO-DO sigmoid must not be used to regress !!!!
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.sum_last_stage_channels,
                out_channels=self.sum_last_stage_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(self.sum_last_stage_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=self.sum_last_stage_channels,
                out_channels=1,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0),
            nn.Sigmoid()
        )

    def forward(self, rgb):
        ori_h, ori_w = rgb.shape[2], rgb.shape[3]

        x = self.encoder(rgb)
        encoder_output = x

        x = self.depth_estimation_subnet(x)
        from_depth_estimation = x

        # Upsampling 4 times
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)

        x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.last_layer(x)

        x = F.interpolate(x, size=(ori_h, ori_w), mode='bilinear', align_corners=True)

        return encoder_output, from_depth_estimation, x

    def init_weights(self, pretrained = ''):
        self.encoder.init_weights(pretrained)
