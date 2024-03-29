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

class RGB2DepthNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        extra = config.MODEL.EXTRA

        self.encoder = Backbone(config, 3)
        self.inplanes = self.encoder.last_stage_channels
        last_inp_channels = np.int(np.sum(np.asarray(self.inplanes)))

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=1,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0),
            nn.Sigmoid()
        )

    def forward(self, rgb):
        ori_h, ori_w = rgb.shape[2], rgb.shape[3]

        x = self.encoder(rgb)

        # Upsampling 4 times
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)

        x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.last_layer(x)

        x = F.interpolate(x, size=(ori_h, ori_w), mode='bilinear', align_corners=True)

        return x

    def init_weights(self, pretrained = ''):
        pprint('=> init weights from Gaussion distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        pprint('=> init weights from ImageNet pretraining')
        pprint('=> init weights for encoder(rgb2depth)')
        self.encoder.init_weights(pretrained)