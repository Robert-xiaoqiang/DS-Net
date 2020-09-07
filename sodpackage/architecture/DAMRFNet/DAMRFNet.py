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

class DAMRFNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        extra = config.MODEL.EXTRA
        self.rgb_net = Backbone(config)
        self.depth_net = Backbone(config, 1)
        self.inplanes = self.rgb_net.last_stage_channels
        
        self.dams = nn.ModuleList([ DAM(p, p) for p in self.inplanes ])

        last_inp_channels = np.int(np.sum(self.inplanes))

        self.sod_last_layer = nn.Sequential(
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

        self.depth_last_layer = nn.Sequential(
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

    def forward(self, rgb, depth):
        ori_h, ori_w = rgb.shape[2], rgb.shape[3]

        rgb_feature = self.rgb_net(rgb)
        depth_feature = self.depth_net(depth)

        fused_feature = [ self.dams[i](rgb_feature[i], depth_feature[i]) for i in range(len(self.dams)) ]
        x = fused_feature

        # Upsampling 4 times
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)

        x = torch.cat([x[0], x1, x2, x3], 1)
        sod_segmentation = self.sod_last_layer(x)
        depth_estimation = self.depth_last_layer(x)

        sod_segmentation = F.interpolate(sod_segmentation, size=(ori_h, ori_w), mode='bilinear', align_corners=True)
        depth_estimation = F.interpolate(depth_estimation, size=(ori_h, ori_w), mode='bilinear', align_corners=True)

        return sod_segmentation, depth_estimation 

    def init_weights(self, pretrained = ''):
        pprint('=> init weights for rgb stream')
        self.rgb_net.init_weights(pretrained)
        pprint('=> init weights for depth stream')
        self.depth_net.init_weights(pretrained)
