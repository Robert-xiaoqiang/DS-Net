import torch
from torch import nn
from troch.nn import init
from torch.nn import functional as F

from .Backbone import Backbone
from .RGB2DepthNetEncoder import RGB2DepthNetEncoder

from ..Component.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d
BN_MOMENTUM = 0.01

class AffinityLayer(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        self.encoder = nn.Conv2d(inplanes, inplanes // 2, 1, stride = 1, padding = 0)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.encoder(x)
        # BHW * C
        x_transposed = x.permute(0, 2, 3, 1)
        lhs = x_transposed.view(B * H * W, C)
        rhs = lhs.transpose()
        
        # all-pair / pair-wise weight or relation
        similarity = torch.mm(lhs, rhs)

        # row normalization
        y = F.normalize(similarity, p = 2, dim = 1)

        return y

class DiffusionLayer(nn.Module):
    beta = 0.4
    def __init__(self, inplanes):
        super().__init__()
        self.decoder = nn.Conv2d(inplanes, inplanes * 2, 1, stride = 1, padding = 0)

    def forward(self, x, combination):
        B, C, H, W = x.shape
        combination = self.decoder(combination)

        x_transposed = x.permute(0, 2, 3, 1)
        # BHW * C
        propagation = x_transposed.view(B * H * W, C)
        for _ in range(4):
            # BHW * BHW, BHW * C
            propagation = torch.mm(combination, propagation)
        
        propagation = propagation.view(B, H, W, C).permute(0, 3, 1, 2)

        y = beta * propagation + (1 - beta) * x

        return y

class DepthAwarenessModule(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.conv_mask = nn.Conv2d(self.inplanes, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.softmax_channel = nn.Softmax(dim=1)
        self.channel_mul_conv = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.planes, self.inplanes, kernel_size=(1, 1), stride=(1, 1), bias=False),
        )

    def spatial_pool(self, depth_feature):
        batch, channel, height, width = depth_feature.size()
        input_x = depth_feature
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(depth_feature)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        # context attention
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x, depth_feature):
        # [N, C, 1, 1]
        context = self.spatial_pool(depth_feature)
        # [N, C, 1, 1]
        channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
        # channel-wise attention
        out1 = torch.sigmoid(depth_feature * channel_mul_term)
        # fusion
        out = x * out1

        return torch.sigmoid(out)

class ComplementaryGatedFusion(nn.Module):
    def __init__(self, inplanes):
        self.inplanes = inplanes
        # requires_grad == True
        self.alpha1 = nn.Parameter(torch.tensor(1.0))
        self.alpha2 = nn.Parameter(torch.tensor(1.0))
        # self.alpha3 = nn.Parameter(troch.tensor(1.0))

        self.affinity1 = AffinityLayer(self.inplanes)
        self.affinity2 = AffinityLayer(self.inplanes)
        # self.affinity3 = AffinityLayer(self.inplanes)

        self.diffusion = DiffusionLayer(self.inplanes // 2)

        self.dam = DepthAwarenessModule(self.inplanes, self.inplanes)

    def forward(self, from_depth_estimation, from_rgb, from_depth_extraction):
        x1 = self.affinity1(from_depth_estimation)
        x2 = self.affinity2(from_rgb)
        # x3 = self.affinity3(from_depth_extraction)

        combination = self.alpha1 * x1 + self.alpha2 * x2 #+ self.alpha3 * x3

        # note: from_rgb not x2
        enhanced_rgb = self.diffusion(from_rgb, combination)

        y = self.dam(enhanced_rgb, from_depth_extraction)

        return y

class DecoderSubnet(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        self.inplanes = inplanes
        self.layer = nn.Sequential(
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
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0)
        )
    def forward(self, x):
        y = self.layer(x)

        return y

class D2DNet(nn.Module):
    def __init__(self, config):
        super.__init__()
        self.config = config
        extra = self.config.MODEL.EXTRA

        self.rgb2depth = RGB2DepthNetEncoder(self.config)
        last_inp_channels = self.rgb2depth.last_inp_channels
        self.depth_estimation_subnet = DecoderSubnet(last_inp_channels)
        self.rgb_subnet = DecoderSubnet(last_inp_channels)
        self.depth_extraction_encoder = Backbone(self.config, 1)
        self.depth_extraction_subnet = DecoderSubnet(last_inp_channels)   
        self.cgf = ComplementaryGatedFusion(last_inp_channels)
        self.last_layer = nn.Sequential(
            DecoderSubnet(last_inp_channels),
            nn.Sigmoid()
        )

    @staticmethod
    def merge(x):
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)

        y = torch.cat([x[0], x1, x2, x3], 1)

        return y

    def forward(self, rgb, depth):
        ori_h, ori_w = rgb.shape[2], rgb.shape[3]
        # list of features
        common_encoder_feature = self.rgb2depth(rgb)
        depth_extraction_feature = self.depth_extraction_encoder(depth)

        common_encoder_feature = D2DNet.merge(common_encoder_feature)
        depth_extraction_feature = D2DNet.merge(depth_extraction_feature)

        from_depth_estimation = self.depth_estimation_subnet(common_encoder_feature)
        from_rgb = self.rgb_subnet(common_encoder_feature)
        from_depth_extraction = self.depth_extraction_subnet(depth_extraction_feature)
        
        adaptive_combination = self.cgf(from_depth_estimation, from_rgb, from_depth_extraction)
        
        y = self.last_layer(adaptive_combination)
        y = F.interpolate(y, size=(ori_h, ori_w), mode='bilinear', align_corners=True)
        
        return y

    def init_weights(self, pretrained_backbone, pretrained_rgb2depth_encoder):
        