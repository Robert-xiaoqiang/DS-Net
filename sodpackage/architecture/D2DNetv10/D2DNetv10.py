import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from pprint import pprint

import os

from .Backbone import Backbone
from .RGB2DepthNet import RGB2DepthNet

from ..Component.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d
BN_MOMENTUM = 0.01

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

class DepthGatedModule(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        self.inplanes = inplanes

        self.rgb_encoder = nn.Conv2d(self.inplanes, self.inplanes, 1, stride = 1, padding = 0)
        self.depth_lhs_encoder = nn.Conv2d(self.inplanes, self.inplanes, 1, stride = 1, padding = 0)
        self.depth_rhs_encoder = nn.Conv2d(self.inplanes, self.inplanes, 1, stride = 1, padding = 0)

        self.decoder = nn.Sequential(
            nn.Conv2d(self.inplanes, self.inplanes, 1, stride = 1, padding = 0),
            BatchNorm2d(self.inplanes),
            nn.ReLU(inplace = True)
        )

    def forward(self, x, from_depth_estimation):
        B, C, H, W = x.shape

        x = self.rgb_encoder(x)
        lhs = self.depth_lhs_encoder(from_depth_estimation)
        rhs = self.depth_rhs_encoder(from_depth_estimation)

        transposed_x = x.permute(0, 2, 3, 1).reshape(-1, C)
        lhs = lhs.permute(0, 2, 3, 1).reshape(-1, C)
        rhs = rhs.permute(0, 2, 3, 1).reshape(-1, C)
        # BHW * BHW inner-product similarity
        logits = torch.matmul(lhs, rhs.T)
        # row posterior/weight/relation/interaction normalization
        weight = F.softmax(logits, dim = 1)
        enhanced_x = torch.matmul(weight, transposed_x).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        y = self.decoder(enhanced_x)

        return y

class DepthDistillingModule(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        self.inplanes = inplanes
        self.encoder = nn.Conv2d(self.inplanes, self.inplanes, 1, stride = 1, padding = 0)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.inplanes, self.inplanes // 4, 1, stride = 1, padding = 0),
            BatchNorm2d(self.inplanes // 4),
            nn.ReLU(inplace = True),
            nn.Conv2d(self.inplanes // 4, 1, 1, stride = 1, padding = 0),
            nn.Sigmoid()
        )

    def forward(self, d):
        ed = self.encoder(d)
        y = self.decoder(ed + d)

        return y

class ComplementaryGatedFusion(nn.Module):
    def __init__(self, multi_scale_inplanes):
        super().__init__()
        self.multi_scale_inplanes = multi_scale_inplanes
        self.ns = len(self.multi_scale_inplanes)
        self.dams = nn.ModuleList([ DepthAwarenessModule(p, p) for p in self.multi_scale_inplanes ])
        self.dgms = nn.ModuleList([ DepthGatedModule(p) for p in self.multi_scale_inplanes ])
        self.ddms = nn.ModuleList([ DepthDistillingModule(p) for p in self.multi_scale_inplanes ])
    
    def forward(self, from_depth_estimation, from_rgb, from_depth_extraction):
        dg_feature = [ self.dgms[i](from_depth_extraction[i], from_depth_estimation[i]) for i in range(self.ns) ]
        da_feature = [ self.dams[i](from_rgb[i], from_depth_extraction[i]) for i in range(self.ns) ]
        dd_map = [ self.ddms[i](from_rgb) for i in range(self.ns) ]
        y = [ dd_map[i] * da_feature[i] + dg_feature[i] for i in range(self.ns) ]

        return y

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
    def __init__(self, multi_scale_inplanes):
        super().__init__()
        # list channels
        self.multi_scale_inplanes = multi_scale_inplanes
        self.ns = len(self.multi_scale_inplanes)
        self.layers = nn.ModuleList([ DecoderSubnet(p) for p in self.multi_scale_inplanes ])

    def forward(self, x):
        assert len(x) == self.ns, 'please make sure multi-scale output'
        return [ self.layers[i](x[i]) for i in range(self.ns) ]

class DecoderReconstructor(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        self.inplanes = inplanes
        self.outplanes = inplanes // 2
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.inplanes,
                out_channels=self.outplanes,
                kernel_size=3,
                stride=1,
                padding=1),
            BatchNorm2d(self.outplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=self.outplanes,
                out_channels=self.outplanes,
                kernel_size=3,
                stride=1,
                padding=1)
        )
    def forward(self, x):
        y = self.layer(x)

        return y

class MSDecoderReconstructor(nn.Module):
    def __init__(self, multi_scale_inplanes):
        super().__init__()
        # list channels
        self.multi_scale_inplanes = multi_scale_inplanes
        self.ns = len(self.multi_scale_inplanes)
        self.layers = nn.ModuleList([ DecoderReconstructor(p) for p in self.multi_scale_inplanes ])

    def forward(self, x):
        assert len(x) == self.ns, 'please make sure multi-scale output'
        y = [ self.layers[i](x[i]) for i in range(self.ns) ]
        return y

class DisentangleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction = 'mean')

    def forward(self, reconstructed, original):
        assert reconstructed.shape == original.shape, 'different shape in DisentangleLoss'
        softmax_reconstructed = F.softmax(reconstructed, dim = 1)
        softmax_original = F.softmax(original, dim = 1)

        ret = self.loss(softmax_reconstructed, softmax_original)
        return ret

class MSDisentangleLoss(nn.Module):
    def __init__(self, multi_scale_inplanes):
        super().__init__()
        # list channels
        self.multi_scale_inplanes = multi_scale_inplanes
        self.ns = len(self.multi_scale_inplanes)
        self.layers = nn.ModuleList([ nn.MSELoss(reduction = 'mean') for _ in range(self.ns) ])

    def forward(self, reconstructed, original):
        assert len(reconstructed) == len(original) == self.ns, 'please make sure multi-scale output'
        
        # reconstruction(loss) of 4 different scales
        loss = [ self.layers[i](reconstructed[i], original[i]) for i in range(self.ns) ]
        # sum the list of losses for every input RGB image
        y = sum(loss)

        return y

class D2DNetv10(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        extra = self.config.MODEL.EXTRA

        self.rgb2depth = RGB2DepthNet(self.config)
        last_stage_channels = self.rgb2depth.last_stage_channels
        sum_last_stage_channels = self.rgb2depth.sum_last_stage_channels
        self.ns = len(last_stage_channels)

        self.rgb_subnet = MSDecoderSubnet(last_stage_channels)
        self.depth_estimation_subnet = MSDecoderSubnet(last_stage_channels)

        self.disentangle_reconstructor = MSDecoderReconstructor(list(map(lambda x: 2*x, last_stage_channels)))
        self.disentangle_loss = MSDisentangleLoss(last_stage_channels)

        self.depth_extraction_encoder = Backbone(self.config, 1)
        self.cgf = ComplementaryGatedFusion(last_stage_channels)

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=sum_last_stage_channels,
                out_channels=sum_last_stage_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(sum_last_stage_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=sum_last_stage_channels,
                out_channels=1,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0),
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
        common_encoder_feature, from_depth_estimation, depth_output = self.rgb2depth(rgb)
        from_rgb = self.rgb_subnet(common_encoder_feature)

        disentangled_feature = [ torch.cat([ from_rgb[i], from_depth_estimation[i] ], dim = 1) for i in range(self.ns) ]
        reconstructed_feature = self.disentangle_reconstructor(disentangled_feature)
        reconstruct_loss = self.disentangle_loss(reconstructed_feature, common_encoder_feature)

        depth_extraction_feature = self.depth_extraction_encoder(depth)
        from_depth_extraction = depth_extraction_feature # depth_extraction subnet ????

        adaptive_combination = self.cgf(from_depth_estimation, from_rgb, from_depth_extraction)
        
        adaptive_combination = D2DNetv10.merge(adaptive_combination)
        y = self.last_layer(adaptive_combination)

        y = F.interpolate(y, size=(ori_h, ori_w), mode='bilinear', align_corners=True)
        sod_output = y

        output = sod_output if self.config.TRAIN.MTL_OUTPUT == 'single' else (sod_output, depth_output, reconstruct_loss)
        
        return output

    def init_weights(self, pretrained_backbone = ''):
        pprint('=> init weights from Gaussion distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        pprint('=> init weights from ImageNet pretraining')
        pprint('=> init weights for encoder(rgb2depth)')
        self.rgb2depth.init_weights(pretrained_backbone)
        pprint('=> init weights for encoder(depth extraction)')
        self.depth_extraction_encoder.init_weights(pretrained_backbone)

    def init_pretext(self, pretrained_rgb2depth = ''):
        pprint('=> init weights from pretext task pretraining')
        if os.path.isfile(pretrained_rgb2depth):
            pprint('=> loading depth estimation pretrained model {}'.format(pretrained_rgb2depth))
            # gpu to cpu
            persistable_dict = torch.load(pretrained_rgb2depth, map_location = lambda storage, loc: storage)
            pretrained_dict = persistable_dict['model_state_dict']
            resumed_epoch = persistable_dict['epoch']
            # strip depth estimation head or preserve it for depth supervision when sod model training
            # gpu fullmodel to cpu standalone heuristicly
            modified_dict = { }
            for k, v in pretrained_dict.items():
                modified_k = k[7+6:]
                if modified_k not in self.rgb2depth.state_dict():
                    pprint('dismatched key: {}'.format(k))
                else:
                    modified_dict[modified_k] = v
            self.rgb2depth.load_state_dict(modified_dict)
            pprint('=> loaded depth estimation pretrained model {} from best epoch {}'.format(pretrained_rgb2depth, resumed_epoch))
        else:
            pprint('=> cannot find depth estimation pretrained model')
