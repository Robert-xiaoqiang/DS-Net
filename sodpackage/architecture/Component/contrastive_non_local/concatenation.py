import torch
from torch import nn
from torch.nn import functional as F


class NonLocalBlockBase(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super().__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.concat_project = nn.Sequential(
            nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
            nn.ReLU()
        )

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # (b, c, N, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
        # (b, c, 1, N)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)

        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        f = self.concat_project(concat_feature)
        b, _, h, w = f.size()
        f = f.view(b, h, w)

        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)

        return W_y


class NonLocalBlock1D(NonLocalBlockBase):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super().__init__(in_channels,
                            inter_channels=inter_channels,
                            dimension=1, sub_sample=sub_sample,
                            bn_layer=bn_layer)
    def forward(self, x):
        response = super().forward(x)
        y = response + x

        return y


class NonLocalBlock2D(NonLocalBlockBase):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super().__init__(in_channels,
                            inter_channels=inter_channels,
                            dimension=2, sub_sample=sub_sample,
                            bn_layer=bn_layer)

    def forward(self, x):
        response = super().forward(x)
        y = response + x

        return y


class NonLocalBlock3D(NonLocalBlockBase):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super().__init__(in_channels,
                            inter_channels=inter_channels,
                            dimension=3, sub_sample=sub_sample,
                            bn_layer=bn_layer)

    def forward(self, x):
        response = super().forward(x)
        y = response + x

        return y

# n-d data -> n+1 view
# every view -> in_channel and inter_channel(defautl None for halving the corresponding in_channel)
# type(in_channels) == type(inter_channels) == list
class MultiViewNonLocalBlockBase(nn.Module):
    def __init__(self, n_views, in_channels, inter_channels = None, dimension = 3, sub_sample = True, bn_layer = True):
        super().__init__()
        self.n_views = n_views
        self.blocks = nn.ModuleList([ NonLocalBlockBase(in_channels[i], inter_channels[i] if inter_channels is not None else None, dimension, sub_sample, bn_layer) for i in range(self.n_views) ])
    
    def forward(self, x):
        raise NotImplementedError

class MultiViewNonLocalBlock1D(MultiViewNonLocalBlockBase):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super().__init__(2, in_channels, inter_channels=inter_channels, dimension=1, sub_sample=sub_sample, bn_layer=bn_layer)

    def forward(self, x):
        B, C, H = x.shape
        x_c = x
        response_c = self.blocks[0](x_c)

        x_h = x.permute(0, 2, 1)
        response_h = self.blocks[1](x_h)
        response_h = response_h.permute(0, 2, 1)

        y = response_c + response_h + x

        return y

class MultiViewNonLocalBlock2D(MultiViewNonLocalBlockBase):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super().__init__(3, in_channels, inter_channels=inter_channels, dimension=2, sub_sample=sub_sample, bn_layer=bn_layer)

    def forward(self, x):
        B, C, H, W = x.shape
        x_c = x
        response_c = self.blocks[0](x_c)

        x_h = x.permute(0, 2, 1, 3)
        response_h = self.blocks[1](x_h)
        response_h = response_h.permute(0, 2, 1, 3)

        x_w = x.permute(0, 3, 2, 1)
        response_w = self.blocks[2](x_w)
        response_w = response_w.permute(0, 3, 2, 1)

        y = response_c + response_h + response_w + x

        return y

class MultiViewNonLocalBlock3D(MultiViewNonLocalBlockBase):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super().__init__(4, in_channels, inter_channels=inter_channels, dimension=3, sub_sample=sub_sample, bn_layer=bn_layer)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x_c = x
        response_c = self.blocks[0](x_c)

        x_t = x.permute(0, 2, 1, 3, 4)
        response_t = self.blocks[1](x_t)
        response_t = response_t.permute(0, 2, 1, 3, 4)

        x_h = x.permute(0, 3, 2, 1, 4)
        response_h = self.blocks[2](x_h)
        response_h = response_h.permute(0, 3, 2, 1, 4)

        x_w = x.permute(0, 4, 2, 3, 1)
        response_w = self.blocks[3](x_w)
        response_w = response_w.permute(0, 4, 2, 3, 1)

        y = response_c + response_t + response_h + response_w + x

        return y


if __name__ == '__main__':
    import torch

    for (sub_sample, bn_layer) in [(True, True), (False, False), (True, False), (False, True)]:
        img = torch.zeros(2, 3, 20)
        net = MultiViewNonLocalBlock1D([3,20], sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img)
        print(out.size())

        img = torch.zeros(2, 3, 20, 20)
        net = MultiViewNonLocalBlock2D([3,20,20], sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img)
        print(out.size())

        img = torch.randn(2, 3, 8, 20, 20)
        net = MultiViewNonLocalBlock3D([3,8,20,20], sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img)
        print(out.size())