from networks.encoders.net_details import Conv2D, conv_resblock_2D
import torch.nn as nn
import torch
import math

class Encoder(nn.Module):

    def __init__(self, in_dims, nf=32):
        super(Encoder, self).__init__()
        self.conv0 = Conv2D(in_dims, nf)
        self.conv1 = conv_resblock_2D(nf, nf)
        self.conv2 = conv_resblock_2D(nf, 2 * nf, stride=2)
        self.conv3 = conv_resblock_2D(2 * nf, 4 * nf, stride=2)

    def forward(self, x):
        x_ = self.conv0(x)
        f1 = self.conv1(x_)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        return [f1, f2, f3]

class Encoder_and_Fusion(nn.Module):

    def __init__(self, in_dims, nf=32 * 2):
        super(Encoder_and_Fusion, self).__init__()
        self.encoder_x = Encoder(in_dims, nf)
        self.encoder_y = Encoder(in_dims, nf)
        self.fuse_1 = conv_resblock_2D(2 * nf, nf)
        self.fuse_2 = conv_resblock_2D(4 * nf, 2 * nf)
        self.fuse_3 = conv_resblock_2D(8 * nf, 4 * nf)
        self._init_weight()

    def forward(self, x, y):
        fx = self.encoder_x(x)
        fy = self.encoder_y(y)
        f1 = self.fuse_1(torch.cat([fx[0], fy[0]], 1))
        f2 = self.fuse_2(torch.cat([fx[1], fy[1]], 1))
        f3 = self.fuse_3(torch.cat([fx[2], fy[2]], 1))
        return [f1, f2, f3]

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
