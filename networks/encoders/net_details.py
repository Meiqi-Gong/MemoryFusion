import torch.nn as nn

def actFunc(name):
    if name == 'relu':
        return nn.ReLU(inplace=False)
    elif name == 'leaky_relu':
        return nn.LeakyReLU(0.2, inplace=False)
    elif name == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f'Unknown activation function: {name}')

class ResBlock2D(nn.Module):
    """
    Residual block
    """

    def __init__(self, in_chs, activation='leaky_relu', batch_norm=False):
        super(ResBlock2D, self).__init__()
        op = []
        for i in range(1):
            op.append(nn.Conv2d(in_chs, in_chs, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            if batch_norm:
                op.append(nn.BatchNorm2d(in_chs))
            if i == 0:
                op.append(actFunc(activation))
        self.main_branch = nn.Sequential(*op)

    def forward(self, x):
        out = self.main_branch(x)
        out = out + x
        return out

def conv_resblock_2D(in_channels, out_channels, stride=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1)), actFunc('leaky_relu'), ResBlock2D(out_channels))

def upconv_resblock_2D(in_channels, out_channels, stride=1):
    return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1), actFunc('leaky_relu'), ResBlock2D(out_channels))

class Conv2D(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1))
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x
