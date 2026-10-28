import torch
import torch.nn.functional as F
from torch import nn

class GroupNorm1D(nn.Module):

    def __init__(self, indim, groups=8):
        super().__init__()
        self.gn = nn.GroupNorm(groups, indim)

    def forward(self, x):
        return self.gn(x.permute(1, 2, 0)).permute(2, 0, 1)

class GNActDWConv2d(nn.Module):

    def __init__(self, indim, gn_groups=32):
        super().__init__()
        self.gn = nn.GroupNorm(gn_groups, indim)
        self.conv = nn.Conv2d(indim, indim, 5, dilation=1, padding=2, groups=indim, bias=False)

    def forward(self, x, size_2d):
        h, w = size_2d
        _, bs, c = x.size()
        x = x.view(h, w, bs, c).permute(2, 3, 0, 1)
        x = self.gn(x)
        x = F.gelu(x)
        x = self.conv(x)
        x = x.view(bs, c, h * w).permute(2, 0, 1)
        return x

def seq_to_2d(tensor, size_2d):
    h, w = size_2d
    _, n, c = tensor.size()
    tensor = tensor.view(h, w, n, c).permute(2, 3, 0, 1).contiguous()
    return tensor

class DropPath(nn.Module):

    def __init__(self, drop_prob=None, batch_dim=0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.batch_dim = batch_dim

    def forward(self, x):
        return self.drop_path(x, self.drop_prob)

    def drop_path(self, x, drop_prob):
        if drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - drop_prob
        shape = [1 for _ in range(x.ndim)]
        shape[self.batch_dim] = x.shape[self.batch_dim]
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class DropOutLogit(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropOutLogit, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return self.drop_logit(x, self.drop_prob)

    def drop_logit(self, x, drop_prob):
        if drop_prob == 0.0 or not self.training:
            return x
        random_tensor = drop_prob + torch.rand(x.shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        mask = random_tensor * 100000000.0 if x.dtype == torch.float32 else random_tensor * 10000.0
        output = x - mask
        return output
