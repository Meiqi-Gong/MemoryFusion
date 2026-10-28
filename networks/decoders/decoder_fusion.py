import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.encoders.net_details import upconv_resblock_2D, conv_resblock_2D
from utils.tensor import lbc_2_bchw
from loss.loss import SoftAdaptiveSmoothing

class decoder(nn.Module):

    def __init__(self, in_dim, out_dim, curr_dims=[32 * 2, 64 * 2, 256]):
        super().__init__()
        self.combine_lstt = conv_resblock_2D(in_dim, in_dim)
        self.up1 = upconv_resblock_2D(curr_dims[-1] * 2, curr_dims[-2])
        self.up1_out = nn.Conv2d(curr_dims[-1] * 2, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.up2 = upconv_resblock_2D(curr_dims[-2] * 2, curr_dims[-3])
        self.up2_out = nn.Conv2d(curr_dims[-2] * 2, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.final = conv_resblock_2D(curr_dims[-3] * 2, out_dim)
        self.smoother1 = SoftAdaptiveSmoothing(alpha_motion=0.8, alpha_static=0.5, blur_kernel_size=5, device='cuda')
        self.smoother2 = SoftAdaptiveSmoothing(alpha_motion=0.8, alpha_static=0.5, blur_kernel_size=5, device='cuda')
        self.smoother3 = SoftAdaptiveSmoothing(alpha_motion=0.8, alpha_static=0.5, blur_kernel_size=5, device='cuda')
        self._init_weight()

    def forward(self, curr_embbeds, lstt_embbeds, vis, ir):
        size = curr_embbeds[-1].shape[-2:]
        curr_embbeds[-1] = self.smoother1(curr_embbeds[-1], vis)
        curr_embbeds[-2] = self.smoother2(curr_embbeds[-2], vis)
        curr_embbeds[-3] = self.smoother3(curr_embbeds[-3], vis)
        lstt_embbeds = [lbc_2_bchw(lstt, size) for lstt in lstt_embbeds]
        lstt_feat = self.combine_lstt(torch.cat(lstt_embbeds, 1))
        x = torch.cat([lstt_feat, curr_embbeds[-1]], 1)
        x1_out = self.up1_out(x)
        +F.interpolate(vis, scale_factor=0.25, mode='bilinear', align_corners=False)
        x = torch.cat([self.up1(x), curr_embbeds[-2]], 1)
        x2_out = self.up2_out(x)
        +F.interpolate(vis, scale_factor=0.5, mode='bilinear', align_corners=False)
        x = torch.cat([self.up2(x), curr_embbeds[-3]], 1)
        x = self.final(x) + vis
        return (x1_out, x2_out, x, lstt_feat, curr_embbeds)

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class TemporalDetailRefinement(nn.Module):
    """
    改进版 TemporalDetailRefinement
    - 高频增强 + 残差连接
    - 3D 卷积
    - Temporal Attention 平滑帧间 flickering
    - 可选 Channel Attention 增强细节
    """

    def __init__(self, in_channels=9, out_channels=[16, 32, 16], kernel_size=(3, 3, 3), stride=1, padding=1, num_layers=3, use_channel_attention=True):
        super().__init__()
        self.use_channel_attention = use_channel_attention
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv3d(in_channels if i == 0 else out_channels[i - 1], out_channels[i], kernel_size=kernel_size, stride=stride, padding=padding))
            g = min(8, out_channels[i])
            layers.append(nn.GroupNorm(num_groups=g, num_channels=out_channels[i]))
            layers.append(nn.ReLU(inplace=True))
        self.refine = nn.Sequential(*layers)
        self.head = nn.Conv3d(out_channels[-1], 3, kernel_size=3, padding=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, mean=0.0, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, vis, ir):
        """
        x: [B, C, T, H, W]
        """
        x_input = torch.cat([x, vis, ir], dim=1)
        feat = self.refine(x_input)
        residual = self.head(feat)
        out = x + residual
        return out
