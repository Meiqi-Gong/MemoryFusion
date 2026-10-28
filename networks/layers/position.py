import math
import torch
import torch.nn as nn

def generate_coord(x):
    _, _, h, w = x.size()
    device = x.device
    col = torch.arange(0, h, device=device)
    row = torch.arange(0, w, device=device)
    grid_h, grid_w = torch.meshgrid(col, row, indexing='ij')
    return (grid_h, grid_w)

class PositionEmbeddingSine(nn.Module):

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        grid_y, grid_x = generate_coord(x)
        y_embed = grid_y.unsqueeze(0).float()
        x_embed = grid_x.unsqueeze(0).float()
        if self.normalize:
            eps = 1e-06
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

def get_temporal_positional_encoding(max_sequence_len, channels, device, is_normalize=False, scale=2 * math.pi, is_debug=False):
    position = torch.arange(max_sequence_len, device=device)
    if is_normalize:
        position = position / position[-1] * scale
    position.unsqueeze_(1)
    div_term = 1.0 / 10000.0 ** (torch.arange(0, channels, 2, device=device).float() / channels)
    position_div_term = position * div_term
    temporal_encoding = torch.zeros((max_sequence_len, 1, 1, channels), device=device)
    temporal_encoding_sin = torch.sin(position_div_term)
    temporal_encoding_cos = torch.cos(position_div_term)
    temporal_encoding[:, 0, 0, 0::2] = temporal_encoding_sin
    temporal_encoding[:, 0, 0, 1::2] = temporal_encoding_cos
    return temporal_encoding
