from math import exp
from torch import nn as nn
from torch.nn import functional as F
import torch
from einops import rearrange

def laplacian_var_noise_score(img, alpha=0.7):
    """
    计算 Laplacian Variance - α * 高频噪声比例
    img: [B, C, H, W] in [0, 1] or [0, 255]
    alpha: 噪声惩罚系数
    return: [B] 分数（越高越好）
    """
    if img.size(1) == 3:
        img = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
    lap = F.conv2d(img, lap_kernel, padding=1)
    lap_var = lap.var(dim=[1, 2, 3])
    low_pass_kernel = torch.ones((1, 1, 5, 5), device=img.device) / 25.0
    smooth = F.conv2d(img, low_pass_kernel, padding=2)
    high_freq = img - smooth
    high_freq_energy = (high_freq ** 2).mean(dim=[1, 2, 3])
    total_energy = (img ** 2).mean(dim=[1, 2, 3]) + 1e-08
    high_freq_ratio = high_freq_energy / total_energy
    score = lap_var - alpha * high_freq_ratio
    return score

class Fusion_singleloss(nn.Module):

    def __init__(self):
        super(Fusion_singleloss, self).__init__()
        print('Building Fusion_loss() as loss function ~')
        self.loss_func_ssim = L_SSIM(window_size=13, size_average=False)
        self.loss_func_grad = L_Gradient_Max()
        self.loss_func_max = L_Intensity_Max()
        self.loss_func_consist = L_Intensity_Consist()
        self.loss_func_color = L_Color()

    def forward(self, img_f, img_ir, img_vi, t_num, b, int_weight=10, consist_weight=1, ssim_ir_weight=1, ssim_weight=1, ir_compose=1, color_weight=20, text_weight=10, temp_weight=2, regular=False):
        b, c, h, w = img_f.shape
        Y_vi, Cb_vi, Cr_vi = RGB2YCrCb(img_vi.float())
        Y_ir, _, _ = RGB2YCrCb(img_ir.float())
        Y_f, Cb_f, Cr_f = RGB2YCrCb(img_f.float())
        loss_ssim = ssim_weight * (self.loss_func_ssim(Y_vi, Y_f) + ssim_ir_weight * self.loss_func_ssim(Y_ir, Y_f))
        loss_max = int_weight * self.loss_func_max(Y_f, Y_vi, Y_ir, reduction='none')
        loss_color = color_weight * self.loss_func_color(Cb_f, Cr_f, Cb_vi, Cr_vi, reduction='none')
        loss_text = text_weight * self.loss_func_grad(self.rgb2gray(Y_f), self.rgb2gray(Y_vi), self.rgb2gray(Y_ir), regular, reduction='none')
        total_loss = loss_ssim + loss_color + loss_text + loss_max
        return {'loss_intensity_max': loss_max, 'loss_color': loss_color, 'loss_grad': loss_text, 'loss_ssim': loss_ssim, 'loss_fusion': total_loss}

    def rgb2gray(self, image):
        b, c, h, w = image.size()
        if c == 1:
            return image
        image_gray = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        image_gray = image_gray.unsqueeze(dim=1)
        return image_gray

class Fusion_allloss(nn.Module):

    def __init__(self):
        super(Fusion_allloss, self).__init__()
        print('Building Fusion_loss() as loss function ~')
        self.loss_func_ssim = L_SSIM(window_size=13)
        self.loss_func_grad = L_Gradient_Max()
        self.loss_func_max = L_Intensity_Max()
        self.loss_func_consist = L_Intensity_Consist()
        self.loss_func_color = L_Color()

    def cos_consist(self, img_f, img_ir):
        num_frames = img_f.shape[1]
        diff_loss = 0
        for t in range(1, num_frames):
            output_diff = img_f[:, t] - img_f[:, t - 1]
            input_diff = img_ir[:, t] - img_ir[:, t - 1]
            cosine_sim = F.cosine_similarity(output_diff, input_diff, dim=1)
            diff_loss += (1 - cosine_sim).mean()
        diff_loss /= num_frames - 1
        return diff_loss

    def forward(self, img_f, img_ir, img_vi, t_num, b, int_weight=10, consist_weight=1, ssim_ir_weight=1, ssim_weight=1, ir_compose=1, color_weight=20, text_weight=10, temp_weight=2, regular=False, temporal='first_stage'):
        bt, c, h, w = img_f.shape
        Y_vi, Cb_vi, Cr_vi = RGB2YCrCb(img_vi.float())
        Y_ir, _, _ = RGB2YCrCb(img_ir.float())
        Y_f, Cb_f, Cr_f = RGB2YCrCb(img_f.float())
        if temporal == 'first_stage':
            loss_temp_consist = temp_weight * self.cos_consist(Y_f.view(t_num, b, -1).permute(1, 0, 2), Y_ir.view(t_num, b, -1).permute(1, 0, 2))
        elif temporal == 'second_stage':
            Y_f_t = rearrange(Y_f, '(t b) c h w -> b c t h w', b=b)
            Y_ir_t = rearrange(Y_ir, '(t b) c h w -> b c t h w', b=b)
            loss_temp_consist = temp_weight * self.cos_consist(Y_f_t.squeeze(1), Y_ir_t.squeeze(1))
        loss_ssim = ssim_weight * (self.loss_func_ssim(Y_vi, Y_f) + ssim_ir_weight * self.loss_func_ssim(Y_ir, Y_f))
        loss_max = int_weight * self.loss_func_max(Y_f, Y_vi, Y_ir)
        loss_color = color_weight * self.loss_func_color(Cb_f, Cr_f, Cb_vi, Cr_vi)
        loss_text = text_weight * self.loss_func_grad(Y_f, Y_vi, Y_ir, regular)
        total_loss = loss_ssim + loss_color + loss_text + loss_temp_consist + loss_max
        return {'loss_intensity_max': loss_max, 'loss_color': loss_color, 'loss_grad': loss_text, 'loss_temporary_consist': loss_temp_consist, 'loss_ssim': loss_ssim, 'loss_fusion': total_loss}

    def rgb2gray(self, image):
        b, c, h, w = image.size()
        if c == 1:
            return image
        image_gray = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        image_gray = image_gray.unsqueeze(dim=1)
        return image_gray

class Fusion_loss_stage2(nn.Module):

    def __init__(self):
        super(Fusion_loss_stage2, self).__init__()
        print('Building Fusion_loss() as loss function ~')
        self.loss_func_ssim = L_SSIM(window_size=13)
        self.loss_func_grad = L_Gradient_Max()
        self.loss_func_consist = L_Intensity_Consist()

    def cos_consist(self, img_f, img_ir):
        num_frames = img_f.shape[1]
        diff_loss = 0
        for t in range(1, num_frames):
            output_diff = img_f[:, t] - img_f[:, t - 1]
            input_diff = img_ir[:, t] - img_ir[:, t - 1]
            cosine_sim = F.cosine_similarity(output_diff, input_diff, dim=1)
            diff_loss += (1 - cosine_sim).mean()
        diff_loss /= num_frames - 1
        return diff_loss

    def forward(self, img_f, img_ir, img_vi, t_num, b, int_weight=10, consist_weight=1, ssim_ir_weight=1, ssim_weight=1, ir_compose=1, color_weight=20, text_weight=10, temp_weight=2, regular=False):
        bt, c, h, w = img_f.shape
        Y_vi, Cb_vi, Cr_vi = RGB2YCrCb(img_vi.float())
        Y_ir, _, _ = RGB2YCrCb(img_ir.float())
        Y_f, Cb_f, Cr_f = RGB2YCrCb(img_f.float())
        loss_ssim = ssim_weight * (self.loss_func_ssim(Y_vi, Y_f) + ssim_ir_weight * self.loss_func_ssim(Y_ir, Y_f))
        loss_text = text_weight * self.loss_func_grad(Y_f, Y_vi, Y_ir, regular)
        total_loss = loss_ssim + loss_text
        return {'loss_grad': loss_text, 'loss_ssim': loss_ssim, 'loss_fusion': total_loss}

    def rgb2gray(self, image):
        b, c, h, w = image.size()
        if c == 1:
            return image
        image_gray = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        image_gray = image_gray.unsqueeze(dim=1)
        return image_gray

class L_SSIM(torch.nn.Module):

    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(L_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        _, channel_2, _, _ = img2.size()
        if channel != channel_2 and channel == 1:
            img1 = torch.concat([img1, img1, img1], dim=1)
            channel = 3
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, val_range=1.0, eps=1e-08):
    padd = window_size // 2
    _, channel, height, width = img1.size()
    if window is None:
        window = create_window(window_size, channel=channel).to(img1.device).type(img1.dtype)
    window = window.to(img1.device).type(img1.dtype)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2
    ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + eps)
    if size_average:
        return 1 - ssim_map.mean()
    else:
        return 1 - ssim_map.mean(1).mean(1).mean(1)

class L_Gradient_Max(nn.Module):

    def __init__(self):
        super(L_Gradient_Max, self).__init__()
        self.grad_operator = Gradient()

    def forward(self, img_f, img_vi, img_ir, regular, reduction='mean'):
        grad_vi_x, grad_vi_y = self.grad_operator(img_vi)
        grad_ir_x, grad_ir_y = self.grad_operator(img_ir)
        grad_f_x, grad_f_y = self.grad_operator(img_f)
        if regular:
            loss = 5 * (torch.abs(grad_vi_x - grad_ir_x) * F.l1_loss(grad_f_x, torch.max(grad_vi_x, grad_ir_x), reduction='none') + torch.abs(grad_vi_y - grad_ir_y) * F.l1_loss(grad_f_y, torch.max(grad_vi_y, grad_ir_y), reduction='none'))
        else:
            loss = F.l1_loss(grad_f_x, torch.max(grad_vi_x, grad_ir_x), reduction='none') + F.l1_loss(grad_f_y, torch.max(grad_vi_y, grad_ir_y), reduction='none')
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'none':
            return loss.mean(1).mean(1).mean(1)

class Gradient(nn.Module):

    def __init__(self):
        super(Gradient, self).__init__()
        self.sobel_x = nn.Parameter(torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3), requires_grad=False)
        self.sobel_y = nn.Parameter(torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3), requires_grad=False)

    def forward(self, img):
        img = F.pad(img, (1, 1, 1, 1), mode='replicate')
        grad_x = F.conv2d(img, self.sobel_x)
        grad_y = F.conv2d(img, self.sobel_y)
        return (torch.abs(grad_x), torch.abs(grad_y))

class L_Intensity_Consist(nn.Module):

    def __init__(self):
        super(L_Intensity_Consist, self).__init__()

    def forward(self, img_f, img_vi, img_ir, ir_compose, consist_mode='l1', reduction='mean'):
        loss_func = F.mse_loss if consist_mode == 'l2' else F.l1_loss
        if reduction == 'mean':
            return loss_func(img_vi, img_f) + ir_compose * loss_func(img_ir, img_f)
        elif reduction == 'none':
            return loss_func(img_vi, img_f, reduction='none').mean(1).mean(1).mean(1) + ir_compose * loss_func(img_ir, img_f, reduction='none').mean(1).mean(1).mean(1)

class L_Color(nn.Module):

    def __init__(self):
        super(L_Color, self).__init__()

    def forward(self, Cb_f, Cr_f, Cb_vi, Cr_vi, reduction='mean'):
        if reduction == 'mean':
            return F.l1_loss(Cb_f, Cb_vi) + F.l1_loss(Cr_f, Cr_vi)
        elif reduction == 'none':
            return F.l1_loss(Cb_f, Cb_vi, reduction='none').mean(1).mean(1).mean(1) + F.l1_loss(Cr_f, Cr_vi, reduction='none').mean(1).mean(1).mean(1)

class L_Intensity_Max(nn.Module):

    def __init__(self):
        super(L_Intensity_Max, self).__init__()

    def forward(self, img_f, img_vi, img_ir, reduction='mean'):
        img_max = torch.max(img_vi, img_ir)
        if reduction == 'mean':
            return F.l1_loss(img_max, img_f)
        elif reduction == 'none':
            return F.l1_loss(img_max, img_f, reduction='none').mean(1).mean(1).mean(1)

def RGB2YCrCb(rgb_image, with_CbCr=False):
    """
    Convert RGB format to YCrCb format.
    Used in the intermediate results of the color space conversion, because the default size of rgb_image is [B, C, H, W].
    :param rgb_image: image data in RGB format
    :param with_CbCr: boolean flag to determine if Cb and Cr channels should be returned
    :return: Y, CbCr (if with_CbCr is True), otherwise Y, Cb, Cr
    """
    R, G, B = (rgb_image[:, 0:1, :], rgb_image[:, 1:2, :], rgb_image[:, 2:3, :])
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.169 * R - 0.331 * G + 0.5 * B + 128 / 255.0
    Cr = 0.5 * R - 0.419 * G - 0.081 * B + 128 / 255.0
    Y, Cb, Cr = (Y.clamp(0.0, 1.0), Cb.clamp(0.0, 1.0), Cr.clamp(0.0, 1.0))
    if with_CbCr:
        return (Y, torch.cat([Cb, Cr], dim=1))
    return (Y, Cb, Cr)

class SoftAdaptiveSmoothing:

    def __init__(self, alpha_motion=0.8, alpha_static=0.5, blur_kernel_size=5, device='cuda'):
        """
        alpha_motion: 运动区域的平滑系数（越大越依赖当前帧）
        alpha_static: 静止区域的平滑系数（越小越依赖历史帧）
        blur_kernel_size: 掩码平滑的卷积核大小
        """
        self.alpha_motion = alpha_motion
        self.alpha_static = alpha_static
        self.blur_kernel_size = blur_kernel_size
        self.device = device
        self.prev_feature = None
        self.prev_vis = None
        self.blur_weight = torch.ones(1, 1, blur_kernel_size, blur_kernel_size, device=device) / blur_kernel_size ** 2

    def _compute_motion_mask(self, current_vis, prev_vis, thresh=0.05, sharpness=10):
        """
        基于特征差分生成软掩码
        current_feature, prev_feature: [B, C, H, W]
        thresh: 区分运动和静止的差分阈值
        sharpness: sigmoid 的陡峭程度
        """
        diff = torch.abs(current_vis - prev_vis).mean(dim=1, keepdim=True)
        diff_norm = diff / (diff.max() + 1e-06)
        motion_mask_soft = torch.sigmoid((diff_norm - thresh) * sharpness)
        return motion_mask_soft

    def _blur_mask(self, mask):
        """对掩码进行均值模糊（避免硬边界伪影）"""
        return F.conv2d(mask, self.blur_weight, padding=self.blur_kernel_size // 2)

    def __call__(self, current_feature, current_vis):
        """
        输入当前帧特征，输出平滑后的特征
        current_feature: [B, C, H, W]
        """
        if self.prev_feature is None:
            smoothed = current_feature
        else:
            motion_mask_soft = self._compute_motion_mask(current_vis, self.prev_vis)
            motion_mask_blur = self._blur_mask(motion_mask_soft)
            alpha_map = self.alpha_motion * motion_mask_blur + self.alpha_static * (1 - motion_mask_blur)
            alpha_map = F.interpolate(alpha_map, size=current_feature.shape[2:], mode='bilinear', align_corners=False)
            smoothed = alpha_map * current_feature + (1 - alpha_map) * self.prev_feature
        self.prev_feature = smoothed.detach()
        self.prev_vis = current_vis.detach()
        return smoothed
