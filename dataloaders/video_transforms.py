"""Training and evaluation transforms used by MemoryFusion."""

import random

import cv2
import numpy as np
import torch


class PairedRandomCrop(object):
    """Random crop for paired GT IR and VI images, including degraded versions."""

    def __init__(self, patch_size=256):
        """
        Args:
            patch_size (int): Patch size to crop from images.
        """
        self.patch_size = patch_size

    def __call__(self, sample):
        """
        Args:
            sample (dict): A sample containing at least 'ir_imgs', 'vis_imgs',
                           and optionally 'ir_noise_imgs', 'vis_blur_imgs'.

        Returns:
            dict: The same sample with cropped images.
        """
        ir_imgs = sample['ir_imgs']
        vis_imgs = sample['vis_imgs']
        ir_noise_imgs = sample['ir_noise_imgs']
        vis_blur_imgs = sample['vis_blur_imgs']

        def ensure_list(x):
            return x if isinstance(x, list) else [x]

        ir_imgs = ensure_list(ir_imgs)
        vis_imgs = ensure_list(vis_imgs)
        if ir_noise_imgs is not None:
            ir_noise_imgs = ensure_list(ir_noise_imgs)
        if vis_blur_imgs is not None:
            vis_blur_imgs = ensure_list(vis_blur_imgs)

        input_type = 'Tensor' if torch.is_tensor(ir_imgs[0]) else 'Numpy'

        if input_type == 'Tensor':
            h, w = ir_imgs[0].size()[-2:]
        else:
            h, w = ir_imgs[0].shape[0:2]

        if h < self.patch_size or w < self.patch_size:
            raise ValueError(f'Image size ({h}, {w}) smaller than patch size ({self.patch_size}, {self.patch_size}).')

        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)

        def crop(imgs):
            if imgs is None:
                return None
            if input_type == 'Tensor':
                return [v[:, :, top:top + self.patch_size, left:left + self.patch_size] for v in imgs]
            else:
                return [v[top:top + self.patch_size, left:left + self.patch_size, ...] for v in imgs]

        ir_imgs = crop(ir_imgs)
        vis_imgs = crop(vis_imgs)
        ir_noise_imgs = crop(ir_noise_imgs)
        vis_blur_imgs = crop(vis_blur_imgs)

        def unwrap(imgs):
            return imgs[0] if len(imgs) == 1 else imgs

        sample['ir_imgs'] = unwrap(ir_imgs)
        sample['vis_imgs'] = unwrap(vis_imgs)
        if ir_noise_imgs is not None:
            sample['ir_noise_imgs'] = unwrap(ir_noise_imgs)
        if vis_blur_imgs is not None:
            sample['vis_blur_imgs'] = unwrap(vis_blur_imgs)

        return sample


class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""
    def __init__(self, prob):
        self.p = prob

    def __call__(self, sample):

        if random.random() < self.p:
            for elem in sample.keys():
                if 'meta' in elem:
                    continue
                if elem == 'vis_imgs' or elem == 'ir_imgs' or elem == 'vis_blur_imgs' or elem == 'ir_noise_imgs':
                    new_tmp = []
                    for tmp_ in sample[elem]:
                        tmp_ = cv2.flip(tmp_, flipCode=1)
                        new_tmp.append(tmp_)
                    sample[elem] = new_tmp
                else:
                    tmp = sample[elem]
                    tmp = cv2.flip(tmp, flipCode=1)
                    sample[elem] = tmp

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):

        for elem in sample.keys():
            if 'meta' in elem:
                continue
            tmp = sample[elem]

            if elem == 'vis_imgs' or elem == 'ir_imgs' or elem == 'vis_blur_imgs' or elem == 'ir_noise_imgs':
                new_tmp = []
                for tmp_ in tmp:
                    if tmp_.ndim == 2:
                        tmp_ = tmp_[:, :, np.newaxis]
                        tmp_ = tmp_.transpose((2, 0, 1))
                        new_tmp.append(torch.from_numpy(tmp_).int())
                    else:
                        tmp_ = tmp_ / 255.
                        tmp_ = tmp_.transpose((2, 0, 1))
                        new_tmp.append(torch.from_numpy(tmp_))
                tmp = new_tmp
            else:
                if tmp.ndim == 2:
                    tmp = tmp[:, :, np.newaxis]
                    tmp = tmp.transpose((2, 0, 1))
                    tmp = torch.from_numpy(tmp).int()
                else:
                    tmp = tmp / 255.
                    tmp = tmp.transpose((2, 0, 1))
                    tmp = torch.from_numpy(tmp)
            sample[elem] = tmp

        return sample
