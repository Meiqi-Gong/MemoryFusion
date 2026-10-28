"""Paired video data for MemoryFusion training."""

import os
from pathlib import Path

import cv2
import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset

cv2.setNumThreads(0)
_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def _read_sequence_list(root, list_file, default_name):
    list_path = Path(list_file) if list_file is not None else Path(root) / default_name
    if not list_path.is_file():
        raise FileNotFoundError(f'Sequence list does not exist: {list_path}')
    names = [line.strip() for line in list_path.read_text(encoding='utf-8-sig').splitlines()
             if line.strip() and not line.lstrip().startswith('#')]
    if not names:
        raise ValueError(f'Sequence list is empty: {list_path}')
    return list(dict.fromkeys(names))


def _collect_aligned_frames(root, sequences, folders, min_frames, max_frames=None):
    """Validate every listed sequence, including files beyond an optional frame cap."""
    root = Path(root)
    result, errors = {}, []
    for sequence in sequences:
        routes, problems = [], []
        for folder in folders:
            directory = root / folder / sequence
            if not directory.is_dir():
                problems.append(f'missing directory {directory}')
                continue
            names = natsorted(p.name for p in directory.iterdir()
                              if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS)
            by_stem = {}
            duplicates = []
            for name in names:
                stem = Path(name).stem
                if stem in by_stem:
                    duplicates.append(stem)
                by_stem[stem] = name
            if duplicates:
                problems.append(f'{folder} has duplicate frame basenames: {duplicates[:5]}')
            if not names:
                problems.append(f'no image frames in {directory}')
            routes.append(by_stem)
        if not problems:
            reference = set(routes[0])
            for folder, frames in zip(folders[1:], routes[1:]):
                if set(frames) != reference:
                    missing = natsorted(reference - set(frames))[:5]
                    extra = natsorted(set(frames) - reference)[:5]
                    problems.append(f'{folder} frame basenames differ from {folders[0]} '
                                    f'(missing={missing}, extra={extra})')
            if len(reference) < min_frames:
                problems.append(f'{len(reference)} frames; at least {min_frames} required')
        if problems:
            errors.append(f'{sequence}: ' + '; '.join(problems))
            continue
        stems = natsorted(routes[0])
        if max_frames is not None:
            stems = stems[:max_frames]
        result[sequence] = tuple([frames[stem] for stem in stems] for frames in routes)
    if errors:
        raise ValueError('Invalid video sequences:\n' + '\n'.join(errors))
    return result


def _read_modalities(paths, sequence, expected_shape=None):
    images = []
    for path in paths:
        image = cv2.imread(os.fspath(path))
        if image is None:
            raise ValueError(f'Cannot read image in sequence {sequence}: {path}')
        if expected_shape is None:
            expected_shape = image.shape
        if image.shape != expected_shape:
            raise ValueError(f'Image dimensions differ in sequence {sequence}: '
                             f'{path} has {image.shape}, expected {expected_shape}')
        images.append(np.array(image, dtype=np.float32))
    return images


class M3SVDTrain(Dataset):
    def __init__(self, vis_root, ir_root, imglistdic, transform=None, rgb=True,
                 repeat_time=1, rand_gap=3, seq_len=5, rand_reverse=True,
                 dynamic_merge=True, enable_prev_frame=False, merge_prob=0.3,
                 max_obj_n=10, ignore_thresh=1.0, ignore_in_merge=False):
        if seq_len < 2:
            raise ValueError('Training seq_len must be at least 2')
        if repeat_time < 1 or int(repeat_time) != repeat_time:
            raise ValueError('repeat_time must be a positive integer')
        if not imglistdic:
            raise ValueError('Training dataset contains no sequences')
        errors = []
        for name, routes in imglistdic.items():
            if len(routes) != 4 or len({len(route) for route in routes}) != 1:
                errors.append(f'{name}: four aligned modality lists are required')
            elif len(routes[0]) < seq_len:
                errors.append(f'{name}: {len(routes[0])} frames; seq_len={seq_len} required')
        if errors:
            raise ValueError('Invalid training sequences:\n' + '\n'.join(errors))
        self.vis_root, self.ir_root = os.fspath(vis_root), os.fspath(ir_root)
        self.rand_gap, self.seq_len = rand_gap, seq_len
        self.rand_reverse, self.repeat_time = rand_reverse, repeat_time
        self.transform, self.rgb = transform, rgb
        self.imglistdic = imglistdic
        self.seqs = list(imglistdic)
        print('Video Num: {} X {}'.format(len(self.seqs), self.repeat_time))

    def __len__(self):
        return int(len(self.seqs) * self.repeat_time)

    def reverse_seq(self, *lists):
        if np.random.randint(2) == 1:
            return [values[::-1] for values in lists]
        return lists

    def sample_sequence(self, idx):
        sequence = self.seqs[idx % len(self.seqs)]
        routes = self.imglistdic[sequence]
        frame_num = len(routes[0])
        if self.rand_reverse:
            routes = self.reverse_seq(*routes)
        # Keep the original contiguous sampling distribution and random-number order.
        start = np.random.randint(0, frame_num - self.seq_len + 1)
        images = [[], [], [], []]
        roots = (self.vis_root, self.ir_root, self.vis_root, self.ir_root)
        folders = ('VI', 'IR_en', 'VI_blur_new', 'IR_noise2')
        for index in range(start, start + self.seq_len):
            paths = [os.path.join(root, folder, sequence, names[index])
                     for root, folder, names in zip(roots, folders, routes)]
            frames = _read_modalities(paths, sequence,
                                      images[0][0].shape if images[0] else None)
            for values, image in zip(images, frames):
                values.append(image[:, :, [2, 1, 0]] if self.rgb else image)
        sample = dict(zip(('vis_imgs', 'ir_imgs', 'vis_blur_imgs', 'ir_noise_imgs'), images))
        sample['meta'] = {'seq_name': sequence, 'frame_num': frame_num}
        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except Exception as exc:
                raise ValueError(f'Transform failed for sequence {sequence}: {exc}') from exc
        return sample

    def __getitem__(self, idx):
        return self.sample_sequence(idx)


class M3SVD_Train(M3SVDTrain):
    def __init__(self, split=('train',), transform=None, rgb=True, repeat_time=1,
                 rand_gap=3, seq_len=5, rand_reverse=True, dynamic_merge=True,
                 enable_prev_frame=False, max_obj_n=10, merge_prob=0.3,
                 ignore_thresh=1.0, ignore_in_merge=False, root=None, list_file=None):
        # split and the VOS-related arguments remain accepted for config compatibility.
        root = root if root is not None else os.environ.get('MEMORYFUSION_TRAIN_ROOT', 'datasets/train')
        sequences = _read_sequence_list(root, list_file, 'train.txt')
        frames = _collect_aligned_frames(root, sequences,
                                         ('VI', 'IR_en', 'VI_blur_new', 'IR_noise2'),
                                         min_frames=max(2, seq_len))
        super().__init__(root, root, frames, transform=transform, rgb=rgb,
                         repeat_time=repeat_time, rand_gap=rand_gap, seq_len=seq_len,
                         rand_reverse=rand_reverse, dynamic_merge=dynamic_merge,
                         enable_prev_frame=enable_prev_frame, merge_prob=merge_prob,
                         max_obj_n=max_obj_n, ignore_thresh=ignore_thresh,
                         ignore_in_merge=ignore_in_merge)
