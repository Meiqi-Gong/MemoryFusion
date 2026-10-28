"""Aligned paired video sequences for MemoryFusion evaluation."""

import os

from torch.utils.data import Dataset

from .train_datasets import _collect_aligned_frames, _read_modalities, _read_sequence_list


class M3SVDTest(Dataset):
    def __init__(self, image_root, imglistdic, rgb=True, transform=None, resolution=None):
        if not imglistdic:
            raise ValueError('Test dataset contains no sequences')
        errors = []
        for sequence, routes in imglistdic.items():
            if len(routes) != 4 or len({len(route) for route in routes}) != 1:
                errors.append(f'{sequence}: four aligned modality lists are required')
            elif len(routes[0]) < 2:
                errors.append(f'{sequence}: at least 2 frames are required')
        if errors:
            raise ValueError('Invalid test sequences:\n' + '\n'.join(errors))
        self.image_root = os.fspath(image_root)
        self.imglistdic, self.seqs = imglistdic, list(imglistdic)
        self.num_frame = len(imglistdic)
        self.transform, self.rgb, self.resolution = transform, rgb, resolution

    def __len__(self):
        return len(self.seqs)

    def read_image(self, vislist, irlist, vis_blurlist, ir_noiselist, seqname):
        images = [[], [], [], []]
        folders = ('VI_en', 'IR_en', 'VI_blur_new', 'IR_noise2')
        for names in zip(vislist, irlist, vis_blurlist, ir_noiselist):
            paths = [os.path.join(self.image_root, folder, seqname, name)
                     for folder, name in zip(folders, names)]
            frames = _read_modalities(paths, seqname,
                                      images[0][0].shape if images[0] else None)
            for values, image in zip(images, frames):
                values.append(image[:, :, [2, 1, 0]] if self.rgb else image)
        return images

    def __getitem__(self, idx):
        sequence = self.seqs[idx]
        routes = self.imglistdic[sequence]
        images = self.read_image(*routes, sequence)
        sample = dict(zip(('vis_imgs', 'ir_imgs', 'vis_blur_imgs', 'ir_noise_imgs'), images))
        sample['meta'] = {'seq_name': sequence, 'frame_num': len(routes[0]),
                          'flip': False, 'img_name': routes[0]}
        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except Exception as exc:
                raise ValueError(f'Transform failed for sequence {sequence}: {exc}') from exc
        return sample


class M3SVD_Test(M3SVDTest):
    def __init__(self, split=('test',), transform=None, rgb=True, is_oracle=False,
                 test_frame=None, root=None, list_file=None, max_frames=None):
        # test_frame is the legacy name for max_frames; None reads the whole sequence.
        if test_frame is not None:
            if max_frames is not None and max_frames != test_frame:
                raise ValueError('test_frame and max_frames specify different limits')
            max_frames = test_frame
        if max_frames is not None and (not isinstance(max_frames, int) or max_frames < 2):
            raise ValueError('max_frames must be None or an integer of at least 2')
        root = root if root is not None else os.environ.get('MEMORYFUSION_TEST_ROOT', 'datasets/test')
        sequences = _read_sequence_list(root, list_file, 'test_all.txt')
        frames = _collect_aligned_frames(root, sequences,
                                         ('VI_en', 'IR_en', 'VI_blur_new', 'IR_noise2'),
                                         min_frames=2, max_frames=max_frames)
        self.is_oracle = is_oracle
        super().__init__(root, frames, transform=transform, rgb=rgb)
