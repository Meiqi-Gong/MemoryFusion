"""Single-GPU evaluation with strict checkpoints and optional Stage2 refinement."""

from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloaders.eval_datasets import M3SVD_Test
import dataloaders.video_transforms as tr
from networks.decoders.decoder_fusion import TemporalDetailRefinement
from networks.engines import build_engine
from networks.models import build_vos_model
from utils.checkpoint import load_state_dict_strict as load_checkpoint_strict


class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.gpu = cfg.TEST_GPU_ID
        if not torch.cuda.is_available():
            raise RuntimeError("MemoryFusion evaluation requires a CUDA GPU")
        if self.gpu >= torch.cuda.device_count():
            raise ValueError(f"GPU {self.gpu} is not available")
        torch.cuda.set_device(self.gpu)
        self.device = torch.device("cuda", self.gpu)

        self.model = build_vos_model(cfg.MODEL_VOS, cfg).to(self.device)
        load_checkpoint_strict(self.model, cfg.TEST_CKPT_PATH_stage1)
        self.model.eval()
        self.eval_engine = build_engine(
            cfg.MODEL_ENGINE,
            "eval",
            aot_model=self.model,
            gpu_id=self.gpu,
            long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP,
        ).eval()

        self.model_refine = None
        if getattr(cfg, "TEST_WITH_STAGE2", False):
            self.model_refine = TemporalDetailRefinement()
            load_checkpoint_strict(self.model_refine, cfg.TEST_CKPT_PATH_stage2)
            self.model_refine.to(self.device).eval()

        self.result_root = Path(cfg.OUTPUT_PATH)
        self.dataset = M3SVD_Test(
            transform=transforms.Compose([tr.ToTensor()]),
            root=getattr(cfg, "DATA_TEST_ROOT", None),
            list_file=getattr(cfg, "DATA_TEST_LIST", None),
            max_frames=getattr(cfg, "TEST_MAX_FRAMES", 200),
        )
        if not len(self.dataset):
            raise ValueError("No test sequences were found")
        print(f"Evaluating {len(self.dataset)} sequences on {self.device}; "
              f"Stage2 {'enabled' if self.model_refine is not None else 'disabled'}")
        print(f"Output directory: {self.result_root}")

    def refine(self, fused, visible, infrared):
        """Refine full sequences, or explicitly selected memory-saving windows."""
        window = getattr(self.cfg, "TEST_REFINE_WINDOW", 0)
        if window == 0:
            return self.model_refine(fused, visible, infrared)

        # The halo covers four temporal convolutions. GroupNorm still uses each
        # window's statistics, so windowed output differs from full-sequence output.
        halo = 4
        if window <= 2 * halo:
            raise ValueError("Stage2 window must be zero or greater than eight")
        core_size = window - 2 * halo
        frame_count = fused.shape[2]
        result = torch.empty(fused.shape, dtype=fused.dtype, device="cpu")
        for start in range(0, frame_count, core_size):
            stop = min(start + core_size, frame_count)
            indices = [
                min(max(index, 0), frame_count - 1)
                for index in range(start - halo, start - halo + window)
            ]
            chunk = self.model_refine(
                fused[:, :, indices],
                visible[:, :, indices],
                infrared[:, :, indices],
            )
            core = chunk[:, :, halo:halo + stop - start]
            if not torch.isfinite(core).all():
                raise FloatingPointError("Non-finite Stage2 prediction")
            result[:, :, start:stop] = core.detach().cpu()
        return result

    def save_results(self, results, sequence):
        if not torch.isfinite(results).all():
            raise FloatingPointError(f"Non-finite output in sequence {sequence}")
        frames = (results.squeeze(0).permute(1, 2, 3, 0) * 255).clamp(0, 255).byte().cpu().numpy()
        destination = self.result_root / sequence
        destination.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(frames, start=1):
            path = destination / f"{index}.jpg"
            if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                raise OSError(f"Could not write output image: {path}")

    def evaluating(self):
        loader = DataLoader(self.dataset, batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=True)
        for sample in loader:
            sequence = sample["meta"]["seq_name"][0]
            self.eval_engine.restart_engine()
            visible = [frame.to(self.device, non_blocking=True) for frame in sample["vis_imgs"]]
            infrared = [frame.to(self.device, non_blocking=True) for frame in sample["ir_imgs"]]
            degraded_visible = torch.stack(sample["vis_blur_imgs"], dim=2).to(self.device, non_blocking=True)
            degraded_infrared = torch.stack(sample["ir_noise_imgs"], dim=2).to(self.device, non_blocking=True)
            frame_count = degraded_visible.shape[2]
            if frame_count < 2:
                raise ValueError(f"Sequence {sequence} requires at least two aligned frames")
            if (len(visible) != frame_count or len(infrared) != frame_count
                    or degraded_visible.shape != degraded_infrared.shape):
                raise ValueError(f"Unaligned input frames in sequence {sequence}")

            with torch.inference_mode(), torch.autocast("cuda", enabled=self.cfg.TEST_AMP):
                _, results, _, _ = self.eval_engine(
                    visible,
                    infrared,
                    degraded_visible,
                    degraded_infrared,
                    frame_num=frame_count,
                    use_prev_pred=True,
                    step=1,
                    return_features=False,
                    compute_loss=False,
                )
                if self.model_refine is not None:
                    results = self.refine(results, degraded_visible, degraded_infrared)
            self.save_results(results, sequence)
            print(f"Finished {sequence}: {frame_count} frames")
            self.eval_engine.restart_engine()
            del visible, infrared, degraded_visible, degraded_infrared, results
            torch.cuda.empty_cache()
