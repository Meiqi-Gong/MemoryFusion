"""Single-GPU training for fusion and temporal refinement."""
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid

from dataloaders.train_datasets import M3SVD_Train
from dataloaders.eval_datasets import M3SVD_Test
import dataloaders.video_transforms as tr
from loss.loss import Fusion_loss_stage2
from networks.decoders.decoder_fusion import TemporalDetailRefinement
from networks.models import build_vos_model
from networks.engines import build_engine
from utils.learning import adjust_learning_rate, get_trainable_params
from utils.ema import ExponentialMovingAverage, get_param_buffer_for_ema


class Trainer:
    def __init__(self, rank, cfg, enable_amp=True, stage='stage1'):
        if rank != 0 or cfg.TRAIN_GPUS != 1:
            raise ValueError('This release supports single-GPU training only.')
        self.rank, self.cfg, self.stage = rank, cfg, stage
        self.gpu = cfg.DIST_START_GPU
        self.device = torch.device('cuda', self.gpu)
        torch.cuda.set_device(self.gpu)
        torch.backends.cudnn.benchmark = not getattr(cfg, 'FIX_RANDOM', False)
        self.print_log(json.dumps(cfg.__dict__, indent=2, sort_keys=True))
        self.model = build_vos_model(cfg.MODEL_VOS, cfg).to(self.device)
        if stage == 'stage1':
            self._configure_trainable_parameters()
        else:
            self.model.requires_grad_(False)
            self.model.eval()
        self.engine = build_engine(cfg.MODEL_ENGINE, 'train', aot_model=self.model,
                                   gpu_id=self.gpu, long_term_mem_gap=cfg.TRAIN_LONG_TERM_MEM_GAP)
        self.eval_engine = build_engine(cfg.MODEL_ENGINE, 'eval', aot_model=self.model,
                                        gpu_id=self.gpu, long_term_mem_gap=cfg.TRAIN_LONG_TERM_MEM_GAP)
        self.dist_engine, self.dist_eval_engine = self.engine, self.eval_engine
        self.enable_amp = enable_amp
        self.scaler = torch.amp.GradScaler('cuda', enabled=enable_amp)
        self.model_refine = TemporalDetailRefinement().to(self.device) if stage == 'stage2' else None
        self.train_model = self.model_refine if stage == 'stage2' else self.model
        self.base_lr = cfg.TRAIN_LR_refine if stage == 'stage2' else cfg.TRAIN_LR
        groups = get_trainable_params(
            self.train_model, base_lr=self.base_lr, weight_decay=cfg.TRAIN_WEIGHT_DECAY,
            use_frozen_bn=cfg.MODEL_FREEZE_BN,
            exclusive_wd_dict=cfg.TRAIN_WEIGHT_DECAY_EXCLUSIVE,
            no_wd_keys=cfg.TRAIN_WEIGHT_DECAY_EXEMPTION)
        if cfg.TRAIN_OPT == 'sgd':
            self.optimizer = torch.optim.SGD(groups, lr=self.base_lr,
                                              momentum=cfg.TRAIN_SGD_MOMENTUM, nesterov=True)
        else:
            self.optimizer = torch.optim.AdamW(groups, lr=self.base_lr,
                                                weight_decay=cfg.TRAIN_WEIGHT_DECAY)
        if stage == 'stage2':
            self.optimizer_refine = self.optimizer
            self.calculate_allloss = Fusion_loss_stage2().to(self.device)
        self.best_fusion_loss = float('inf')
        self.step, self.epoch = cfg.TRAIN_START_STEP, 0
        self.prepare_dataset()
        self.ema = None
        self.process_pretrained_model()
        if stage == 'stage1':
            self.ema_params = get_param_buffer_for_ema(self.model, update_buffer=not cfg.MODEL_FREEZE_BN)
            decay = max(0., 1. - 1. / (cfg.TRAIN_TOTAL_STEPS * cfg.TRAIN_EMA_RATIO))
            self.ema = ExponentialMovingAverage(self.ema_params, decay=decay)
            ema_state = getattr(self, '_resume_ema', None)
            if ema_state:
                self.ema.num_updates = ema_state['num_updates']
                self.ema.shadow_params = [v.to(self.device) for v in ema_state['shadow_params']]
            else:
                self.ema.num_updates = max(0, self.step - 1)
        self.tb_writer = SummaryWriter(cfg.DIR_TB_LOG)

    def _configure_trainable_parameters(self):
        cfg = self.cfg
        if cfg.MODEL_FREEZE_BACKBONE:
            self.model.encoder.requires_grad_(False)
        if cfg.FREEZE_AOT_EXCEPT_TEMPORAL_EMB:
            self.model.requires_grad_(False)
            for name, parameter in self.model.named_parameters():
                if 'cur_pos_emb' in name or 'mem_pos_emb' in name:
                    parameter.requires_grad = True
        if cfg.FREEZE_AOT_EXCEPT_GRU:
            self.model.requires_grad_(False)
            for name, module in self.model.named_modules():
                if 'grus' in name:
                    module.requires_grad_(True)

    def process_pretrained_model(self):
        cfg = self.cfg
        if cfg.PRETRAIN:
            checkpoint = torch.load(cfg.PRETRAIN_MODEL, map_location='cpu', weights_only=True)
            if checkpoint.get('train_stage') == 'stage2':
                raise ValueError('--pretrained_path must contain Stage 1 model weights, not refinement weights.')
            state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
            state = {key.removeprefix('module.'): value for key, value in state.items()}
            self.model.load_state_dict(state, strict=True)
            self.print_log(f'Loaded Stage 1 initialization: {cfg.PRETRAIN_MODEL}')
        if self.stage == 'stage2':
            if not cfg.PRETRAIN:
                raise ValueError('Stage 2 requires a pretrained Stage 1 checkpoint.')
            self.model.requires_grad_(False)
            self.model.eval()
        if not cfg.TRAIN_RESUME:
            return
        checkpoint = torch.load(cfg.TRAIN_RESUME_PATH, map_location='cpu', weights_only=True)
        if checkpoint.get('train_stage', self.stage) != self.stage:
            raise ValueError('Resume checkpoint belongs to another training stage.')
        self.train_model.load_state_dict(checkpoint['state_dict'], strict=True)
        if 'optimizer' not in checkpoint:
            raise ValueError('Resume requires a training checkpoint containing optimizer state.')
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint.get('scaler'):
            self.scaler.load_state_dict(checkpoint['scaler'])
        self.best_fusion_loss = checkpoint.get('best_fusion_loss', float('inf'))
        self.step = cfg.TRAIN_RESUME_STEP if cfg.TRAIN_RESUME_STEP > 0 else checkpoint.get('next_step')
        if self.step is None:
            raise ValueError('Old checkpoints require --resume_step with the next training step.')
        self.epoch = checkpoint.get('epoch', 0)
        self._resume_ema = checkpoint.get('ema')
        rng = checkpoint.get('rng_state')
        if rng:
            torch.set_rng_state(rng['torch'])
            torch.cuda.set_rng_state(rng['cuda'], self.device)
            numpy_state = rng['numpy']
            np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                                 numpy_state[2], numpy_state[3], numpy_state[4]))
            random.setstate(rng['python'])
        self.print_log(f'Resume {self.stage}: {cfg.TRAIN_RESUME_PATH}, next step {self.step}.')

    def prepare_dataset(self):
        cfg = self.cfg
        if cfg.DATASETS != ['M3SVD']:
            raise ValueError('Only M3SVD is supported in this release.')
        transform = transforms.Compose([tr.PairedRandomCrop(cfg.PATCH_SIZE),
                                        tr.RandomHorizontalFlip(cfg.DATA_RANDOMFLIP), tr.ToTensor()])
        dataset = M3SVD_Train(
            root=getattr(cfg, 'DATA_TRAIN_ROOT', None), list_file=getattr(cfg, 'DATA_TRAIN_LIST', None),
            transform=transform, repeat_time=100, seq_len=cfg.DATA_SEQ_LEN,
            rand_gap=cfg.DATA_RANDOM_GAP_VOST, rand_reverse=cfg.DATA_RANDOM_REVERSE_SEQ,
            merge_prob=cfg.DATA_DYNAMIC_MERGE_PROB, max_obj_n=cfg.MODEL_MAX_OBJ_NUM,
            ignore_thresh=cfg.DATA_VOST_IGNORE_THRESH, ignore_in_merge=cfg.IGNORE_IN_MERGE)
        self.train_sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True)
        options = dict(num_workers=cfg.DATA_WORKERS, pin_memory=True)
        if cfg.DATA_WORKERS > 0:
            options['prefetch_factor'] = 4
        self.train_loader = DataLoader(dataset, batch_size=cfg.TRAIN_BATCH_SIZE,
                                       sampler=self.train_sampler, drop_last=True, **options)
        if not len(self.train_loader):
            raise ValueError('Training dataset is too small for the requested batch size.')
        self.val_loader = None
        if cfg.VAL_TBLOG_STEP > 0:
            dataset = M3SVD_Test(
                root=getattr(cfg, 'DATA_TEST_ROOT', None), list_file=getattr(cfg, 'DATA_TEST_LIST', None),
                max_frames=getattr(cfg, 'TEST_MAX_FRAMES', 205),
                split=[cfg.VAL_DATASET_SPLIT], transform=transforms.Compose([tr.ToTensor()]))
            if not len(dataset):
                raise ValueError('Validation dataset is empty; use --val_step 0 to disable validation.')
            self.val_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                          num_workers=cfg.TEST_WORKERS, pin_memory=True)

    def _batch_to_device(self, sample, max_frames=None):
        values = []
        for key in ('vis_imgs', 'ir_imgs', 'vis_blur_imgs', 'ir_noise_imgs'):
            frames = sample[key] if max_frames is None else sample[key][:max_frames]
            values.append([frame.to(self.device, non_blocking=True) for frame in frames])
        vis, ir, degraded_vis, degraded_ir = values
        return vis, ir, torch.stack(degraded_vis, dim=2), torch.stack(degraded_ir, dim=2)

    @staticmethod
    def _flatten_video(video):
        b, c, t, h, w = video.shape
        return video.permute(2, 0, 1, 3, 4).reshape(t * b, c, h, w)

    def _set_lr(self, step):
        cfg = self.cfg
        if step % cfg.TRAIN_LR_UPDATE_STEP:
            return
        use_prev_pred = step >= int(cfg.TRAIN_SEQ_TRAINING_START_RATIO * cfg.TRAIN_TOTAL_STEPS)
        adjust_learning_rate(
            self.optimizer, base_lr=self.base_lr, p=cfg.TRAIN_LR_POWER,
            itr=step, max_itr=cfg.TRAIN_TOTAL_STEPS, restart=cfg.TRAIN_LR_RESTART,
            warm_up_steps=cfg.TRAIN_LR_WARM_UP_RATIO * cfg.TRAIN_TOTAL_STEPS,
            is_cosine_decay=cfg.TRAIN_LR_COSINE_DECAY, min_lr=cfg.TRAIN_LR_MIN,
            encoder_lr_ratio=cfg.TRAIN_LR_ENCODER_RATIO if self.stage == 'stage1' else 1.,
            freeze_params=cfg.TRAIN_SEQ_TRAINING_FREEZE_PARAMS if self.stage == 'stage1' and use_prev_pred else [])

    def _backward(self, loss):
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite training loss at step {self.step}: {loss.item()}')
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.train_model.parameters(), self.cfg.TRAIN_CLIP_GRAD_NORM)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def sequential_training(self):
        """Use the original Stage 1 engine, loss, LR schedule, clipping and EMA."""
        self._training_loop(stage2=False)

    def stage2_training(self):
        """Optimize only temporal refinement using frozen Stage 1 predictions."""
        self._training_loop(stage2=True)

    def _stage2_forward(self, vis, ir, degraded_vis, degraded_ir):
        self.model.eval()
        self.eval_engine.eval()
        self.eval_engine.restart_engine()
        # The inference wrapper defaults to one sample, while Stage 2 trains in batches.
        self.eval_engine.aot_engines.restart_engine(batch_size=degraded_vis.shape[0])
        with torch.no_grad():
            _, stage1, _, _ = self.eval_engine(
                vis, ir, degraded_vis, degraded_ir, frame_num=degraded_vis.shape[2],
                use_prev_pred=True, step=self.step, return_features=False, compute_loss=False)
        refined = self.model_refine(stage1.detach(), degraded_vis, degraded_ir)
        t, b = refined.shape[2], refined.shape[0]
        losses = self.calculate_allloss(
            self._flatten_video(refined), self._flatten_video(torch.stack(ir, dim=2)),
            self._flatten_video(torch.stack(vis, dim=2)), t, b)
        return losses, refined

    def _training_loop(self, stage2):
        cfg = self.cfg
        self.train_model.train()
        if stage2:
            self.model.eval()
        else:
            self.engine.train()
        last_completed, last_saved = self.step - 1, None
        while self.step <= cfg.TRAIN_TOTAL_STEPS:
            self.train_sampler.set_epoch(self.epoch)
            self.epoch += 1
            for sample in self.train_loader:
                if self.step > cfg.TRAIN_TOTAL_STEPS:
                    break
                self._set_lr(self.step)
                vis, ir, degraded_vis, degraded_ir = self._batch_to_device(sample)
                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=self.enable_amp):
                    if stage2:
                        losses, results = self._stage2_forward(vis, ir, degraded_vis, degraded_ir)
                        loss = losses['loss_fusion']
                    else:
                        self.engine.restart_engine(degraded_vis.shape[0], True)
                        use_prev_pred = self.step >= int(cfg.TRAIN_SEQ_TRAINING_START_RATIO * cfg.TRAIN_TOTAL_STEPS)
                        losses, loss, results, _, _ = self.engine(
                            vis, ir, degraded_vis, degraded_ir, degraded_vis.shape[0],
                            use_prev_pred=use_prev_pred, step=self.step,
                            tf_board=cfg.TRAIN_TBLOG and self.step % cfg.TRAIN_TBLOG_STEP == 0)
                self._backward(loss)
                if self.ema is not None:
                    self.ema.update(self.ema_params)
                completed = self.step
                self.step += 1
                last_completed = completed
                if completed % cfg.TRAIN_LOG_STEP == 0 or completed == 1:
                    self.print_log(f'[{self.stage} step {completed}] loss={loss.item():.6f}')
                if completed % cfg.TRAIN_TBLOG_STEP == 0:
                    self.log_to_tensorboard(losses, results, torch.stack(vis, dim=2),
                                            torch.stack(ir, dim=2), degraded_vis, degraded_ir, completed)
                validation_loss = None
                if self.val_loader is not None and completed % cfg.VAL_TBLOG_STEP == 0:
                    validation_loss = self.validate(completed)
                improved = validation_loss is not None and validation_loss < self.best_fusion_loss
                if improved:
                    self.best_fusion_loss = validation_loss
                if completed % cfg.TRAIN_SAVE_STEP == 0 or improved:
                    self.save_checkpoint(completed, is_best=improved)
                    last_saved = completed
                del loss, losses, results, vis, ir, degraded_vis, degraded_ir
        if last_completed >= cfg.TRAIN_START_STEP and last_saved != last_completed:
            self.save_checkpoint(last_completed)
        self.tb_writer.flush()
        self.print_log(f'{self.stage} training finished at step {last_completed}.')

    @torch.no_grad()
    def validate(self, step):
        self.model.eval()
        self.eval_engine.eval()
        if self.model_refine is not None:
            self.model_refine.eval()
        totals, count = {}, 0
        for sample in self.val_loader:
            vis, ir, degraded_vis, degraded_ir = self._batch_to_device(sample, self.cfg.DATA_SEQ_LEN)
            frames = min(self.cfg.DATA_SEQ_LEN, degraded_vis.shape[2])
            vis, ir = vis[:frames], ir[:frames]
            degraded_vis, degraded_ir = degraded_vis[:, :, :frames], degraded_ir[:, :, :frames]
            with torch.amp.autocast('cuda', enabled=False):
                if self.stage == 'stage2':
                    losses, _ = self._stage2_forward(vis, ir, degraded_vis, degraded_ir)
                else:
                    self.eval_engine.restart_engine()
                    losses, _, _, _ = self.eval_engine(
                        vis, ir, degraded_vis, degraded_ir, frame_num=frames,
                        use_prev_pred=True, step=step, return_features=False)
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.) + value.detach().float().mean().item()
            count += 1
        if not count or 'loss_fusion' not in totals:
            raise RuntimeError('Validation returned no fusion losses.')
        means = {key: value / count for key, value in totals.items()}
        for key, value in means.items():
            self.tb_writer.add_scalar(f'val_loss/{key}', value, step)
        self.print_log(f'[{self.stage} validation {step}] {means}')
        self.train_model.train()
        if self.stage == 'stage1':
            self.engine.train()
        else:
            self.model.eval()
        return means['loss_fusion']

    def save_checkpoint(self, completed_step, is_best=False):
        cfg = self.cfg
        numpy_state = np.random.get_state()
        numpy_state = (numpy_state[0], numpy_state[1].tolist(), numpy_state[2], numpy_state[3], numpy_state[4])
        checkpoint = {
            'state_dict': self.train_model.state_dict(), 'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(), 'step': completed_step, 'next_step': self.step,
            'epoch': self.epoch, 'train_stage': self.stage, 'best_fusion_loss': self.best_fusion_loss,
            'rng_state': {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(self.device),
                          'numpy': numpy_state, 'python': random.getstate()},
        }
        if self.ema is not None:
            checkpoint['ema'] = {'num_updates': self.ema.num_updates, 'shadow_params': self.ema.shadow_params}
        if self.stage == 'stage2':
            checkpoint['stage1_checkpoint'] = cfg.PRETRAIN_MODEL
        directory = Path(cfg.DIR_CKPT)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, directory / f'latest_{self.stage}.pth')
        torch.save(checkpoint, directory / f'save_{self.stage}_{completed_step}.pth')
        if is_best:
            torch.save(checkpoint, directory / f'best_{self.stage}.pth')
        snapshots = sorted(directory.glob(f'save_{self.stage}_*.pth'), key=lambda p: int(p.stem.rsplit('_', 1)[1]))
        for old in snapshots[:-max(1, cfg.TRAIN_MAX_KEEP_CKPT)]:
            old.unlink()
        if self.ema is not None:
            self.ema.store(self.ema_params)
            try:
                self.ema.copy_to(self.ema_params)
                directory = Path(cfg.DIR_EMA_CKPT)
                directory.mkdir(parents=True, exist_ok=True)
                # EMA exports contain inference weights; use ordinary checkpoints to resume training.
                ema_checkpoint = {'state_dict': self.model.state_dict(), 'step': completed_step,
                                   'train_stage': 'stage1', 'ema_inference': True}
                torch.save(ema_checkpoint, directory / 'latest_stage1.pth')
                if is_best:
                    torch.save(ema_checkpoint, directory / 'best_stage1.pth')
            finally:
                self.ema.restore(self.ema_params)
        self.print_log(f'Saved {self.stage} step {completed_step}; next step {self.step}.')

    def log_to_tensorboard(self, all_loss, results, vis_imgs, ir_imgs,
                           vis_blur_imgs, ir_noise_imgs, step, num_visualize=3):
        for key, value in all_loss.items():
            self.tb_writer.add_scalar(f'loss/{key}', value.detach().float().mean().item(), step)
        frame = min(results.shape[2] - 1, num_visualize // 2)
        images = []
        for batch in range(min(2, results.shape[0])):
            for video in (ir_noise_imgs, vis_blur_imgs, ir_imgs, vis_imgs, results):
                images.append(video[batch, :, frame].detach().float().cpu().clamp(0, 1))
        self.tb_writer.add_image('images', make_grid(images, nrow=5, padding=2), step)

    def print_log(self, message):
        print(message)

    def close(self):
        self.tb_writer.close()
