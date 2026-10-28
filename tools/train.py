"""Train MemoryFusion Stage 1 or temporal refinement on one GPU."""
import argparse
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.get_config import get_config
from networks.managers.trainer import Trainer
from utils.utils import Tee, make_log_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp_name', default='memoryfusion')
    parser.add_argument('--stage', default='memoryfusion', help='output run name')
    parser.add_argument('--train_stage', choices=['stage1', 'stage2'], default='stage1')
    parser.add_argument('--model', default='r50_aotl')
    parser.add_argument('--start_gpu', type=int, default=0)
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--amp', dest='amp', action='store_true')
    parser.add_argument('--no_amp', dest='amp', action='store_false')
    parser.set_defaults(amp=True)
    parser.add_argument('--pretrained_path', default='', help='Stage 1 initialization/checkpoint')
    parser.add_argument('--no_pretrain', action='store_true', help='explicit random initialization')
    parser.add_argument('--resume_path', default='', help='resume the selected training stage, including optimizer')
    parser.add_argument('--resume_step', type=int, default=-1, help='next step; needed for old checkpoints without metadata')
    parser.add_argument('--train_root', default=None)
    parser.add_argument('--test_root', default=None)
    parser.add_argument('--train_list', default=None)
    parser.add_argument('--test_list', default=None)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--patch_size', type=int, default=224)
    parser.add_argument('--test_max_frames', type=int, default=205)
    parser.add_argument('--save_step', type=int, default=2000)
    parser.add_argument('--val_step', type=int, default=1000, help='0 disables validation')
    parser.add_argument('--datasets', nargs='+', default=['M3SVD'])
    parser.add_argument('--lr', type=float, default=-1.)
    parser.add_argument('--total_step', type=int, default=-1)
    parser.add_argument('--start_step', type=int, default=-1)
    parser.add_argument('--log', default='./logs')
    parser.add_argument('--fix_random', action='store_true')
    parser.add_argument('--debug_fix_random', action='store_true')
    parser.add_argument('--ltmb_update_strategy', default='ours', choices=['ours', 'random', 'fifo', 'attention', 'rmem'])
    parser.add_argument('--debug_ltmb_update', action='store_true')
    parser.add_argument('--recon_score_weight', type=float, default=1.)
    parser.add_argument('--rmem_relevance_ema_beta', type=float, default=.2)
    parser.add_argument('--rmem_freshness_weight', type=float, default=1.5)
    parser.add_argument('--rmem_freshness_add', type=float, default=8.)
    args = parser.parse_args()
    if args.gpu_num != 1:
        parser.error('This release supports single-GPU training only (--gpu_num 1).')
    if args.batch_size < 1 or args.seq_len < 2 or args.patch_size < 32 or args.patch_size % 4:
        parser.error('batch_size >= 1, seq_len >= 2, and patch_size >= 32 divisible by 4 are required.')
    if args.workers < 0 or args.save_step < 1 or args.val_step < 0 or args.test_max_frames < 2:
        parser.error('workers/val_step must be nonnegative; save_step >= 1 and test_max_frames >= 2 are required.')
    import torch
    if args.start_gpu < 0 or not torch.cuda.is_available() or args.start_gpu >= torch.cuda.device_count():
        parser.error('A CUDA GPU is required; --start_gpu must select an available visible CUDA device.')
    if args.datasets != ['M3SVD']:
        parser.error('This release supports only M3SVD paired video data.')
    if args.no_pretrain and args.pretrained_path:
        parser.error('--no_pretrain cannot be combined with --pretrained_path.')
    if args.train_stage == 'stage2' and not args.pretrained_path:
        parser.error('Stage 2 requires --pretrained_path pointing to the trained Stage 1 model.')
    for label, path in [('pretrained', args.pretrained_path), ('resume', args.resume_path)]:
        if path and not os.path.isfile(path):
            parser.error(f'{label} checkpoint does not exist: {path}')
    if args.resume_step > 0 and not args.resume_path:
        parser.error('--resume_step requires --resume_path.')
    cfg = get_config(args.stage, args.exp_name, args.model)
    cfg.TRAIN_STAGE = args.train_stage
    cfg.DATASETS = args.datasets
    cfg.TRAIN_GPUS = 1
    cfg.DIST_ENABLE = False
    cfg.DIST_START_GPU = args.start_gpu
    cfg.TRAIN_BATCH_SIZE = args.batch_size
    cfg.DATA_WORKERS = cfg.TEST_WORKERS = args.workers
    cfg.DATA_SEQ_LEN, cfg.PATCH_SIZE = args.seq_len, args.patch_size
    if args.train_root is not None:
        cfg.DATA_TRAIN_ROOT = args.train_root
    if args.test_root is not None:
        cfg.DATA_TEST_ROOT = args.test_root
    if args.train_list is not None:
        cfg.DATA_TRAIN_LIST = args.train_list
    if args.test_list is not None:
        cfg.DATA_TEST_LIST = args.test_list
    cfg.TEST_MAX_FRAMES = args.test_max_frames
    cfg.TRAIN_SAVE_STEP, cfg.VAL_TBLOG_STEP = args.save_step, args.val_step
    cfg.PRETRAIN = bool(args.pretrained_path)
    cfg.PRETRAIN_FULL = True
    cfg.PRETRAIN_MODEL = args.pretrained_path
    cfg.TRAIN_AUTO_RESUME = False
    cfg.TRAIN_RESUME = bool(args.resume_path)
    cfg.TRAIN_RESUME_PATH, cfg.TRAIN_RESUME_STEP = args.resume_path, args.resume_step
    if args.lr > 0:
        cfg.TRAIN_LR = cfg.TRAIN_LR_refine = args.lr
    if args.total_step > 0:
        cfg.TRAIN_TOTAL_STEPS = args.total_step
    if args.start_step > 0:
        cfg.TRAIN_START_STEP = args.start_step
    cfg.LTMB_UPDATE_STRATEGY = args.ltmb_update_strategy
    cfg.DEBUG_LTMB_UPDATE = args.debug_ltmb_update
    cfg.RECON_SCORE_WEIGHT = args.recon_score_weight
    cfg.RMEM_RELEVANCE_EMA_BETA = args.rmem_relevance_ema_beta
    cfg.RMEM_FRESHNESS_WEIGHT = args.rmem_freshness_weight
    cfg.RMEM_FRESHNESS_ADD = args.rmem_freshness_add
    cfg.DEBUG_FIX_RANDOM, cfg.FIX_RANDOM = args.debug_fix_random, args.fix_random
    cfg.save_self()
    if cfg.FIX_RANDOM:
        import numpy as np
        import torch
        random.seed(2)
        np.random.seed(3)
        torch.manual_seed(4)
        torch.cuda.manual_seed_all(5)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    log_dir = make_log_dir(args.log, cfg.EXP_NAME)
    tee = Tee(os.path.join(log_dir, 'print.log'))
    sys.stdout = tee
    trainer = None
    try:
        trainer = Trainer(rank=0, cfg=cfg, enable_amp=args.amp, stage=args.train_stage)
        if args.train_stage == 'stage2':
            trainer.stage2_training()
        else:
            trainer.sequential_training()
    finally:
        if trainer is not None:
            trainer.close()
        tee.close()


if __name__ == '__main__':
    main()
