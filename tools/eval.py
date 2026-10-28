"""Evaluate MemoryFusion on aligned visible/infrared video sequences."""

import argparse
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from networks.managers.evaluator import Evaluator
from tools.get_config import get_config


def frame_limit(value):
    if value.lower() in {"none", "all"}:
        return None
    number = int(value)
    if number == 0:
        return None
    if number < 2:
        raise argparse.ArgumentTypeError("max_frames must be at least 2, or 0/none for all frames")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_name", default="MemoryFusion")
    parser.add_argument("--stage", default="memoryfusion")
    parser.add_argument("--model", default="r50_aotl", choices=["r50_aotl"])
    parser.add_argument("--result_path", default="results/test",
                        help="result base; output defaults to RESULT_PATH_ema")
    parser.add_argument("--output_path", default="", help="explicit output directory")
    parser.add_argument("--test_root", default=None, help="root containing the four aligned image folders")
    parser.add_argument("--test_list", default=None, help="text file containing one sequence name per line")
    parser.add_argument("--max_frames", type=frame_limit, default=200,
                        help="maximum frames per sequence (default: 200); 0/none means all")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--gpu_num", type=int, default=1, choices=[1], help="only single-GPU evaluation is supported")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ckpt_path_stage1", default="checkpoints/stage1.pth")
    parser.add_argument("--ckpt_path_stage2", default="checkpoints/stage2.pth")
    parser.add_argument("--with_stage2", action="store_true",
                        help="apply temporal refinement after Stage1; the original evaluation uses Stage1 only")
    parser.add_argument("--refine_window", type=int, default=0,
                        help="0 refines the whole sequence; values >8 use windows with a four-frame halo. "
                             "Windowing changes GroupNorm statistics and therefore output values")
    parser.add_argument("--former_mem_len", type=int, default=0)
    parser.add_argument("--latter_mem_len", type=int, default=8)
    parser.add_argument("--ltmb_update_strategy", choices=["ours"], default="ours")
    parser.add_argument("--recon_score_weight", type=float, default=None)
    parser.add_argument("--debug_ltmb_update", action="store_true")
    parser.add_argument("--fix_random", action="store_true")
    args = parser.parse_args()
    if args.gpu_id < 0:
        parser.error("--gpu_id must be nonnegative")
    if args.former_mem_len < 0 or args.latter_mem_len < 1:
        parser.error("memory lengths require former_mem_len >= 0 and latter_mem_len >= 1")
    if args.refine_window != 0 and args.refine_window <= 8:
        parser.error("--refine_window must be 0 or greater than 8")
    return args


def main():
    args = parse_args()
    cfg = get_config(args.stage, args.exp_name, args.model)
    cfg.TEST_GPU_ID = args.gpu_id
    cfg.TEST_GPU_NUM = 1
    cfg.TEST_AMP = args.amp
    cfg.TEST_CKPT_PATH_stage1 = args.ckpt_path_stage1
    cfg.TEST_CKPT_PATH_stage2 = args.ckpt_path_stage2
    cfg.TEST_WITH_STAGE2 = args.with_stage2
    cfg.TEST_REFINE_WINDOW = args.refine_window
    cfg.TEST_MAX_FRAMES = args.max_frames
    if args.test_root is not None:
        cfg.DATA_TEST_ROOT = args.test_root
    if args.test_list is not None:
        cfg.DATA_TEST_LIST = args.test_list
    cfg.FORMER_MEM_LEN = args.former_mem_len
    cfg.LATTER_MEM_LEN = args.latter_mem_len
    cfg.LTMB_UPDATE_STRATEGY = args.ltmb_update_strategy
    cfg.DEBUG_LTMB_UPDATE = args.debug_ltmb_update
    if args.recon_score_weight is not None:
        cfg.RECON_SCORE_WEIGHT = args.recon_score_weight
    cfg.FIX_RANDOM = args.fix_random
    cfg.result_path = args.result_path
    cfg.OUTPUT_PATH = args.output_path or args.result_path + "_ema"

    if args.fix_random:
        random.seed(2)
        np.random.seed(3)
        torch.manual_seed(4)
        torch.cuda.manual_seed_all(6)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    Evaluator(cfg).evaluating()


if __name__ == "__main__":
    main()
