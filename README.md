<h1 align="center">MemoryFusion: Cross-Temporal Memory Learning for <br> Multimodal Video Fusion</h1>

This repository contains the training and evaluation code for **MemoryFusion**.

---

## Environment

The code has been validated with:

- Python 3.10
- PyTorch 2.4.1
- torchvision 0.19.1
- CUDA 12.1
- Single CUDA GPU

Install the required dependencies with:

```bash
conda activate Fusion_Mem
pip install -r requirements.txt
```

---

## Dataset Preparation

The expected dataset structure is:

```text
datasets/
├── train/
│   ├── train.txt
│   ├── VI/
│   │   └── <sequence>/*.jpg
│   ├── IR_en/
│   │   └── <sequence>/*.jpg
│   ├── VI_blur_new/
│   │   └── <sequence>/*.jpg
│   └── IR_noise2/
│       └── <sequence>/*.jpg
│
└── test/
    ├── test_all.txt
    ├── VI_en/
    │   └── <sequence>/*.jpg
    ├── IR_en/
    │   └── <sequence>/*.jpg
    ├── VI_blur_new/
    │   └── <sequence>/*.jpg
    └── IR_noise2/
        └── <sequence>/*.jpg
```

Each line in `train.txt` or `test_all.txt` should contain one sequence name.

For each sequence:

- all four input streams must contain frames with matching filenames;
- corresponding frames must have the same spatial resolution;
- both width and height must be divisible by 4;
- each sequence must contain at least 2 frames;
- training sequences must contain at least `--seq_len` frames;
- training images must be no smaller than `--patch_size`.

Dataset paths can be specified either through command-line arguments:

```bash
--train_root
--test_root
--train_list
--test_list
```

or through environment variables:

```bash
MEMORYFUSION_TRAIN_ROOT
MEMORYFUSION_TEST_ROOT
```

---

## Training

MemoryFusion is trained in two stages.

### Stage 1: Memory-Guided Fusion

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --stage stage1 \
  --train_stage stage1 \
  --batch_size 2 \
  --seq_len 30
```

### Stage 2: Temporal Refinement

Stage 1 is frozen while only the temporal refinement network is trained.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --stage stage2 \
  --train_stage stage2 \
  --batch_size 2 \
  --seq_len 30 \
  --pretrained_path checkpoints/stage1.pth
```

### Useful Training Options

`--total_step`  
Sets the total number of training iterations.

`--workers 0`  
Disables multiprocessing for data loading.

`--no_amp`  
Disables automatic mixed-precision training.

`--val_step 0`  
Disables validation during training.

Training outputs are stored in:

```text
results/<experiment>/<stage>/
```

Logs are stored in:

```text
logs/
```

---

## Resume Training

New-format checkpoints store:

- model parameters,
- optimizer state,
- AMP scaler state,
- training step,
- EMA state,
- random-number-generator state.

Training can be resumed with `--resume_path`.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --stage stage1 \
  --resume_path results/memoryfusion_R50_AOTL_Temp_pe_Slot_4/stage1/ckpt/latest_stage1.pth
```

Legacy checkpoints do not contain `next_step`. When resuming from an old checkpoint, additionally specify:

```bash
--resume_step <step>
```

Note that resumed training restarts the data-loading epoch. Therefore, the exact sample order is not guaranteed to match an uninterrupted training run.

For Stage-2 resume training, the corresponding Stage-1 checkpoint must still be provided.

EMA-exported checkpoints are intended for inference only.

---

## Evaluation

Place the pretrained weights at:

```text
checkpoints/
├── stage1.pth
└── stage2.pth
```

### Stage-1 Evaluation

This corresponds to the original evaluation entry point. By default, each sequence is limited to at most 200 frames.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py \
  --output_path results/test_stage1
```

### Full Two-Stage Evaluation

To run both stages on the complete video sequence:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py \
  --with_stage2 \
  --max_frames 0 \
  --output_path results/test_stage2
```

The fused frames are saved as:

```text
<output_path>/<sequence>/1.jpg
<output_path>/<sequence>/2.jpg
...
```

Setting

```bash
--max_frames 0
```

processes all frames in each sequence.

All model parameters are loaded strictly during evaluation, and all networks are explicitly switched to `eval()` mode.

The original evaluation entry point did not explicitly switch Stage 1 to evaluation mode. Therefore, the cleaned implementation may produce slightly different numerical outputs after disabling stochastic training-time layers, although the core network architecture and loss computation remain unchanged.

---

## Stage-2 Memory Usage

By default, Stage 2 processes the entire sequence at once.

To reduce peak GPU memory usage during temporal refinement, use:

```bash
--refine_window 30
```

For example:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py \
  --with_stage2 \
  --max_frames 0 \
  --refine_window 30 \
  --output_path results/test_stage2
```

Note that `--refine_window` only reduces the memory usage of Stage 2. Stage 1 still loads and processes the full input sequence.
