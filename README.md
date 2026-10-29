<div align="center">

# 🧠 MemoryFusion

### Cross-Temporal Memory Learning for Multimodal Video Fusion

🎉 **Accepted to NeurIPS 2026!**

Training and evaluation code for **MemoryFusion**.

</div>

---

## ⚙️ Environment

Tested with **Python 3.10**, **PyTorch 2.4.1**, **torchvision 0.19.1**, **CUDA 12.1**, and a single CUDA GPU.

```bash
conda activate Fusion_Mem
pip install -r requirements.txt
```

## 📁 Dataset

Expected directory structure:

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

Each line in `train.txt` or `test_all.txt` contains one sequence name. For each sequence:

- All four streams must have matching frame filenames and resolutions.
- Frame width and height must be divisible by **4**.
- Each sequence needs at least **2 frames**. Training sequences need at least `--seq_len` frames, with images no smaller than `--patch_size`.

Dataset locations can be set with `--train_root`, `--test_root`, `--train_list`, and `--test_list`, or with the `MEMORYFUSION_TRAIN_ROOT` and `MEMORYFUSION_TEST_ROOT` environment variables.

## 🚀 Training

MemoryFusion is trained in two stages.

**Stage 1 — Memory-Guided Fusion**

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --stage stage1 \
  --train_stage stage1 \
  --batch_size 2 \
  --seq_len 30
```

**Stage 2 — Temporal Refinement**

Stage 1 is frozen; only the temporal refinement network is trained.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --stage stage2 \
  --train_stage stage2 \
  --batch_size 2 \
  --seq_len 30 \
  --pretrained_path checkpoints/stage1.pth
```

Training outputs are saved to `results/<experiment>/<stage>/`, and logs to `logs/`.

<details>
<summary><b>Training options and checkpoint resume</b></summary>

Useful options:

- `--total_step`: total training iterations
- `--workers 0`: disable multiprocessing for data loading
- `--no_amp`: disable mixed-precision training
- `--val_step 0`: disable validation

Resume with `--resume_path`:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --stage stage1 \
  --resume_path results/memoryfusion_R50_AOTL_Temp_pe_Slot_4/stage1/ckpt/latest_stage1.pth
```

New-format checkpoints include model, optimizer, AMP scaler, step, EMA, and RNG states. For legacy checkpoints without `next_step`, also set `--resume_step <step>`.

Resumed runs restart the data-loading epoch, so their sample order may differ. Stage-2 resume still requires the Stage-1 checkpoint. EMA-exported checkpoints are for inference only.

</details>

## 🎬 Evaluation

Place pretrained weights at `checkpoints/stage1.pth` and `checkpoints/stage2.pth`.

**Stage 1** (up to 200 frames per sequence by default):

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py \
  --output_path results/test_stage1
```

**Full two-stage evaluation** (all frames):

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py \
  --with_stage2 \
  --max_frames 0 \
  --output_path results/test_stage2
```

Fused frames are saved as `<output_path>/<sequence>/1.jpg`, `2.jpg`, and so on.

**Memory tip:** Stage 2 processes the entire sequence by default. Add `--refine_window 30` to the two-stage command to reduce Stage-2 peak GPU memory usage. Stage 1 still processes the full sequence. Set `--max_frames 0` to process every frame.

<details>
<summary><b>Evaluation reproducibility note</b></summary>

All parameters are loaded strictly and networks are set to `eval()`. The original Stage-1 evaluation entry point did not explicitly enable evaluation mode, so the cleaned implementation may produce slightly different numerical outputs after disabling stochastic training-time layers. The core network architecture and loss computation remain unchanged.

</details>
