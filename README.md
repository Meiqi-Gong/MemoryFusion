<div align="center">

# 🧠 MemoryFusion

### Cross-Temporal Memory Learning for Multimodal Video Fusion

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia&logoColor=white)
![GPU](https://img.shields.io/badge/GPU-Single_GPU-555555?logo=nvidia&logoColor=white)
![Pipeline](https://img.shields.io/badge/Training-Two_Stages-8A2BE2)

**Official training and evaluation code for MemoryFusion.**

</div>

---

## 🧭 Navigation

- [🛠️ Environment](#environment)
- [📂 Dataset Preparation](#dataset-preparation)
- [🚀 Training](#training)
- [🔄 Resume Training](#resume-training)
- [📊 Evaluation](#evaluation)
- [💾 Stage-2 Memory Usage](#stage-2-memory-usage)

<a id="environment"></a>
## 🛠️ Environment

The code has been validated with the following setup:

| Component | Version / Configuration |
| :--- | :--- |
| 🐍 Python | `3.10` |
| 🔥 PyTorch | `2.4.1` |
| 🖼️ torchvision | `0.19.1` |
| ⚡ CUDA | `12.1` |
| 🎮 GPU | Single CUDA GPU |

**📦 Install dependencies**

```bash
conda activate Fusion_Mem
pip install -r requirements.txt
```

<a id="dataset-preparation"></a>
## 📂 Dataset Preparation

Organize the data using the directory structure below:

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

> 📝 **Sequence lists:** Each line in `train.txt` or `test_all.txt` should contain one sequence name.

**✅ Dataset checklist**

- [ ] All four input streams contain frames with matching filenames.
- [ ] Corresponding frames have the same spatial resolution.
- [ ] Both width and height are divisible by **4**.
- [ ] Every sequence contains at least **2 frames**.
- [ ] Training sequences contain at least `--seq_len` frames.
- [ ] Training images are no smaller than `--patch_size`.

**🗂️ Configure dataset paths**

Use command-line arguments:

```bash
--train_root
--test_root
--train_list
--test_list
```

Or environment variables:

```bash
MEMORYFUSION_TRAIN_ROOT
MEMORYFUSION_TEST_ROOT
```

<a id="training"></a>
## 🚀 Training

MemoryFusion uses a **two-stage training pipeline**:

| Stage | Objective | Training behavior |
| :---: | :--- | :--- |
| 🧠 **Stage 1** | Memory-Guided Fusion | Train the Stage-1 network |
| 🎞️ **Stage 2** | Temporal Refinement | Freeze Stage 1 and train only the temporal refinement network |

### 🧠 Stage 1: Memory-Guided Fusion

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --stage stage1 \
  --train_stage stage1 \
  --batch_size 2 \
  --seq_len 30
```

### 🎞️ Stage 2: Temporal Refinement

Stage 1 is **frozen** while only the temporal refinement network is trained.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --stage stage2 \
  --train_stage stage2 \
  --batch_size 2 \
  --seq_len 30 \
  --pretrained_path checkpoints/stage1.pth
```


<a id="evaluation"></a>
## 📊 Evaluation

**📥 Pretrained checkpoints**

Place the weights at:

```text
checkpoints/
├── stage1.pth
└── stage2.pth
```

### 🧠 Stage-1 Evaluation

This corresponds to the original evaluation entry point. By default, each sequence is limited to at most **200 frames**.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py \
  --output_path results/test_stage1
```

### 🎬 Full Two-Stage Evaluation

Run both stages on the **complete video sequence**:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py \
  --with_stage2 \
  --max_frames 0 \
  --output_path results/test_stage2
```

**🖼️ Fused frames are saved as:**

```text
<output_path>/<sequence>/1.jpg
<output_path>/<sequence>/2.jpg
...
```

> 💡 **Tip:** Set `--max_frames 0` to process **all frames** in each sequence.

**🔎 Evaluation behavior**

- All model parameters are loaded strictly during evaluation.
- All networks are explicitly switched to `eval()` mode.

> ℹ️ **Reproducibility note:** The original evaluation entry point did not explicitly switch Stage 1 to evaluation mode. The cleaned implementation may therefore produce slightly different numerical outputs after disabling stochastic training-time layers, although the core network architecture and loss computation remain unchanged.

<a id="stage-2-memory-usage"></a>
## 💾 Stage-2 Memory Usage

By default, Stage 2 processes the **entire sequence at once**.

To reduce peak GPU memory consumption during temporal refinement, specify a refinement window:

```bash
--refine_window 30
```

**▶️ Example**

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py \
  --with_stage2 \
  --max_frames 0 \
  --refine_window 30 \
  --output_path results/test_stage2
```

> ⚠️ **Memory note:** `--refine_window` reduces memory usage for **Stage 2 only**. Stage 1 still loads and processes the full input sequence.

---

<div align="center">

**🧠 MemoryFusion · Cross-Temporal Memory Learning · 🎬 Multimodal Video Fusion**

</div>
