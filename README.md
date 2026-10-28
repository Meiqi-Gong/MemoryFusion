# MemoryFusion

保留当前融合编码器、AOT 记忆网络、损失、M3SVD 数据读取及 Stage 2 时序细化网络的训练和测试代码。移除旧骨干、失效入口、数据退化生成脚本、实验日志、结果、缓存和 IDE 配置。

## 环境

验证环境：Python 3.10、PyTorch 2.4.1 / torchvision 0.19.1、CUDA 12.1，单张 CUDA GPU。

```bash
conda activate Fusion_Mem
pip install -r requirements.txt
```

## 数据

```text
datasets/
  train/
    train.txt
    VI/<sequence>/*.jpg
    IR_en/<sequence>/*.jpg
    VI_blur_new/<sequence>/*.jpg
    IR_noise2/<sequence>/*.jpg
  test/
    test_all.txt
    VI_en/<sequence>/*.jpg
    IR_en/<sequence>/*.jpg
    VI_blur_new/<sequence>/*.jpg
    IR_noise2/<sequence>/*.jpg
```

列表每行一个序列名；四路对应帧需同名、同尺寸，宽高为 4 的倍数，至少两帧。训练序列帧数至少为 `--seq_len`，尺寸不小于 `--patch_size`。可用 `--train_root/--test_root`、`--train_list/--test_list`，或环境变量 `MEMORYFUSION_TRAIN_ROOT/MEMORYFUSION_TEST_ROOT` 改路径。

## 训练

Stage 1：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --stage stage1 --train_stage stage1 --batch_size 2 --seq_len 30
```

Stage 2（冻结 Stage 1，仅训练细化网络）：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --stage stage2 --train_stage stage2 --batch_size 2 --seq_len 30 \
  --pretrained_path checkpoints/stage1.pth
```

`--total_step` 设置总步数；`--workers 0` 关闭加载子进程；`--no_amp` 关闭混合精度；`--val_step 0` 关闭训练内验证。输出在 `results/<experiment>/<stage>/`，日志在 `logs/`。

普通新 checkpoint 保存模型、优化器、scaler、步数、EMA 和随机数状态，`--resume_path` 可继续训练。旧 checkpoint 没有 next_step，需加 `--resume_step`。恢复训练会重新进入数据加载 epoch，不保证与未中断运行逐样本一致。Stage 2 续训仍需提供对应 Stage 1 权重；EMA 导出仅供推理。

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py --stage stage1 \
  --resume_path results/memoryfusion_R50_AOTL_Temp_pe_Slot_4/stage1/ckpt/latest_stage1.pth
```

## 测试

权重放在 `checkpoints/stage1.pth`、`checkpoints/stage2.pth`。

Stage 1（原入口实际使用的阶段，默认每序列最多 200 帧）：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py --output_path results/test_stage1
```

两阶段、全部帧：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py \
  --with_stage2 --max_frames 0 --output_path results/test_stage2
```

输出为 `<output_path>/<sequence>/1.jpg, 2.jpg, ...`；`--max_frames 0` 处理全部帧。测试严格加载所有模型参数，并明确进入 `eval()` 模式。原入口未显式设置 Stage 1 为 `eval()`，整理版关闭训练时的随机层后数值可能不同，核心网络和损失计算保持。

Stage 2 默认整段计算；显式 `--refine_window 30` 减小细化计算峰值显存。Stage 1 仍需装载整段序列。

