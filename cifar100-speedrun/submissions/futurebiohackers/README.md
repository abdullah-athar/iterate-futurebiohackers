# futurebiohackers: ResNet9-style CIFAR-100 baseline

Plain PyTorch 2.4 recipe modelled on the organizers' verified reference (repository
README: ResNet9-style, 40 epochs, width 64, 75.36% mean accuracy, 59.30 s prepare + train
on an A100 80GB PCIe). No pretrained weights, no external data, nothing learned survives
between trials. Only `torch` and `benchmark.api` are imported.

## Recipe

| Part | Choice |
| --- | --- |
| Network | ResNet9 (DAWNBench design, `model.py`): conv(3→64) → conv(64→128) + maxpool → residual(128) → conv(128→256) + maxpool → conv(256→512) + maxpool → residual(512) → global maxpool → linear(512→100) × 0.125. 3×3 convolutions without bias, BatchNorm, ReLU. 6.6 M parameters at width 64. |
| Input | The evaluator convention, float32 in [0, 1]. Per-channel mean/std normalization is inside `forward()`, from statistics computed on the training split in `prepare` (never hard-coded). |
| Augmentation | Random 32×32 crop from the 4-pixel zero-padded image + random horizontal flip, on the device, one gather per batch, driven by a per-trial generator seeded with the trial seed. |
| Optimizer | SGD, Nesterov momentum 0.9, weight decay 5e-4 on every parameter, batch 512, last partial batch dropped (97 steps per epoch). |
| Schedule | Per step: linear warmup to lr 0.4 over the first 15% of steps, then linear decay to 0. 40 epochs = 3,880 steps. |
| Loss | Cross-entropy with label smoothing 0.1. |
| Precision | `auto`: bf16 autocast + channels_last on Ampere or newer CUDA GPUs (the A100); fp16 autocast + GradScaler on older CUDA GPUs (free Colab/Kaggle T4 or P100 accuracy checks); fp32 on CPU. Evaluation runs the eager fp32 model (TF32 enabled). |
| Compile | `use_compile` exists but is off: untested without a GPU. |

## Parameters

Pass with `--params '{"epochs": 10}'`. The defaults are the baseline; the official run needs no `--params`.

| Key | Default | Meaning |
| --- | --- | --- |
| `epochs` | 40 | passes over the 50,000 training images |
| `width` | 64 | base channel count (layers use 1×, 2×, 4×, 8×) |
| `batch_size` | 512 | training batch (capped at the dataset size for the synthetic fixtures) |
| `lr` | 0.4 | peak learning rate |
| `momentum` | 0.9 | SGD momentum (Nesterov when > 0) |
| `weight_decay` | 5e-4 | L2 weight decay on all parameters |
| `label_smoothing` | 0.1 | cross-entropy label smoothing |
| `warmup_fraction` | 0.15 | fraction of steps spent warming up to `lr` |
| `precision` | `auto` | `auto`, `bf16`, `fp16` or `fp32` |
| `use_compile` | false | wrap the training model in `torch.compile` |

Unknown keys are rejected so typos cannot silently fall back to the defaults.

## What runs where

- **`build` (untimed, once):** merge parameters, build the model on the device, enable cuDNN
  autotuning, then 3 synthetic forward/backward/optimizer steps on random uint8 images plus
  two synthetic eval forwards (batch 1024 and the 784-image final batch). No real data, no
  trial seed; the synthetic generator uses the constant 0.
- **`prepare` (timed, every trial):** `reset_parameters()` on every module (weights, BatchNorm
  running statistics and counters) using the torch RNG the harness just seeded with the trial
  seed, drop gradients, new optimizer (empty momentum buffers), new GradScaler if fp16, new
  `torch.Generator(seed)` for shuffling and augmentation, copy images and labels to the
  device, compute the normalization statistics from the training split.
- **`train` (timed):** the loop above. Returns `state.model`, the eager module.
- **Evaluation:** one forward per batch in `eval()` mode, no augmentation, no state change.

## Local checks

```bash
uv run python -m benchmark.run --submission futurebiohackers --device cpu --synthetic --n 2
uv run python -m benchmark.data --root data
uv run python -m benchmark.run --submission futurebiohackers --device cpu --n 2 \
    --no-accuracy-target --eval-timeout 120 --params '{"epochs": 1, "width": 16}'
uv run python -m benchmark.run --submission futurebiohackers --n 1          # GPU, real recipe
```

On CPU the 10,000-image test pass exceeds the 5 s evaluation limit even at width 16, so
development runs on CPU pass `--eval-timeout`. The GPU run uses the official limits.
