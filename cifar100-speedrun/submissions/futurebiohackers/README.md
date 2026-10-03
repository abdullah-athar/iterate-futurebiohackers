# futurebiohackers: faster CIFAR-100 training

A convolutional image classifier based on Keller Jordan's
[airbench](https://github.com/KellerJordan/cifar10-airbench), trained from scratch.
Three convolution groups of 64, 256 and 768 channels, three convolutions each with a
residual connection over the last two. The first group is deliberately narrow: it
runs at 31x31 and 15x15, where each channel costs 4-16x more than in the later
groups and cuDNN reaches only 23-34% of A100 peak; the capacity sits in the last
group, which runs at 3x3 at 78-89% of peak.

Training: 9.5 epochs at batch 1024, Nesterov SGD (lr 11.5, weight decay 0.017 per 1024
examples, momentum 0.85, BatchNorm-bias lr 32x), label smoothing 0.25, logit scale
1.25/9, BatchNorm momentum 0.5, lookahead weight average. The first half of the steps
trains on 28x28 bilinear downsamples of the images (0.77x the FLOPs), the second half
at 32x32. Augmentation: alternating flip, 2-pixel translation, per-image brightness
and contrast jitter of 0.2. Normalization and patch whitening of the training images
run inside the timer.

Implementation details that change the time but not the model:

- The SGD step is one fused CUDA kernel (`fused_sgd`).
- BatchNorm runs in fp16 like the rest of the network (`bn_dtype: "half"`).
- The forward pass and loss compile as one graph (`compile_step`), with static
  shapes so no recompilation happens inside a trial; every (batch, resolution)
  pair gets its own graph, warmed in `build`.
- Random crops use one gather per epoch (`crop_mode: "indexed"`) instead of 25
  masked copies, which also avoids host syncs.
- The final global pool is `max(dim).values`. `adaptive_max_pool2d` backward uses
  an atomic kernel (about 0.3 s per trial in the profile), and `amax` trains to
  NaN under `torch.compile` in torch 2.4.
- `prepare` takes 28 ms instead of 170 ms: the dirac part of the conv init is
  vectorized (`nn.init.dirac_` launches one kernel per channel), and the 28 px
  resize is two matmuls instead of `F.interpolate` (65 ms on fp16 channels-last).

## Development results

Confirmation runs for the submitted 9.5-epoch defaults (3 October): 40 trials per seed set, the
75% target enforced, cold `build`, Modal NVIDIA A100-SXM4-80GB at the 400 W power limit.

| Seeds | Recipe | Mean accuracy | Min / max trial | Mean preparation + training | Cold build | Qualified |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0-39 | **9.5 epochs (submitted defaults)** | **75.28% (sd 0.24)** | 74.71% / 75.85% | **4.96 s** (sd 0.01) | 196 s | yes |
| 40-79 | **9.5 epochs (submitted defaults)** | **75.32% (sd 0.29)** | 74.79% / 76.02% | **4.94 s** (sd 0.01) | 259 s | yes |
| 0-39 | 9.0 epochs (previous defaults) | 75.12% (sd 0.26) | | 4.72 s (sd 0.01) | 193 s | yes |

The 9.5-epoch defaults trade 0.24 s for about 0.2 points of margin over the 75% target. At 9
epochs the 40-seed mean sat within a few standard errors of the target (the previous estimate
below is a 10% chance of a 40-seed draw falling short); the two 40-seed sets above both clear an
internal floor of 75.15%. Under an emulated four-CPU container quota (the official judging
limit) the cold build of the 9-epoch graphs, which this recipe shares, took 255 s.

Earlier results for the 9-epoch recipe:

Each row pair ran back to back in one Modal A100-SXM4-80GB container on the same 40
random seeds (drawn like the organizer seed file), 75% target enforced. Seed sets
are independent draws; cards differ in speed, so compare within a pair.

| Seed set | Recipe | Mean accuracy | Mean preparation + training | Qualified |
| --- | --- | ---: | ---: | --- |
| F | **Current defaults** | **75.009% (sd 0.28)** | **4.646 s** (sd 0.010) | yes |
| F | PR #14 as first opened (128/256/768, 8.25 ep) | 75.193% (sd 0.25) | 5.445 s (sd 0.009) | yes |
| E | Current defaults before the `prepare` speedup | 75.083% (sd 0.25) | 5.321 s (sd 0.052) | yes |
| E | PR #14 as first opened | 75.150% (sd 0.23) | 6.153 s (sd 0.060) | yes |
| B | PR #14 as first opened | 75.130% (sd 0.29) | 6.048 s (sd 0.055) | yes |
| B | PR #9 defaults | 75.268% (sd 0.28) | 6.319 s (sd 0.042) | yes |

The current defaults are 13.5-14.7% faster than PR #14 as first opened, which was
3.2-4.3% faster than PR #9. The accuracy margin is thin: over 110 trials the current
defaults average about 75.06%, so a 40-seed draw falls below 75% with an estimated
10% probability. A batch-512, 8-epoch variant (`{"batch_size": 512, "lr": 16,
"epochs": 8}`) matched this accuracy 4% faster over 10 trials and is being validated
on five independent 40-seed sets; the defaults will follow that result.

Run the defaults from the repository root with `just modal 40`.

## Experiments and progress

Tested in same-GPU comparisons against a control (4-10 trials each):

| Option | Result |
| --- | --- |
| `batch_size: 512, lr: 16` | 8 epochs match 9 epochs of batch 1024, 4% faster; lr 11.5 or 18 loses 0.2 pt; 7.5 epochs fails |
| `batch_size: 768, lr: 14`, 8.5 epochs | 2% faster at equal accuracy |
| `batch_schedule` (512 during the 28 px phase) | pending |
| `resolution_schedule: [[24, 0.33], [28, 0.67]]` | 24 px costs 0.8 pt at 10.5 epochs; worse than cutting epochs on CIFAR-100 |
| `resolution_schedule: [[28, 0.6]]`, 9.25 epochs | same as 28 px for half |
| `momentum: 0.9` | -0.08 pt, same time |
| `whiten_bias_epochs: 1` | -0.08 pt, same time |
| `widths: [64, 256, 896]` | more accurate, slower; on the same accuracy/time line |
| `widths: [48, 256, 768]` or `[64, 192, 768]` | less accurate at equal time |
| `inductor_tuning` | coordinate-descent tuning: 0.5% faster, within noise |
| `optimizer: "muon"` (hiverge-style, batched) | 3.5 points less accurate at 8 epochs |
| `activation: "silu"` | 2% faster, 0.4 points less accurate |
| `stem: "patch4s2"` (4x4 stride-2 whitening) | 2-3 points less accurate |
| `inner_kernels: [3, 3, 1]` (1x1 convs in group 3) | 1.9 points less accurate |
| `pool_first` in group 3 | 1 point less accurate |
| `bn_recal_batches` | no gain |
| `ema_every: 0` | 0.6 points less accurate |
| Vision transformer (patch 4, dim 256, 6 layers) | 37% at 9 epochs, 58% at 30 epochs (39 s) |

Convolutions are about 65% of GPU time and Inductor's BatchNorm/activation kernels
about 22%. Inductor's CUDA graphs are on and worth about 1.5%; a whole-run graph
would add little since the GPU is already busy for the whole step.

`--params` exposes smaller early crops (24 or 28 pixels followed by 32), different
block widths/depths, proxy-based hard-example selection, alternative pooling and
optimizer settings, and an optional Triton crop/flip kernel. These experiments are
turned off in the selected defaults. In PR #5's tests, proxy-based selection did
not improve the qualifying result, and the unsuccessful experimental BN/GELU
fusion was excluded from the submitted source.

`scripts/modal_experiments.py` runs bounded comparisons through the unchanged
competition harness and reserves spending against a $50 cap.
`scripts/track_speedrun.py` writes a live log under
`artifacts/runtime-optimization/`, sorted by time, with each experiment's main
changes, accuracy, trial count and paired control. It also generates a local
review dashboard under `.lavish/`.

## Data and evaluation checks

Build warms compilation and kernels using random synthetic images only. Every
trial resets model weights, BatchNorm statistics, gradients, optimizer state,
moving averages and any proxy masks. Preparation and training use only the
harness-provided training images and labels. Evaluation uses one image view and
frozen training statistics; it does not fit to test images, use test-batch
statistics or change registered model state.

The [rules](https://github.com/AIDDA-Institute/CIFAR-100-speedrun/blob/main/RULES.md)
allow comparing reported test accuracy during development. Test results never
choose a stopping point or hard-example masks within a trial. Source review found
no test-data leakage; official acceptance still requires organizer review and the
prescribed environment.

The correctness checks cover resets, immutable training inputs, evaluation state
and batch independence (13 CPU tests), plus exact crop/flip equivalence across
layouts and resolutions (36 GPU checks). The GPU benchmark applies the harness's
own output and state checks. Run the CPU checks from `cifar100-speedrun/` with
`uv run python -m pytest -q ../scripts/test_speedrun_recipe.py`.

Adapted under the MIT license, Copyright (c) 2024 Keller Jordan. The full original
permission notice is preserved in `LICENSE.airbench`.
