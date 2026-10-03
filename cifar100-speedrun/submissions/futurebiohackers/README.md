# futurebiohackers: faster CIFAR-100 training

A convolutional image classifier based on Keller Jordan's
[airbench](https://github.com/KellerJordan/cifar10-airbench), trained from scratch.
Three convolution groups with 128, 256 and 768 channels; the first group has two
convolutions, the others three with a residual connection. Training runs for
8.25 epochs at batch 1024 with Nesterov SGD (lr 11.5, weight decay 0.017 per 1024
examples), label smoothing 0.3, BatchNorm momentum 0.5 and a lookahead weight
average. Normalization and patch whitening of the training images run inside the
timer.

Implementation details that change the time but not the model:

- The SGD step is one fused CUDA kernel (`fused_sgd`).
- BatchNorm runs in fp16 like the rest of the network (`bn_dtype: "half"`).
- The forward pass and loss compile as one graph (`compile_step`), with static
  shapes so no recompilation happens inside a trial.
- Random crops use one gather per epoch (`crop_mode: "indexed"`) instead of 25
  masked copies, which also avoids host syncs.
- The final global pool is `max(dim).values`. `adaptive_max_pool2d` backward uses
  an atomic kernel (about 0.3 s per trial in the profile), and `amax` trains to
  NaN under `torch.compile` in torch 2.4.

## Development results

40 random seeds (drawn like the organizer seed file), both recipes back to back
in one Modal A100-SXM4-80GB container, 75% target enforced:

| Seed set | Recipe | Mean accuracy | Mean preparation + training | Qualified |
| --- | --- | ---: | ---: | --- |
| A | This recipe without fused SGD | 75.094% (sd 0.25) | 5.627 s (sd 0.015) | yes |
| A | PR #9 defaults | 75.169% (sd 0.24) | 5.816 s (sd 0.016) | yes |
| B | **This recipe (defaults, fused SGD)** | **75.130% (sd 0.29)** | **6.048 s** (sd 0.055) | yes |
| B | PR #9 defaults | 75.268% (sd 0.28) | 6.319 s (sd 0.042) | yes |

Seed sets A and B are independent random draws; each row pair ran in one
container. Set B landed on a slower card, so compare within a pair: the recipe is
3.2% faster than PR #9 without fused SGD and 4.3% faster with it. Its accuracy
margin is thin: across 105 trials it averages about 75.11%, so a different
40-seed draw falls below 75% with an estimated 2-3% probability.
Official judging uses an A100 80GB PCIe (300 W), which will be slower than these
SXM (400 W) timings.

Run the defaults from the repository root with `just modal 40`.

## Experiments and progress

Tested in same-GPU comparisons against a control (4-10 trials each) and left off
by default:

| Option | Result |
| --- | --- |
| `resolution_schedule: [[28, 0.5]]` | 28 px for the first half: about 3% better than cutting epochs |
| `widths: [64, 256, 768]`, depths 3, 9 epochs | 8.4% faster than PR #9 at -0.06 pt (5 trials) |
| `inductor_tuning` | coordinate-descent tuning: 0.5% faster, within noise |
| `optimizer: "muon"` (hiverge-style, batched) | 3.5 points less accurate at 8 epochs |
| `activation: "silu"` | 2% faster, 0.4 points less accurate |
| `color_jitter` | no accuracy gain on CIFAR-100 |
| `stem: "patch4s2"` (4x4 stride-2 whitening) | 2-3 points less accurate |
| `inner_kernels: [3, 3, 1]` (1x1 convs in group 3) | 1.9 points less accurate |
| `bn_recal_batches` | no gain |
| `ema_every: 0` | 0.6 points less accurate |

Per-layer cuDNN throughput at batch 1024 ranges from 23-34% of A100 fp16 peak for
the first 31x31 convolution to 78-89% for the 768-channel convolutions at 3x3.
Convolutions are about 65% of GPU time and Inductor's BatchNorm/activation kernels
about 22%.

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
