# futurebiohackers: faster CIFAR-100 training

A convolutional image classifier based on Keller Jordan's
[airbench](https://github.com/KellerJordan/cifar10-airbench), trained from scratch.
Three convolution groups of 64, 256 and 768 channels, three convolutions each with a
residual connection over the last two; in the last group that residual pair runs through 512
channels (3x3 768 -> 512 -> 768) and in the second group through 192 channels (3x3 256 -> 192
-> 256), while each group's input, residual and output keep its width. The classifier reads the
per-channel maximum plus the per-channel mean of the last group's map. The first group is deliberately narrow: it
runs at 31x31 and 15x15, where each channel costs 4-16x more than in the later
groups and cuDNN reaches only 23-34% of A100 peak; the capacity sits in the last
group, which runs at 3x3 at 78-89% of peak.

Training: 10 epochs at batch 1024, Nesterov SGD (lr 11.5, weight decay 0.017 per 1024
examples, momentum 0.85, BatchNorm-bias lr 32x), label smoothing 0.25, logit scale
1.25/9, BatchNorm momentum 0.5, lookahead weight average. The first quarter of the
examples trains on 24x24 bilinear downsamples of the images (0.56x the FLOPs), the
second quarter on 28x28 (0.77x), the second half at 32x32. Augmentation: alternating flip, 2-pixel translation, per-image brightness
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
- The global pooling (max + mean, described above) is written with `F.max_pool2d` and
  `F.avg_pool2d` over the whole map (`global_pool: "fullpool_avgsum"`), which `torch.compile`
  lowers to pointwise kernels; the reduction kernels it generates for `max(dim).values` and
  `mean` (PR #24) cost 0.13 s more per trial at identical results. `adaptive_max_pool2d`
  backward uses an atomic kernel (about 0.3 s per trial in the profile), and `amax` trains to
  NaN under `torch.compile` in torch 2.4.
- `prepare` takes 35-85 ms instead of 170 ms: the dirac part of the conv init is
  vectorized (`nn.init.dirac_` launches one kernel per channel), and the 28 px
  resize is two matmuls instead of `F.interpolate` (65 ms on fp16 channels-last).

## Development results

Confirmation runs for the submitted defaults (192-channel residual pair in the second group,
512-channel pair in the last, max + mean pooling through the pooling kernels, 24 px first
quarter, 28 px second quarter, 10 epochs), 4 October: 40 trials per seed set, the 75% target enforced, Modal NVIDIA A100-SXM4-80GB
at the 400 W power limit ("A100 SXM 400 W"). Cards at the same power limit differ by one to two
percent in speed, so each seed set ran this recipe and the previous one (PR #23) back to back in
one container with a warm compile cache; the cold build is reported separately below. Official
judging runs on an A100 80GB PCIe, where times are higher (about 8% on the earlier recipe); the
ranking between recipes carries over.

| Seeds | Recipe | Mean accuracy | Min / max trial | Mean preparation + training | Qualified |
| --- | --- | ---: | ---: | ---: | --- |
| 0-39 | **submitted defaults** | **75.16% (sd 0.23)** | 74.52% / 75.60% | **4.23 s** (sd 0.02) | yes |
| 0-39 | previous recipe (PR #23), same container | 75.20% (sd 0.27) | | 4.50 s | yes |
| 40-79 | **submitted defaults** | **75.19% (sd 0.28)** | 74.64% / 76.16% | **4.15 s** (sd 0.01) | yes |
| 40-79 | previous recipe (PR #23), same container | 75.22% (sd 0.25) | | 4.42 s | yes |
| 0-39 / 40-79 | PR #24 (same pair, max + mean through the compiled reductions), paired vs PR #23 | 75.13% / 75.20% | | 4.34 / 4.33 s (-0.12 / -0.14 s) | yes |
| 0-39 / 40-79 | PR #23 standalone, cold build 381 / 278 s | 75.19% / 75.25% | | 4.56 / 4.49 s | yes |
| 0-39 / 40-79 | 768-wide pair, 9.5 epochs (PR #22), cold build 275 / 287 s | 75.18% / 75.25% | | 4.51 / 4.55 s | yes |
| 0-39 | 9.0 epochs, 28 px first half (PR #14 defaults), cold build 193 s | 75.12% (sd 0.26) | | 4.72 s (sd 0.01) | yes |
| 0-39 / 40-79 | 9.5 epochs, 28 px first half (PR #21), cold build 196 / 259 s | 75.28% / 75.32% | | 4.96 / 4.94 s | yes |

Paired differences against PR #23 on the same seeds and card: seeds 0-39 -0.05 +- 0.06 points,
-0.26 s; seeds 40-79 -0.02 +- 0.06 points, -0.27 s. No trial produced a non-finite loss (counted
on the device in every run).

Why this recipe: three changes on PR #23. The residual pair of the second group through 192
channels, together with the max + mean pooling, saves 0.13 s for about 0.05 points (PR #24: 40
paired trials per seed set, -0.07 +- 0.06 and -0.02 +- 0.06 points, -0.12 and -0.14 s; the pair
alone was screened at 8 trials: +0.07 +- 0.15 points, -0.13 s). The final pooling's max and mean
are written as `F.max_pool2d` / `F.avg_pool2d`, which `torch.compile` lowers to pointwise kernels
instead of the reduction kernels it generates for `max(dim).values` and `mean`: identical
results and another 0.13 s (the max alone through `F.max_pool2d`: -0.03 +- 0.08 points, -0.14 s
over 16 paired trials). The per-channel mean added to the max costs no time; it measured +0.30 +- 0.14
points in an 8-trial screen and about +0.06 points over 80 paired trials on PR #23. Together:
-0.26 / -0.27 s for -0.05 +- 0.06 / -0.02 +- 0.06 points on the two seed sets (16 paired trials:
-0.06 +- 0.11 points, -0.24 s). Fewer epochs do not pay: with the compiled pooling, 9.875 epochs
lost 0.10 points for 0.03 s and 9.75 epochs about 0.20 points for about 0.09 s. The residual pair of the last group is about a
third of the network's FLOPs, and running it through 512 channels saves 0.31 s for 0.35 points
(paired 8-trial screens at 400 W); half an epoch more buys the points back at a net gain of
about 0.1 s (PR #23). The 24x24 first quarter in front of the 28x28 phase saves 0.38 s at equal
accuracy, and the epoch ladder is linear at about 1.05 points per second. Every optimizer and
augmentation knob was re-screened on this schedule (learning rate, weight decay, BatchNorm-bias
learning rate, momentum, BatchNorm momentum, label smoothing, warmup, final learning rate,
lookahead period, translation, jitter, cutout, batch 512/768, batch schedules, logit scale): all
flat or worse. Under an emulated four-CPU container quota (the official judging limit) the cold
build of exactly this configuration took 289 s (limit 600 s; PR #24's took 280 s, PR #23's 268 s).

Earlier results for the 9-epoch recipe:

Each row pair ran back to back in one Modal A100-SXM4-80GB container on the same 40
random seeds (drawn like the organizer seed file), 75% target enforced. Seed sets
are independent draws; cards differ in speed, so compare within a pair.

| Seed set | Recipe | Mean accuracy | Mean preparation + training | Qualified |
| --- | --- | ---: | ---: | --- |
| F | PR #14 defaults (9 epochs, 28 px first half) | 75.009% (sd 0.28) | 4.646 s (sd 0.010) | yes |
| F | PR #14 as first opened (128/256/768, 8.25 ep) | 75.193% (sd 0.25) | 5.445 s (sd 0.009) | yes |
| E | PR #14 defaults before the `prepare` speedup | 75.083% (sd 0.25) | 5.321 s (sd 0.052) | yes |
| E | PR #14 as first opened | 75.150% (sd 0.23) | 6.153 s (sd 0.060) | yes |
| B | PR #14 as first opened | 75.130% (sd 0.29) | 6.048 s (sd 0.055) | yes |
| B | PR #9 defaults | 75.268% (sd 0.28) | 6.319 s (sd 0.042) | yes |

PR #14's defaults were 13.5-14.7% faster than PR #14 as first opened, which was 3.2-4.3%
faster than PR #9; their accuracy margin was thin (about 75.06% over 110 trials, an estimated
10% chance of a 40-seed draw below 75%), which the later PRs addressed with more epochs and
the changes above. The batch-512, 8-epoch variant mentioned in earlier revisions was not
adopted.

Run the defaults with the harness from `cifar100-speedrun/`:
`uv run python -m benchmark.run --submission futurebiohackers --n 40`
(`just modal 40` is the team repository's Modal wrapper around the same command).

## Experiments and progress

Tested in same-GPU comparisons against a control (8 to 40 paired trials each):

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
| `g2_pair: "inner192"` (this PR) | -0.13 s for about 0.05 points; group 3 pairs at 448 / 384 sit on the accuracy/time line; a 1x1-3x3-1x1 bottleneck in group 3 loses 1.4 points for 0.7 s |
| `global_pool: "fullpool_avgsum"` (this PR), `"maxmean_sum"` (PR #24), `"maxmean_cat"` | the same max + mean through `F.max_pool2d` / `F.avg_pool2d`: -0.13 s at identical results; the summed max + mean measured +0.3 points in an 8-trial screen (+0.2 for the concatenated one) and about +0.06 at 40 trials |
| squeeze-and-excitation on groups 2-3, multi-scale head | break-even and -0.3 points |
| wider group 1 (`widths: [128, 256, 768]`) | on or below the accuracy/time line |
| `inductor_tuning` | coordinate-descent tuning: 0.5% faster, within noise |
| `optimizer: "muon"` (hiverge-style, batched) | 3.5 points less accurate at 8 epochs |
| `activation: "silu"` | 2% faster, 0.4 points less accurate |
| `stem: "patch4s2"` (4x4 stride-2 whitening) | 2-3 points less accurate |
| `inner_kernels: [3, 3, 1]` (1x1 convs in group 3) | 1.9 points less accurate |
| `pool_first` in group 3 | 1 point less accurate |
| `bn_recal_batches` | no gain |
| `ema_every: 0` | 0.6 points less accurate |
| Vision transformer (patch 4, dim 256, 6 layers) | 37% at 9 epochs, 58% at 30 epochs (39 s) |

Convolutions are about 75% of GPU time and Inductor's BatchNorm/activation kernels about
20% (profile of this recipe: GPU busy 99%, no host synchronisation inside the step, about
180 kernels per step). Inductor's CUDA graphs are on; a whole-run graph would add little
since the GPU is already busy for the whole step.

`--params` exposes other resolution schedules, block widths/depths and residual-pair widths,
proxy-based hard-example selection, other pooling and optimizer settings, and an optional
Triton crop/flip kernel; these are off in the defaults (the 24/28 px schedule and the max + mean
pooling are on). In PR #5's tests, proxy-based selection did not improve the qualifying result,
and the unsuccessful experimental BN/GELU fusion was excluded from the submitted source.

Team-repository tooling (not part of this folder): `scripts/modal_experiments.py` runs bounded
comparisons through the unchanged competition harness, and `scripts/track_speedrun.py` writes
a live log of the experiments with each one's main changes, accuracy, trial count and paired
control.

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
own output and state checks. The CPU checks live in the team repository
(`scripts/test_speedrun_recipe.py`; from `cifar100-speedrun/`:
`uv run python -m pytest -q ../scripts/test_speedrun_recipe.py`).

Adapted under the MIT license, Copyright (c) 2024 Keller Jordan. The full original
permission notice is preserved in `LICENSE.airbench`.
