# futurebiohackers: faster CIFAR-100 training

A convolutional image classifier based on Keller Jordan's
[airbench](https://github.com/KellerJordan/cifar10-airbench), trained from scratch.
Three convolution groups of 64, 256 and 768 channels, three convolutions each with a
residual connection over the last two; in the last group that residual pair runs through 512
channels (3x3 768 -> 512 -> 768) while its input, residual and output stay at 768; in the first
and second groups the pair is a 3x3 followed by a 1x1 convolution (at 64 and 256 channels). The classifier
reads the per-channel maximum plus the per-channel mean of the last group's map. The first group is deliberately narrow: it
runs at 31x31 and 15x15, where each channel costs 4-16x more than in the later
groups and cuDNN reaches only 23-34% of A100 peak; the capacity sits in the last
group, which runs at 3x3 at 78-89% of peak.

Training: 11 epochs at batch 1024, Nesterov SGD (lr 11.5, weight decay 0.017 per 1024
examples, momentum 0.85, BatchNorm-bias lr 32x), label smoothing 0.25, logit scale
1.25/9, BatchNorm momentum 0.5, lookahead weight average. The first 15% of the
examples train on 20x20 bilinear downsamples of the images (0.39x the FLOPs), the next 35% on
24x24 (0.56x), the second half at 32x32. Augmentation: alternating flip, 2-pixel translation, per-image brightness
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
  `mean` cost 0.13 s more per trial at identical results. `adaptive_max_pool2d` backward uses
  an atomic kernel (about 0.3 s per trial in the profile), and `amax` trains to NaN under
  `torch.compile` in torch 2.4.
- `prepare` takes 35-85 ms instead of 170 ms: the dirac part of the conv init is
  vectorized (`nn.init.dirac_` launches one kernel per channel), and each
  downsampling resize is two matmuls instead of `F.interpolate` (65 ms on fp16 channels-last).

## Development results

Confirmation runs for the submitted defaults (20 px for the first 15% of the examples, 24 px to the
half and 32 px after; the residual pairs of groups 1 and 2 as a 3x3 then a 1x1 convolution at full width; the 512-channel pair in the
last group; max + mean pooling through the pooling kernels; 11 epochs), 4 October: 40 trials per
seed set, the 75% target enforced, Modal NVIDIA A100-SXM4-80GB at the 400 W power limit ("A100 SXM
400 W"). Cards at the same power limit differ by one to two percent in speed, so each seed set ran
this recipe and the previous one (PR #28) back to back in one container, both compiled cold (fresh
Inductor cache) under a hard four-CPU limit, the judges' quota; the build times of the first four rows are
those real cold builds (the older rows' cold builds ran in 20-CPU containers). Official judging runs on an A100 80GB PCIe, where times are higher (about 8-9%
on the earlier recipe); the ranking between recipes carries over.

| Seeds | Recipe | Mean accuracy | Min / max trial | Mean preparation + training | Qualified |
| --- | --- | ---: | ---: | ---: | --- |
| 0-39 | **submitted defaults**, cold build 380 s at 4 CPUs | **75.32% (sd 0.28)** | 74.81% / 75.98% | **3.96 s** (sd 0.01) | yes |
| 0-39 | previous recipe (PR #28), same container, cold build 326 s | 75.19% (sd 0.23) | 74.66% / 75.71% | 4.24 s | yes |
| 40-79 | **submitted defaults**, cold build 350 s at 4 CPUs | **75.28% (sd 0.27)** | 74.77% / 75.91% | **3.95 s** (sd 0.01) | yes |
| 40-79 | previous recipe (PR #28), same container, cold build 299 s | 75.18% (sd 0.27) | 74.63% / 75.78% | 4.25 s | yes |
| 0-39 / 40-79 | PR #28 paired vs PR #25 in one container, warm compile cache | 75.24% / 75.23% vs 75.25% / 75.31% | | 4.14 / 4.24 s vs 4.29 / 4.38 s | yes |
| 0-39 / 40-79 | PR #28 vs PR #25, both compiled cold at 4 CPUs (builds 267 / 272 s) | 75.25% / 75.26% vs 75.24% / 75.27% | | 4.32 / 4.14 s vs 4.46 / 4.28 s | yes |
| 0-39 / 40-79 | PR #25 paired vs PR #23 in one container | 75.24% / 75.31% vs 75.20% / 75.22% | | 4.33 / 4.23 s vs 4.47 / 4.36 s | yes |
| 0-39 / 40-79 | PR #23 standalone, cold build 381 / 278 s | 75.19% / 75.25% | | 4.56 / 4.49 s | yes |
| 0-39 / 40-79 | 768-wide pair, 9.5 epochs (PR #22), cold build 275 / 287 s | 75.18% / 75.25% | | 4.51 / 4.55 s | yes |
| 0-39 | 9.0 epochs, 28 px first half (PR #14 defaults), cold build 193 s | 75.12% (sd 0.26) | | 4.72 s (sd 0.01) | yes |
| 0-39 / 40-79 | 9.5 epochs, 28 px first half (PR #21), cold build 196 / 259 s | 75.28% / 75.32% | | 4.96 / 4.94 s | yes |

Paired differences against PR #28 on the same seeds and card: seeds 0-39 +0.13 +- 0.06 points,
-0.27 s; seeds 40-79 +0.10 +- 0.06 points, -0.31 s (+0.11 +- 0.04 points over the 80
pairs). All 160 trials of the two containers finished and qualified, none below 74.63%, so none
diverged (this file carries no device-side non-finite counter).

Why this recipe: two changes on PR #28 and one more epoch, every comparison below made with both
recipes compiled cold in one container unless stated otherwise (which removes the one-to-two-percent speed differences between cards
and any kernel choice inherited from a warm compile cache). (1) The first resolution phase trains at
20 px instead of 24 px and ends at 15% of the examples, and the 24 px phase then runs to the half in
place of the 28 px phase; on this recipe at 10.5 epochs the schedule alone measured -0.04 +- 0.05
points for -0.11 s (16 paired trials), and on the 192-channel recipe of PR #27 at 10 epochs
-0.23 +- 0.07 points for -0.40 s (8 trials): the lost accuracy is bought back with more epochs at a
net gain, since the epoch ladder is linear at about 1.05 points per second. Schedules that keep a
28 px phase (20 -> 24 -> 28 -> 32) save less and need eight compiled graphs, whose four-CPU cold
build measured 441 s on a sibling schedule ([[20, 0.15], [24, 0.3], [28, 0.5]]). (2) The residual pairs of groups 1 and 2 become a 3x3 followed by a
1x1 at full width: group 1's pair on PR #27 -0.13 s at +0.02 +- 0.13 points (8 paired trials), group
2's about -0.25 s for roughly -0.05 points at 10 epochs (40 paired trials per seed set on PR #25's
recipe, warm compile cache; on this recipe, cold, 10.5 epochs: +0.02 +- 0.10 points / -0.07 s and
10.75 epochs: +0.19 +- 0.06 / +0.04 s, 16 trials each, against the 10-epoch control).
Stacked, the three changes measured -0.06 +- 0.09 points / -0.41 s at 10.5 epochs and -0.05 +- 0.11 /
-0.33 s at 10.75 (16 paired trials each), so the submitted recipe takes 11 epochs for the accuracy
margin. Without group 2's pair, the schedule plus group 1's pair at 10.5 epochs measured
+0.03 +- 0.09 points for -0.20 s (16 trials).

The rest is PR #28, one change on PR #25: the 24 px phase runs for the first 35% of the examples
instead of the first quarter, so the 28 px phase shrinks from a quarter to 15%. It saves 0.15 /
0.14 s per trial for -0.01 +- 0.05 / -0.07 +- 0.06 points on the two seed sets (-0.04 +- 0.04 over
the 80 paired trials); the screens before it: -0.12 s at +0.15 +- 0.09 points over 8 paired trials
and -0.13 s at -0.01 +- 0.06 over 16 on the group-2-pair recipe of PR #26, -0.15 s at -0.07 +- 0.08
over 16 on PR #25's recipe. The 35% point comes from the schedule ladder: on the 9-epoch PR #14 recipe a 24 px first half saved
0.73 s for 0.34 +- 0.11 points (8 paired trials, below 75%), and a 28 px phase to 60% was neutral on
PR #26's recipe. On PR #26's recipe the
same change lands at 75.07% on seeds 40-79, below that PR's floor, so it is proposed on this one.
The rest is PR #25: the final pooling's max and mean are written as `F.max_pool2d` and
`F.avg_pool2d`, which `torch.compile` lowers to pointwise kernels instead of the reduction
kernels it generates for `max(dim).values` and `mean`; the result is identical and 0.13 s per
trial faster (the max alone through `F.max_pool2d`: -0.03 +- 0.08 points,
-0.14 s over 16 paired trials; with the mean through `F.avg_pool2d`: +0.11 +- 0.09 points, -0.12 s;
the 40-trial pairs above: +0.03 +- 0.06 and +0.09 +- 0.06 points, -0.13 s on both seed sets).
Adding the per-channel mean to the max costs no time; it measured +0.30 +- 0.14 points in an
8-trial screen and about +0.06 points over the 80 paired trials. The residual pair of the last
group is about a third of the network's FLOPs, and running it through 512 channels saves 0.31 s
for 0.35 points (paired 8-trial screens at 400 W); half an epoch more buys the points back at a
net gain of about 0.1 s (PR #23). The 24x24 first quarter in front of the 28x28 phase saves
0.38 s at equal accuracy, and the epoch ladder is linear at about 1.05 points per second. Every
optimizer and
augmentation knob was re-screened on this schedule (learning rate, weight decay, BatchNorm-bias
learning rate, momentum, BatchNorm momentum, label smoothing, warmup, final learning rate,
lookahead period, translation, jitter, cutout, batch 512/768, batch schedules, logit scale): all
flat or worse. Under the four-CPU container quota (the official judging limit) the cold build of exactly this
configuration took 380 and 350 s in the two confirmation containers (limit 600 s; PR #28's
folder built in 326 and 299 s in the same containers, in 267-272 s when PR #28 was verified). The
six compiled graphs (three resolutions, two whitening-bias flags) are the same as PR #28's; the
20 px phase and the 1x1 change the kernels inside them, not their number.

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
| `resolution_schedule: [[20, 0.15], [24, 0.5]]` (this PR) | 20 px for the first 15% of the examples, 24 px to the half, 32 px after: on this recipe at 10.5 epochs -0.04 +- 0.05 points / -0.11 s (16 paired trials, cold); on PR #27's recipe at 10 epochs -0.23 / -0.40 s (8 trials); 20 -> 24 -> 28 -> 32 (eight graphs): less saving and a 441 s four-CPU build |
| `inner_kernels: [[3, 1], [3, 1], 3]` (this PR) | groups 1 and 2 as 3x3 then 1x1 at full width: group 2's alone about -0.25 s at 10 epochs for roughly -0.05 points (40 paired trials per seed set on PR #25's recipe, warm cache); the full stack with the schedule -0.06 +- 0.09 / -0.41 s at 10.5 epochs, -0.05 +- 0.11 / -0.33 s at 10.75 (16 paired trials each, cold) |
| `resolution_schedule: [[24, 0.35], [28, 0.5]]` (PR #28) | -0.15 / -0.14 s at -0.01 / -0.07 points over 40 paired trials per seed set vs PR #25 (warm cache; -0.14 / -0.15 s cold at 4 CPUs) |
| `resolution_schedule: [[24, 0.33], [28, 0.67]]` (9-epoch recipe) | 24 px costs 0.8 pt at 10.5 epochs; worse than cutting epochs on CIFAR-100 |
| `resolution_schedule: [[28, 0.6]]`, 9.25 epochs | same as 28 px for half |
| `momentum: 0.9` | -0.08 pt, same time |
| `whiten_bias_epochs: 1` | -0.08 pt, same time |
| `widths: [64, 256, 896]` | more accurate, slower; on the same accuracy/time line |
| `widths: [48, 256, 768]` or `[64, 192, 768]` | less accurate at equal time |
| `global_pool: "fullpool_avgsum"` (PR #25) | max through `F.max_pool2d`: -0.14 s at 0 points (16 paired trials); max + mean through `F.max_pool2d` / `F.avg_pool2d`: -0.12 s, +0.11 +- 0.09 points; the mean through the compiled reduction gives back half the time |
| `g2_pair: "inner192"` (PR #24) | -0.13 s for about 0.05 points; group 3 pairs at 448 / 384 sit on the accuracy/time line; a 1x1-3x3-1x1 bottleneck in group 3 loses 1.4 points for 0.7 s |
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
Triton crop/flip kernel; these are off in the defaults (the 20/24 px schedule, the mixed residual pairs and the max + mean
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
