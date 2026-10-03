# futurebiohackers: faster CIFAR-100 training

This is a convolutional image classifier based on Keller Jordan's
[airbench](https://github.com/KellerJordan/cifar10-airbench), trained from scratch.
The selected model uses three convolution blocks with 128, 256 and 768 channels;
the first block has two convolutions and the others three.
It makes the first two blocks smaller, where images are largest and processing is
expensive, and gives the final block more capacity to distinguish the 100 classes.

Training uses 8.25 epochs, batches of 1024 images, half precision, channels-last
memory layout, Nesterov SGD and label smoothing. A moving average stabilizes the
weights. Training-image normalization and patch whitening happen inside the timer.
The final spatial pooling now covers the whole feature map, so experimental
smaller-crop training also works.

## Development results

Paired 40-trial runs on a Hugging Face Jobs NVIDIA A100-SXM4-80GB (400 W), seeds 0-39,
each job running the recipes back to back in one container
(`scripts/hf_ab.sh 40 '[...]'` from the repository root):

| Recipe | Mean accuracy | Mean preparation + training |
| --- | ---: | ---: |
| Selected: depths 2/3/3, widths 128/256/768, lr 11.5, wd 0.017, BN momentum 0.5, 8.25 epochs | 75.10% (± 0.25 pp) | 5.843 s |
| Selected, second job | 75.19% (± 0.18 pp) | 5.865 s |
| Selected at 8.5 epochs | 75.27% (± 0.22 pp) | 6.004 s |
| Previous 96/256/768, lr 9.0, wd 0.012, BN momentum 0.6, 8.5 epochs | 74.91% (± 0.25 pp) | 6.114 s |

The accuracy margin is small; 8.5 epochs is the safer fallback. Timings on the official
A100 80GB PCIe will differ.

### Earlier Modal screen

The latest completed screen compares both recipes on the same Modal A100 SXM GPU
allocation, with three trials each:

| Recipe | Mean accuracy | Mean preparation + training |
| --- | ---: | ---: |
| Original 128/384/576 model | 75.150% | 7.991 s |
| Selected 96/256/768 model | 75.143% | 6.501 s |

The selected model was **18.6% faster** in this screen. A fresh 40-trial comparison
is running; these three-trial results do not establish challenge qualification.
The under-three-second target has not been reached. Modal A100 SXM measurements
are development results; official judging requires an A100 80GB PCIe and the
organizer's 40 seeds.

The original recipe in PR #3 previously reached 75.29% in 8.11 s across 40 trials
on another Modal allocation. Allocation differences make its time unsuitable as
a paired control for this screen.

Run the selected defaults from the repository root with `just modal 40`.
The three-trial selected screen is `20261003T153812Z-d85ef349`; its paired control
is `20261003T152954Z-bb27a23e`.

## Tuning log: what worked and what didn't

Paired screens on Hugging Face Jobs A100-SXM4-80GB (`scripts/hf_ab.sh`): each job runs the
control (that round's defaults) and its variants back to back in one container with the same
seeds, so differences within a job are comparable; absolute times drift about 0.1-0.2 s
between containers. Trials per arm in brackets. Accuracy std of a 10-trial mean is about
0.08 pp, so gaps under about 0.15 pp are noise.

### Worked (kept)

Against #5's defaults (96/256/768, lr 9.0, wd 0.012, BN momentum 0.6, 8.5 epochs), whose
controls averaged 74.84-75.02% at 5.94-6.20 s:

| Change | Result vs paired control |
| --- | --- |
| lr 11.5 + wd 0.015 | 75.31% vs 74.92%, same time [10] |
| lr 10.5 alone / wd 0.015 alone | 75.05% / 75.09% vs ~74.92%, same time [10] |
| BN momentum 0.5 (with lr 10.5, wd 0.015) | 75.24% vs 75.02%, same time [10] |
| First group 2 convs at width 128 (`depths` [2, 3, 3], widths 128/256/768) | 74.97% vs 74.91%, 5.92 vs 6.01 s [10] |
| All of the above, wd 0.017, 8.25 epochs (selected) | 75.19% vs 74.91%, 5.87 vs 6.11 s [40] |

Against the earlier #3 recipe (128/384/576, 8.5 epochs), the move to a narrow first group and
wide last group was the big win: 96/384/768 at 8 epochs gave 75.25-75.37% vs 75.17-75.24%,
0.25 s faster [8-10]; 96/320/768 at 8 epochs with label smoothing 0.35 gave 75.09% in 6.36 s
vs 7.28 s [10]. #5 had found the same direction (96/256/768).

### Didn't work

| Change | Result vs paired control |
| --- | --- |
| Turning off lookahead EMA (`ema_every` 0) | 74.09% vs 75.02%, no time saved [10] |
| Batch 2048 (lr 11) | -0.8 pp at 8.5 epochs; at 9.5 epochs it is slower [8] |
| Batch 1536 | -0.3 pp on #5, about 0.07 s faster [10] |
| Label smoothing 0.35 / 0.25 | within noise or worse on #5 [10] |
| Warmup 0.15, final_lr 0 | within noise [8-10] |
| Momentum 0.9 (lr 7.0), lr 7.5, lr 12.5 | -0.1 to -0.3 pp [10] |
| BN momentum 0.4 | no better than 0.5 [10] |
| wd 0.018 alone | no better than 0.015 [10] |
| First group 80 or 64 wide (3 convs) | -0.3 pp; faster but under 75% [10] |
| 2-conv first group at width 96 | 74.84%, under 75% [10] |
| Last group 832 or 896 wide | +0.1-0.2 pp but +0.5-0.7 s [10] |
| 7.75 epochs (any tuned variant) | 74.85-74.93%, under 75% [10] |
| 8 epochs with the selected settings | 75.05% [20]: too close to the bar |

### Robustness of the selected recipe

| Run | Selected | #5 defaults |
| --- | ---: | ---: |
| Seeds 0-39 | 75.19%, 5.865 s | 74.91%, 6.114 s |
| Seeds 0-39 (second container) | 75.10%, 5.843 s | n/a (8.5-epoch arm: 75.27%, 6.004 s) |
| Random seeds from 602989454 | 75.15%, 5.887 s | 74.82%, 6.135 s |
| Random seeds from 934607449 | 75.11%, 5.845 s | 74.90%, 6.115 s |

All arms are 40 trials. Random starts came from `secrets.randbelow(10**9)`; run with
`SEED=<start> scripts/hf_ab.sh 40 '[...]'`.

## Experiments and progress

`--params` exposes smaller early crops (24 or 28 pixels followed by 32), different
block widths/depths, proxy-based hard-example selection, alternative pooling and
optimizer settings, and an optional Triton crop/flip kernel. These experiments are
turned off in the selected defaults. Reduced-resolution candidates lost their
accuracy margin in repeated trials; proxy selection and fused optimizer settings
also failed to improve the qualifying result. The unsuccessful experimental
BN/GELU fusion was excluded from the submitted source.

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
