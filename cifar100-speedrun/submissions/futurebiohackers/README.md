# futurebiohackers: airbench-style CIFAR-100 recipe

An adaptation of Keller Jordan's [airbench](https://github.com/KellerJordan/cifar10-airbench)
(MIT, Copyright (c) 2024 Keller Jordan; the original permission notice is in
`LICENSE.airbench`) to CIFAR-100, trained from scratch with no test-time augmentation.

## Recipe

- **Network**: frozen 2x2 patch-whitening conv (initialized from 5,000 training images in
  `prepare`), then three conv groups of widths 128/256/768. The first group has two 3x3
  convs, the others three with a residual over the last two; each pools 2x2 after its first
  conv. BatchNorm has frozen weights and momentum 0.5, activations are GELU, a plain full-
  map max pool (not adaptive pooling, whose backward uses slow atomics) feeds a linear head
  whose logits are scaled by 1.25/9, and the model runs in fp16 and channels-last.
- **Training**: 8.25 epochs, batch 1024, Nesterov SGD (lr 11.5, wd 0.017 per 1024 examples,
  momentum 0.85, 32x lr on BatchNorm biases). Label smoothing 0.25. 23% warmup, then linear
  decay to 0.07x. Lookahead EMA every 5 steps. The first quarter of the steps trains on
  24x24 crops (a bilinear downscale of the training set made in `prepare`), the rest on
  32x32.
- **Augmentation**: alternating flip, 2-pixel reflect-padded translation, and per-image
  brightness/contrast jitter of strength 0.3 drawn from a generator seeded by the trial seed.
- **Untimed `build`**: `torch.compile(mode="max-autotune")` for the 32x32 graph, the label-smoothed loss compiled
  separately, a second static graph in the default mode for the 24x24 steps, plus a warmup of the training and
  evaluation kernels on random synthetic images at both sizes. `prepare` resets every
  parameter, BatchNorm statistic, optimizer, EMA buffer and the jitter generator before each
  trial.

Every setting can be overridden for experiments with `--params`; the defaults are the
submitted recipe.

## Development results

### Current defaults (stacked on the accuracy-recovery recipe)

Paired 40-trial runs with random seed starts, same Hugging Face Jobs container
(`a100-large`, A100-SXM4-80GB), current defaults against the accuracy-recovery recipe below
(`scripts/hf_ab.sh`). The HF card is faster than Modal's, so compare rows within a job only.

| Seed start | This recipe | Accuracy-recovery recipe (8.75 ep) |
| --- | ---: | ---: |
| 466919630 | 75.24% in 5.109 s | 75.27% in 5.690 s |
| 350181315 | 75.32% in 5.126 s | 75.31% in 5.712 s |

About 0.58 s (10%) faster at equal accuracy. Screened in paired 16-trial runs against the
accuracy-recovery recipe (75.27-75.41% in 5.59-5.67 s in those jobs):

| Change on top of the accuracy-recovery recipe | Accuracy | Time |
| --- | ---: | ---: |
| 2-conv 128-wide first group, lr 11.5, wd 0.017, BN momentum 0.5, 8.25 ep | 75.19-75.23% | 5.22-5.30 s |
| + full-map `max_pool2d` instead of `AdaptiveMaxPool2d` | 75.20% | 5.07 s |
| + compiled label-smoothed loss (**selected**) | 75.25% | 4.99 s |
| + SiLU instead of GELU | 74.81% | 4.93 s |
| at 8.0 epochs | 74.99-75.06% | 4.93-5.07 s |
| first quarter at 24x24 extended to 35% | 75.02% | 5.02 s |
| wd 0.02 / lr 12.5 / BN momentum 0.6 / label smoothing 0.3 | 75.05-75.24% | 5.02-5.08 s |
| lr 11.5, wd 0.017, BN 0.5 only (8.75 ep, original widths) | 75.48% | 5.63 s |

### Accuracy-recovery recipe (previous defaults)

40 trials (seeds 0-39) with the official accuracy target and a cold `build`, on Modal
NVIDIA A100-SXM4-80GB cards (judging hardware), one run per power limit:

| GPU | Mean accuracy | Accuracy std | Mean prepare + train | Time std |
| --- | ---: | ---: | ---: | ---: |
| A100-SXM4-80GB, 400 W | 75.34% | 0.28 pp | 5.76 s | 0.17 s |

The cold `build` took 187 s. No trial had a non-finite loss. Modal handed out no 500 W
card in 11 attempts on 3 October evening; on this network the 500 W cards had run about 7%
faster than the 400 W ones.

**Changes from the previous recipe** (96/256/768 at 9.5 epochs: 75.25% in 6.86 s): 8.75
epochs with the first quarter at 24x24, lr 10.8, label smoothing 0.25, BatchNorm bias lr 32x,
BatchNorm momentum 0.7, logit scale 1.25/9 and brightness/contrast jitter 0.3; measured gain
1.10 s (16%) at equal accuracy on the same card type.

**How the defaults were chosen.** Every knob was screened in paired A/B runs (same seeds,
same container, 8 trials) against the 8.5-epoch version of the previous recipe on 400 W
cards, scoring each change by the time it saves plus the time its accuracy change is worth
(1 pp was worth about 1 s of training near 8.5 epochs). The single accuracy knobs were
within noise on their own but added up: the six-knob stack gave +0.24 to +0.32 pp at equal
time. Training the first quarter of the steps at 24x24 saved 0.5 s for 0.35 pp alone and
paid for itself inside the stack. The stack with 24x24 at 9.0 epochs reached 75.40% in
5.87 s and at 8.75 epochs 75.34% in 5.76 s (both 40 trials, cold build); 8.75 is the
default and `--params '{"epochs": 9.0}'` is the safer alternative. 28x28 for half the steps
(75.34% in 5.92 s at 9.0 epochs) and cutout, momentum 0.8 and longer low-resolution phases
were rejected.
