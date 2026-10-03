# futurebiohackers: airbench-style CIFAR-100 recipe

An adaptation of Keller Jordan's [airbench](https://github.com/KellerJordan/cifar10-airbench)
(MIT, Copyright (c) 2024 Keller Jordan; the original permission notice is in
`LICENSE.airbench`) to CIFAR-100, trained from scratch with no test-time augmentation.

## Recipe

- **Network**: frozen 2x2 patch-whitening conv (initialized from 5,000 training images in
  `prepare`), then three conv groups of widths 96/256/768. Each group has three 3x3 convs
  with 2x2 max-pooling after the first and a residual over the last two. BatchNorm has frozen
  weights and momentum 0.7, activations are GELU, global max pooling feeds a linear head whose
  logits are scaled by 1.25/9, and the model runs in fp16 and channels-last.
- **Training**: 8.75 epochs, batch 1024, Nesterov SGD (lr 10.8, wd 0.012 per 1024 examples,
  momentum 0.85, 32x lr on BatchNorm biases). Label smoothing 0.25. 23% warmup, then linear
  decay to 0.07x. Lookahead EMA every 5 steps. The first quarter of the steps trains on
  24x24 crops (a bilinear downscale of the training set made in `prepare`), the rest on
  32x32.
- **Augmentation**: alternating flip, 2-pixel reflect-padded translation, and per-image
  brightness/contrast jitter of strength 0.3 drawn from a generator seeded by the trial seed.
- **Untimed `build`**: `torch.compile(mode="max-autotune")` for the 32x32 graph, a second
  static graph in the default mode for the 24x24 steps, plus a warmup of the training and
  evaluation kernels on random synthetic images at both sizes. `prepare` resets every
  parameter, BatchNorm statistic, optimizer, EMA buffer and the jitter generator before each
  trial.

Every setting can be overridden for experiments with `--params`; the defaults are the
submitted recipe.

## Development results

40 trials (seeds 0-39) with the official accuracy target and a cold `build`, on Modal
NVIDIA A100-SXM4-80GB cards (judging hardware), one run per power limit:

| GPU | Mean accuracy | Accuracy std | Mean prepare + train | Time std |
| --- | ---: | ---: | ---: | ---: |
| A100-SXM4-80GB, 400 W | 75.34% | 0.28 pp | 5.76 s | 0.17 s |

The cold `build` took 187 s. No trial had a non-finite loss.

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
