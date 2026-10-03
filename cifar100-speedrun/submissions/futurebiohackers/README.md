# futurebiohackers: airbench-style CIFAR-100 recipe

An adaptation of Keller Jordan's [airbench](https://github.com/KellerJordan/cifar10-airbench)
(MIT, Copyright (c) 2024 Keller Jordan; the original permission notice is in
`LICENSE.airbench`) to CIFAR-100, trained from scratch with no test-time augmentation.

## Recipe

- **Network**: frozen 2x2 patch-whitening conv (initialized from 5,000 training images in
  `prepare`), then three conv groups of widths 64/256/768. Each group has three 3x3 convs
  with 2x2 max-pooling after the first and a residual over the last two. BatchNorm has frozen
  weights and momentum 0.7, activations are GELU, global max pooling feeds a linear head whose
  logits are scaled by 1.25/9, and the model runs in fp16 and channels-last.
- **Training**: 9.5 epochs, batch 1024, Nesterov SGD (lr 12.0, wd 0.0168 per 1024 examples,
  momentum 0.85, 16x lr on BatchNorm biases) with PyTorch's fused SGD kernel. Label smoothing
  0.25, computed by a compiled loss function. 23% warmup, then linear decay to 0.07x.
  Lookahead EMA every 5 steps. The first quarter of the steps trains on 24x24 crops (a
  bilinear downscale of the training set made in `prepare`), the rest on 32x32.
- **Augmentation**: alternating flip, 2-pixel reflect-padded translation, and per-image
  brightness/contrast jitter of strength 0.3 drawn from a generator seeded by the trial seed.
- **Untimed `build`**: `torch.compile(mode="max-autotune")` for the 32x32 graph, a second
  static graph in the default mode for the 24x24 steps, the compiled loss, plus a warmup of
  the training and evaluation kernels on random synthetic images at both sizes. `prepare`
  resets every parameter, BatchNorm statistic, optimizer, EMA buffer and the jitter
  generator before each trial.

Every setting can be overridden for experiments with `--params`; the defaults are the
submitted recipe.

## Development results

40 trials per seed set with the official accuracy target and a cold `build`, on Modal
NVIDIA A100-SXM4-80GB cards at the 400 W power limit (judging hardware):

| GPU | Seeds | Mean accuracy | Accuracy std | Mean prepare + train | Time std |
| --- | --- | ---: | ---: | ---: | ---: |
| A100-SXM4-80GB, 400 W | 0-39 | 75.33% | 0.26 pp | 5.09 s | 0.09 s |
| A100-SXM4-80GB, 400 W | 40-79 | 75.28% | 0.26 pp | 5.14 s | 0.07 s |

The cold `build` took 170 s and 197 s. No trial had a non-finite loss. Modal handed out no
500 W card during the measurements; on this family of recipes the 500 W cards had run about 7%
faster than the 400 W ones.

**Changes from the previous recipe** (96/256/768 at 8.75 epochs, bias lr 32x, lr 10.8, wd
0.012: 75.34% in 5.76 s on 400 W): first group narrowed to 64 channels, BatchNorm bias lr 16x,
lr 12.0, weight decay 0.0168, the loss compiled and the optimizer step fused, 9.5 epochs;
measured gain 0.65 s (11%) at equal accuracy on the same card type.

**How the defaults were chosen.** Every change was screened in paired A/B runs (same seeds,
same container, 8 trials) on 400 W cards, scoring each by the time it saves plus the time
its accuracy change is worth (about 1 pp per second on the narrow network). Narrowing the
first group to 64 channels saved 0.7 s for 0.4 pp; on that narrower network the optimizer
knobs that had been flat before became gainers (bias lr 16x +0.14 pp, lr 12 +0.16, weight
decay x1.4 +0.19, together +0.6 pp at equal time), the compiled loss and fused SGD saved
0.15 s for free, and 0.75 extra epochs (cheap here: 1.1 pp per second) restored the margin.
The stack was confirmed at 16 paired trials (75.37% / 5.04 s against the previous recipe's
75.24% / 5.58 s), then at 40 trials on seeds 0-39 and on fresh seeds 40-79 (table above).
Rejected on this network: depth-2 groups (even widened or with a skip connection), 20 px or
longer low-resolution phases, loss-based data filtering, proxy hard-example selection,
batches of 1536/2048, cutout, momentum 0.8/0.9, lookahead every 3/10 steps.
