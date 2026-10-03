# futurebiohackers: airbench-style CIFAR-100 recipe

An adaptation of Keller Jordan's [airbench](https://github.com/KellerJordan/cifar10-airbench)
(MIT, Copyright (c) 2024 Keller Jordan) to CIFAR-100, trained from scratch with no test-time
augmentation.

## Recipe

- **Network**: frozen 2x2 patch-whitening conv (initialized from 5,000 training images in
  `prepare`), then three conv groups of widths 128/384/576. Each group has three 3x3 convs
  with max-pooling and a residual over the last two. BatchNorm has frozen weights and
  momentum 0.6, activations are GELU, the head is linear, and the model runs in fp16 and
  channels-last.
- **Training**: 8.5 epochs, batch 1024, Nesterov SGD (lr 9.0, wd 0.012 per 1024 examples,
  momentum 0.85, 64x lr on BatchNorm biases). Label smoothing 0.3. 23% warmup, then linear
  decay to 0.07x. Lookahead EMA every 5 steps.
- **Augmentation**: alternating flip and 2-pixel reflect-padded translation.
- **Untimed `build`**: `torch.compile(mode="max-autotune")` plus a warmup of the training and
  evaluation kernels on random synthetic images. `prepare` resets every parameter, BatchNorm
  statistic, optimizer and EMA buffer before each trial.

Every setting can be overridden for experiments with `--params`; the defaults are the
submitted recipe.

## Development results

40 trials (seeds 0-39) with the official accuracy target and a cold `build`, on Modal
NVIDIA A100-SXM4-80GB cards at the two power limits Modal hands out:

| GPU | Mean accuracy | Accuracy std | Mean prepare + train | Time std |
| --- | ---: | ---: | ---: | ---: |
| A100-SXM4-80GB, 400 W | 75.20% | 0.28 pp | 7.41 s | 0.01 s |
| A100-SXM4-80GB, 500 W | 75.19% | 0.25 pp | 6.94 s | 0.01 s |

The cold `build` (compile and warmup, untimed) took 133 s and 95 s respectively.
