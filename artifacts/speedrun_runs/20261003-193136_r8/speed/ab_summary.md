# r8/speed

2026-10-03 19:41, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41GJFM8J92Q4Q8V0D4NMSYR, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 544.0 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41GJFM8J92Q4Q8V0D4NMSYR CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compile-loss-fused | `{"compile_loss": true, "fused_sgd": true, "widths": [64, 256, 768]}` | 8/8 | 74.99 | 0.20 | +0.02 +- 0.09 | 4.71 | -0.15 | -0.17 | 130.64 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 74.97 | 0.24 | control | 4.86 | control | control | 23.31 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| gelu-tanh | `{"gelu_approximate": "tanh", "widths": [64, 256, 768]}` | 8/8 | 74.79 | 0.27 | -0.18 +- 0.09 | 5.11 | +0.25 | +0.51 | 92.62 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| triton-crop | `{"crop_mode": "triton", "widths": [64, 256, 768]}` | 8/8 | 74.90 | 0.32 | -0.07 +- 0.07 | 4.85 | -0.01 | +0.09 | 56.97 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
