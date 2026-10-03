# r8/knobs-b

2026-10-03 19:39, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41GJFS253VFCE1N752FR80R, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 488.4 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41GJFS253VFCE1N752FR80R CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wd0.0168 | `{"weight_decay": 0.0168, "widths": [64, 256, 768]}` | 8/8 | 75.09 | 0.21 | +0.19 +- 0.05 | 4.85 | +0.00 | -0.27 | 112.93 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 74.90 | 0.22 | control | 4.84 | control | control | 25.13 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| warmup0.3 | `{"warmup": 0.3, "widths": [64, 256, 768]}` | 8/8 | 74.85 | 0.34 | -0.04 +- 0.14 | 5.17 | +0.33 | +0.39 | 49.74 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| final0.15 | `{"final_lr": 0.15, "widths": [64, 256, 768]}` | 8/8 | 74.74 | 0.35 | -0.15 +- 0.13 | 5.37 | +0.53 | +0.75 | 57.70 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
