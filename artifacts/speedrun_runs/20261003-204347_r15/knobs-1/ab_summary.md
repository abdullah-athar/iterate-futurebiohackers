# r15/knobs-1

2026-10-03 20:50, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41MPJ1HW3JFKMN9P9E5PZ8R, CLOUD_PROVIDER_AZURE/uk), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 410.6 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41MPJ1HW3JFKMN9P9E5PZ8R CLOUD_PROVIDER_AZURE/uk]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr13.2 | `{"lr": 13.2}` | 8/8 | 75.40 | 0.49 | -0.13 +- 0.12 | 5.01 | -0.01 | +0.18 | 36.80 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.53 | 0.22 | control | 5.02 | control | control | 37.94 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| wd0.0202 | `{"weight_decay": 0.0202}` | 8/8 | 75.27 | 0.22 | -0.26 +- 0.11 | 5.06 | +0.04 | +0.41 | 38.16 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| bias12 | `{"bias_scaler": 12.0}` | 8/8 | 75.42 | 0.11 | -0.10 +- 0.07 | 5.13 | +0.11 | +0.26 | 34.57 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
