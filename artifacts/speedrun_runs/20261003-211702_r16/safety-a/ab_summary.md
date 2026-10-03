# r16/safety-a

2026-10-03 21:28, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41PQQ694PRRMQHZZXH9ENZR, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 567.5 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41PQQ694PRRMQHZZXH9ENZR CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| safe-stack-e9.0 | `{"bias_scaler": 16.0, "bn_momentum": 0.7, "color_jitter": [0.3, 0.3]}` | 8/8 | 75.08 | 0.36 | +0.07 +- 0.12 | 4.72 | -0.01 | -0.08 | 168.14 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.01 | 0.13 | control | 4.73 | control | control | 81.60 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| safe-stack-e9.25 | `{"bias_scaler": 16.0, "bn_momentum": 0.7, "color_jitter": [0.3, 0.3], "epochs": 9.25}` | 8/8 | 74.93 | 0.22 | -0.08 +- 0.09 | 4.87 | +0.14 | +0.22 | 32.61 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| safe-stack-e9.5 | `{"bias_scaler": 16.0, "bn_momentum": 0.7, "color_jitter": [0.3, 0.3], "epochs": 9.5}` | 8/8 | 75.27 | 0.24 | +0.26 +- 0.09 | 5.01 | +0.28 | +0.02 | 32.66 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
