# r3/stack

2026-10-03 18:04, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41B84XW6GXSHX79E137DDBR, CLOUD_PROVIDER_AWS/us-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.445, container wall 333.5 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41B84XW6GXSHX79E137DDBR CLOUD_PROVIDER_AWS/us-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 74.98 | 0.22 | control | 6.00 | control | control | 18.15 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| stack4 | `{"bias_scaler": 32.0, "bn_momentum": 0.7, "label_smoothing": 0.25, "lr": 10.8}` | 8/8 | 75.22 | 0.25 | +0.24 +- 0.07 | 6.02 | +0.01 | -0.53 | 35.61 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| stack6 | `{"bias_scaler": 32.0, "bn_momentum": 0.7, "jitter": 0.3, "label_smoothing": 0.25, "lr": 10.8, "momentum": 0.8}` | 8/8 | 75.30 | 0.21 | +0.32 +- 0.13 | 6.12 | +0.11 | -0.61 | 17.08 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| stack6-e8 | `{"bias_scaler": 32.0, "bn_momentum": 0.7, "epochs": 8.0, "jitter": 0.3, "label_smoothing": 0.25, "lr": 10.8, "momentum": 0.8}` | 8/8 | 75.00 | 0.27 | +0.02 +- 0.13 | 5.68 | -0.33 | -0.37 | 16.94 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
