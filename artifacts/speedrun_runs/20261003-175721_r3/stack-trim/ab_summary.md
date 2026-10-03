# r3/stack-trim

2026-10-03 18:05, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41B8D36PN9V9N7MRWHNJ7WR, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.445, container wall 388.3 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41B8D36PN9V9N7MRWHNJ7WR CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 74.99 | 0.23 | control | 6.06 | control | control | 18.18 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| stack6-e7.5 | `{"bias_scaler": 32.0, "bn_momentum": 0.7, "epochs": 7.5, "jitter": 0.3, "label_smoothing": 0.25, "lr": 10.8, "momentum": 0.8}` | 8/8 | 74.61 | 0.28 | -0.38 +- 0.17 | 5.40 | -0.66 | +0.20 | 32.47 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| stack6-res24-s0.5 | `{"bias_scaler": 32.0, "bn_momentum": 0.7, "jitter": 0.3, "label_smoothing": 0.25, "lr": 10.8, "momentum": 0.8, "resolution_switch": 0.5, "train_resolution": 24}` | 8/8 | 74.56 | 0.29 | -0.43 +- 0.12 | 5.00 | -1.06 | -0.09 | 57.81 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| stack6-res28-s0.5 | `{"bias_scaler": 32.0, "bn_momentum": 0.7, "jitter": 0.3, "label_smoothing": 0.25, "lr": 10.8, "momentum": 0.8, "resolution_switch": 0.5, "train_resolution": 28}` | 8/8 | 75.03 | 0.20 | +0.04 +- 0.11 | 5.66 | -0.41 | -0.50 | 50.74 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
