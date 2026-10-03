# r13/s3-cuts-a

2026-10-03 20:08, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41J5YSKE2E3WTQNDHM89W7R, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 514.6 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41J5YSKE2E3WTQNDHM89W7R CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S3-d223 | `{"bias_scaler": 16.0, "compile_loss": true, "depths": [2, 2, 3], "epochs": 9.5, "fused_sgd": true, "lr": 12.0, "weight_decay": 0.0168, "widths": [64, 256, 768]}` | 8/8 | 74.57 | 0.22 | -0.92 +- 0.09 | 4.40 | -0.72 | +0.60 | 101.18 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.49 | 0.19 | control | 5.12 | control | control | 24.87 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| S3-w64-192-768 | `{"bias_scaler": 16.0, "compile_loss": true, "epochs": 9.5, "fused_sgd": true, "lr": 12.0, "weight_decay": 0.0168, "widths": [64, 192, 768]}` | 8/8 | 74.78 | 0.23 | -0.71 +- 0.12 | 4.58 | -0.54 | +0.48 | 92.33 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| S3-poolfirst-g2 | `{"bias_scaler": 16.0, "compile_loss": true, "epochs": 9.5, "fused_sgd": true, "lr": 12.0, "pool_first": [false, true, false], "weight_decay": 0.0168, "widths": [64, 256, 768]}` | 8/8 | 73.21 | 0.17 | -2.28 +- 0.07 | 4.79 | -0.33 | +2.92 | 73.10 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
