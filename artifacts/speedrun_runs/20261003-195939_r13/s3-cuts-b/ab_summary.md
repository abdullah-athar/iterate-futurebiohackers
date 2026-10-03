# r13/s3-cuts-b

2026-10-03 20:08, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41J5YSKQ8CW8RF3H0KDXQER, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 534.4 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41J5YSKQ8CW8RF3H0KDXQER CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S3-poolfirst-g3 | `{"bias_scaler": 16.0, "compile_loss": true, "epochs": 9.5, "fused_sgd": true, "lr": 12.0, "pool_first": [false, false, true], "weight_decay": 0.0168, "widths": [64, 256, 768]}` | 8/8 | 73.50 | 0.41 | -2.02 +- 0.12 | 4.52 | -0.61 | +2.28 | 88.62 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.53 | 0.22 | control | 5.13 | control | control | 29.32 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| S3-d223-e10.5 | `{"bias_scaler": 16.0, "compile_loss": true, "depths": [2, 2, 3], "epochs": 10.5, "fused_sgd": true, "lr": 12.0, "weight_decay": 0.0168, "widths": [64, 256, 768]}` | 8/8 | 74.83 | 0.28 | -0.71 +- 0.15 | 4.89 | -0.24 | +0.76 | 74.56 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| S3-w64-192-768-e10.0 | `{"bias_scaler": 16.0, "compile_loss": true, "epochs": 10.0, "fused_sgd": true, "lr": 12.0, "weight_decay": 0.0168, "widths": [64, 192, 768]}` | 8/8 | 74.99 | 0.37 | -0.54 +- 0.15 | 4.86 | -0.28 | +0.49 | 101.81 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
