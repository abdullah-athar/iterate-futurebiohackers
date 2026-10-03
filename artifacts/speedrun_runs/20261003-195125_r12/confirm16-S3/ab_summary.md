# r12/confirm16-S3

2026-10-03 19:57, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41HPS9BV0S4KTE5F56BM2YR, CLOUD_PROVIDER_AWS/us-west), one container, sequential, 16 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 380.5 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41HPS9BV0S4KTE5F56BM2YR CLOUD_PROVIDER_AWS/us-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S3-e9.25 | `{"bias_scaler": 16.0, "compile_loss": true, "epochs": 9.25, "fused_sgd": true, "lr": 12.0, "weight_decay": 0.0168, "widths": [64, 256, 768]}` | 16/16 | 75.12 | 0.24 | -0.12 +- 0.08 | 4.92 | -0.66 | -0.49 | 24.38 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 16/16 | 75.24 | 0.23 | control | 5.58 | control | control | 23.31 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| S3-e9.5 | `{"bias_scaler": 16.0, "compile_loss": true, "epochs": 9.5, "fused_sgd": true, "lr": 12.0, "weight_decay": 0.0168, "widths": [64, 256, 768]}` | 16/16 | 75.37 | 0.29 | +0.13 +- 0.08 | 5.04 | -0.54 | -0.72 | 23.56 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
