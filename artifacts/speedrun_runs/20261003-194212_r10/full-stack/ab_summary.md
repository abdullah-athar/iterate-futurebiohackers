# r10/full-stack

2026-10-03 19:50, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41H7PG8WXS4A01A9Z2P0D5R, CLOUD_PROVIDER_GCP/us-central), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 411.2 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M41H5SYD4YM54EKJBE4ZKT3R CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41H7KRQZGZZDGZ8ZAER4M1R CLOUD_PROVIDER_UNSPECIFIED/us-east]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41H7PG8WXS4A01A9Z2P0D5R CLOUD_PROVIDER_GCP/us-central]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S3-e9.0 | `{"bias_scaler": 16.0, "compile_loss": true, "epochs": 9.0, "fused_sgd": true, "lr": 12.0, "weight_decay": 0.0168, "widths": [64, 256, 768]}` | 8/8 | 75.06 | 0.27 | +0.22 +- 0.12 | 4.93 | +0.01 | -0.31 | 40.02 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 74.83 | 0.13 | control | 4.92 | control | control | 31.57 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| S3-e9.25 | `{"bias_scaler": 16.0, "compile_loss": true, "epochs": 9.25, "fused_sgd": true, "lr": 12.0, "weight_decay": 0.0168, "widths": [64, 256, 768]}` | 8/8 | 75.24 | 0.24 | +0.41 +- 0.07 | 5.11 | +0.19 | -0.40 | 40.79 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| S3-e9.5 | `{"bias_scaler": 16.0, "compile_loss": true, "epochs": 9.5, "fused_sgd": true, "lr": 12.0, "weight_decay": 0.0168, "widths": [64, 256, 768]}` | 8/8 | 75.53 | 0.22 | +0.70 +- 0.09 | 5.16 | +0.24 | -0.75 | 30.98 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
