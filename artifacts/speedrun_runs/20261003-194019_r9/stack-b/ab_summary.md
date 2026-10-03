# r9/stack-b

2026-10-03 19:47, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41H5RW5HRH19TB78R4XZKNR, CLOUD_PROVIDER_AWS/us-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 336.6 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41H5RW5HRH19TB78R4XZKNR CLOUD_PROVIDER_AWS/us-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S-triton-e8.75 | `{"bias_scaler": 16.0, "crop_mode": "triton", "lr": 12.0, "widths": [64, 256, 768]}` | 8/8 | 74.90 | 0.32 | +0.06 +- 0.13 | 4.86 | +0.04 | -0.05 | 27.40 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 74.83 | 0.13 | control | 4.82 | control | control | 24.73 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| S-triton-e9.0 | `{"bias_scaler": 16.0, "crop_mode": "triton", "epochs": 9.0, "lr": 12.0, "widths": [64, 256, 768]}` | 8/8 | 74.97 | 0.33 | +0.14 +- 0.14 | 4.89 | +0.07 | -0.13 | 25.16 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| S-fused-compiled-e9.0 | `{"bias_scaler": 16.0, "compile_loss": true, "epochs": 9.0, "fused_sgd": true, "lr": 12.0, "widths": [64, 256, 768]}` | 8/8 | 75.13 | 0.22 | +0.30 +- 0.10 | 4.79 | -0.03 | -0.46 | 24.87 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
