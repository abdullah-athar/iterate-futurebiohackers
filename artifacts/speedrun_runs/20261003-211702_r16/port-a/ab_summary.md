# r16/port-a

2026-10-03 21:28, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41PRP9B7PJB3YRNNTWS837R, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 531.4 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41PQQ6Q4ZBV8XZSV9M76K0R CLOUD_PROVIDER_UNSPECIFIED/us-east]', 'NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M41PR1AE33C06JPSHENCVZBR CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M41PR4SDR6HYJ2DBZ0WC5YYR CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M41PRA85Y8ZVB37A35AHXMSR CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M41PRDGSH9Y6SQR1355SKP4R CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M41PRGXKBM0NTB0ZHJHZFNAR CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41PRP9B7PJB3YRNNTWS837R CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bias16 | `{"bias_scaler": 16.0}` | 8/8 | 75.04 | 0.19 | +0.01 +- 0.10 | 4.71 | -0.01 | -0.02 | 173.27 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.04 | 0.20 | control | 4.72 | control | control | 35.24 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| lr13.8 | `{"lr": 13.8}` | 8/8 | 74.83 | 0.24 | -0.20 +- 0.09 | 4.72 | -0.00 | +0.20 | 34.06 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| lr15.3 | `{"lr": 15.3}` | 8/8 | 74.83 | 0.41 | -0.21 +- 0.20 | 4.71 | -0.01 | +0.20 | 34.00 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
