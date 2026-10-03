# r7/stack-arch

2026-10-03 19:35, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41G1XYSBE3YZHMBCAXF41CR, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 785.0 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41G1XYSBE3YZHMBCAXF41CR CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w64-d233 | `{"depths": [2, 3, 3], "widths": [64, 256, 768]}` | 8/8 | 74.56 | 0.29 | -0.73 +- 0.15 | 5.11 | -0.71 | +0.33 | 142.05 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.29 | 0.24 | control | 5.82 | control | control | 40.04 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| w64-d233-triton-e9.5 | `{"crop_mode": "triton", "depths": [2, 3, 3], "epochs": 9.5, "widths": [64, 256, 768]}` | 8/8 | 74.91 | 0.18 | -0.38 +- 0.11 | 5.30 | -0.52 | +0.03 | 88.53 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| w64-triton-s20x20-24x40-28x60-e9.5 | `{"crop_mode": "triton", "epochs": 9.5, "res_schedule": [[20, 0.2], [24, 0.4], [28, 0.6]], "widths": [64, 256, 768]}` | 8/8 | 74.74 | 0.22 | -0.55 +- 0.09 | 5.56 | -0.26 | +0.53 | 235.66 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
