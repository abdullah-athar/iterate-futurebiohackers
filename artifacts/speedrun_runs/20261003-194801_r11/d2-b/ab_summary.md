# r11/d2-b

2026-10-03 19:57, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41HGM8K5MFY7YFMCP4WTP3R, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 542.6 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41HGEA6P0MZC0P7VEE3NW1R CLOUD_PROVIDER_AZURE/eu-west]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41HGM8K5MFY7YFMCP4WTP3R CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| d223 | `{"depths": [2, 2, 3]}` | 8/8 | 74.74 | 0.23 | -0.55 +- 0.13 | 4.89 | -0.88 | -0.08 | 87.66 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.29 | 0.24 | control | 5.77 | control | control | 35.00 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| d232 | `{"depths": [2, 3, 2]}` | 8/8 | 73.71 | 0.23 | -1.58 +- 0.15 | 4.77 | -1.00 | +1.25 | 82.13 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| d322 | `{"depths": [3, 2, 2]}` | 8/8 | 73.86 | 0.29 | -1.43 +- 0.17 | 4.76 | -1.01 | +1.03 | 78.63 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
