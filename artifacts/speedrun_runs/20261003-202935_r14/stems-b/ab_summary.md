# r14/stems-b

2026-10-03 20:37, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41KWY9RE34YXTGBA0WN7CXR, CLOUD_PROVIDER_AWS/us-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 434.0 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41KWJJ52KQ2QX0E5YXY9XQR CLOUD_PROVIDER_AZURE/eu-south]', 'NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41KWRVKC1T70K51VVHT3PBR CLOUD_PROVIDER_AZURE/eu-south]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41KWY9RE34YXTGBA0WN7CXR CLOUD_PROVIDER_AWS/us-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stem-whiten4s2-w96 | `{"stem": "whiten4s2", "widths": [96, 256, 768]}` | 8/8 | 73.91 | 0.22 | -1.61 +- 0.15 | 5.16 | +0.09 | +2.40 | 70.83 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.53 | 0.22 | control | 5.07 | control | control | 25.69 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| stem-whiten3s2 | `{"stem": "whiten3s2"}` | 8/8 | 74.27 | 0.35 | -1.26 +- 0.16 | 4.76 | -0.32 | +1.49 | 77.81 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| stem-whiten3s2-e10.25 | `{"epochs": 10.25, "stem": "whiten3s2"}` | 8/8 | 74.62 | 0.34 | -0.91 +- 0.15 | 5.15 | +0.08 | +1.39 | 25.46 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
