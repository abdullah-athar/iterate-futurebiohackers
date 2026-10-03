# r11/d2-c

2026-10-03 20:01, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41HQ444P3HXTT74NB2FSAYR, CLOUD_PROVIDER_AZURE/eu-south), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 578.4 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41HPXT0HR1732N6JTFNXD9R CLOUD_PROVIDER_UNSPECIFIED/us-east]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41HQ444P3HXTT74NB2FSAYR CLOUD_PROVIDER_AZURE/eu-south]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| d222-skip | `{"depth2_residual": true, "depths": [2, 2, 2]}` | 8/8 | 73.65 | 0.25 | -1.79 +- 0.15 | 4.21 | -1.45 | +1.11 | 90.01 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.43 | 0.30 | control | 5.66 | control | control | 35.11 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| d222-skip-w128-320-896 | `{"depth2_residual": true, "depths": [2, 2, 2], "widths": [128, 320, 896]}` | 8/8 | 74.83 | 0.31 | -0.61 +- 0.17 | 5.75 | +0.09 | +0.96 | 166.17 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| d222-skip-e9.5 | `{"depth2_residual": true, "depths": [2, 2, 2], "epochs": 9.5}` | 8/8 | 74.12 | 0.27 | -1.32 +- 0.15 | 4.56 | -1.09 | +0.79 | 27.60 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
