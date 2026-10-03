# r11/d2-a

2026-10-03 19:59, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41HGMEP8PAW494CZPJ8XJRR, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 653.6 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41HGEA65B5NHXD7K388H43R CLOUD_PROVIDER_AZURE/us-east]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41HGMEP8PAW494CZPJ8XJRR CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| d222 | `{"depths": [2, 2, 2]}` | 8/8 | 73.64 | 0.17 | -1.65 +- 0.09 | 4.26 | -1.35 | +1.02 | 76.35 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.29 | 0.24 | control | 5.61 | control | control | 33.12 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| d222-w128-320-896 | `{"depths": [2, 2, 2], "widths": [128, 320, 896]}` | 8/8 | 74.60 | 0.26 | -0.69 +- 0.13 | 5.83 | +0.21 | +1.20 | 151.67 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| d222-w96-320-1024 | `{"depths": [2, 2, 2], "widths": [96, 320, 1024]}` | 8/8 | 74.82 | 0.19 | -0.47 +- 0.09 | 5.88 | +0.27 | +0.94 | 129.13 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
