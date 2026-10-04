# r27/d-c

2026-10-04 01:09, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M423D39A11V4N3ASC584ZXQR, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 493.2 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M423CXVGQQXGY9K218C2CR1R CLOUD_PROVIDER_AZURE/eu-west]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M423D39A11V4N3ASC584ZXQR CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g3-inner384-e10.5 | `{"epochs": 10.5, "g3_pair": "inner384"}` | 8/8 | 75.10 | 0.22 | +0.03 +- 0.15 | 4.58 | +0.03 | +0.01 | 179.08 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.07 | 0.28 | control | 4.54 | control | control | 46.17 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| g3-inner384-e10.75 | `{"epochs": 10.75, "g3_pair": "inner384"}` | 8/8 | 75.31 | 0.19 | +0.24 +- 0.13 | 4.69 | +0.14 | -0.09 | 45.01 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
