# r16/k4

2026-10-03 21:28, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41PQZ4DFX0VK2R68RR41WYR, CLOUD_PROVIDER_AZURE/eu-south), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 536.9 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41PN24DQB7CHZ09SQ8C75NR CLOUD_PROVIDER_AZURE/eu-west]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41PQZ4DFX0VK2R68RR41WYR CLOUD_PROVIDER_AZURE/eu-south]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e8.5 | `{"epochs": 8.5}` | 8/8 | 74.92 | 0.26 | -0.20 +- 0.14 | 4.36 | -0.24 | -0.05 | 179.13 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.12 | 0.23 | control | 4.61 | control | control | 35.04 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| e9.5 | `{"epochs": 9.5}` | 8/8 | 75.40 | 0.14 | +0.28 +- 0.11 | 4.86 | +0.25 | -0.03 | 35.77 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| e10.0 | `{"epochs": 10.0}` | 8/8 | 75.65 | 0.17 | +0.53 +- 0.12 | 5.11 | +0.51 | -0.03 | 35.38 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
