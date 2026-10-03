# r14/k3

2026-10-03 20:35, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41KWRPYJJYR7CG31F2THP5R, CLOUD_PROVIDER_AWS/us-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 333.1 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41KWJJ5H3REB9VHXC9JBJTR CLOUD_PROVIDER_AZURE/eu-south]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41KWRPYJJYR7CG31F2THP5R CLOUD_PROVIDER_AWS/us-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k-e9.0 | `{"epochs": 9.0}` | 8/8 | 75.06 | 0.27 | -0.47 +- 0.13 | 4.81 | -0.26 | +0.41 | 25.16 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.53 | 0.22 | control | 5.08 | control | control | 23.12 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| k-e10.0 | `{"epochs": 10.0}` | 8/8 | 75.48 | 0.16 | -0.05 +- 0.07 | 5.32 | +0.25 | +0.32 | 23.26 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| k-e10.5 | `{"epochs": 10.5}` | 8/8 | 75.68 | 0.21 | +0.15 +- 0.09 | 5.61 | +0.53 | +0.32 | 23.29 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
