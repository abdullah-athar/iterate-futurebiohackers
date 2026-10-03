# r7/staged-b-1

2026-10-03 19:29, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41G3V3BT1YXZYFNBMD2K8PR, CLOUD_PROVIDER_UNSPECIFIED/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 355.5 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41G1Y2EP7KFH0AP6XHAAGRR CLOUD_PROVIDER_UNSPECIFIED/us-east]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41G3V3BT1YXZYFNBMD2K8PR CLOUD_PROVIDER_UNSPECIFIED/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s20x35 | `{"res_schedule": [[20, 0.35]]}` | 8/8 | 74.82 | 0.14 | -0.47 +- 0.10 | 5.32 | -0.36 | +0.31 | 74.27 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.30 | 0.27 | control | 5.69 | control | control | 31.79 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| s20x25-24x50 | `{"res_schedule": [[20, 0.25], [24, 0.5]]}` | 8/8 | 74.83 | 0.25 | -0.46 +- 0.11 | 4.97 | -0.71 | -0.05 | 44.11 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
