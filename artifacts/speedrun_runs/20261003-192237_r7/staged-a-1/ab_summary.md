# r7/staged-a-1

2026-10-03 19:29, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41G3V44EAKJ5N7WMVGAX5FR, CLOUD_PROVIDER_AWS/us-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 350.9 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41G1XYSAG56AD6SH8WNBY7R CLOUD_PROVIDER_UNSPECIFIED/us-east]', 'NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41G2551KQZQ2XV0J20FFY9R CLOUD_PROVIDER_UNSPECIFIED/us-east]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41G3V44EAKJ5N7WMVGAX5FR CLOUD_PROVIDER_AWS/us-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s20x25-28x50 | `{"res_schedule": [[20, 0.25], [28, 0.5]]}` | 8/8 | 74.84 | 0.20 | -0.45 +- 0.10 | 5.32 | -0.35 | +0.30 | 68.71 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.29 | 0.24 | control | 5.67 | control | control | 24.68 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| s20x20-24x40-28x60 | `{"res_schedule": [[20, 0.2], [24, 0.4], [28, 0.6]]}` | 8/8 | 74.81 | 0.20 | -0.48 +- 0.10 | 5.10 | -0.57 | +0.12 | 64.80 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
