# r7/staged-a-2

2026-10-03 19:27, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41G3V3RZQ04Z1PQEC71KWGR, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 247.9 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41G1YAEFWHPX330HN1E0VJR CLOUD_PROVIDER_UNSPECIFIED/us-east]', 'NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41G2592BGBQXHEMCS4Y6E9R CLOUD_PROVIDER_UNSPECIFIED/us-east]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41G3V3RZQ04Z1PQEC71KWGR CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s24x25-28x50 | `{"res_schedule": [[24, 0.25], [28, 0.5]]}` | 8/8 | 75.14 | 0.22 | -0.15 +- 0.09 | 5.54 | -0.24 | -0.02 | 56.59 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.29 | 0.24 | control | 5.78 | control | control | 35.33 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
