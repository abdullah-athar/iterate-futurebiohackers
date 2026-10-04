# r34/paired40-s40-pool

2026-10-04 02:21, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M427AYQQVAKAB8X88HST5MGR, CLOUD_PROVIDER_AZURE/uk), one container, sequential, 40 trial(s) per run, seeds from 40, warm compile cache, k = 1.0, container wall 694.1 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M427AYQQVAKAB8X88HST5MGR CLOUD_PROVIDER_AZURE/uk]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| avgsum-e10.0-s40 | `{"global_pool": "fullpool_avgsum"}` | 40/40 | 75.31 | 0.26 | +0.09 +- 0.06 | 4.23 | -0.13 | -0.21 | 189.44 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 40/40 | 75.22 | 0.25 | control | 4.36 | control | control | 52.98 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
