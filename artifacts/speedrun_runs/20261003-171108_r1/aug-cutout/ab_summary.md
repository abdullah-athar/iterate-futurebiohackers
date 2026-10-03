# r1/aug-cutout

2026-10-03 17:21, NVIDIA A100-SXM4-80GB @ 500.00 W (task ta-01M418QHJ47BHGV28FWT2NFZBR, CLOUD_PROVIDER_UNSPECIFIED/eu), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.445, container wall 425.8 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M418QHJ47BHGV28FWT2NFZBR CLOUD_PROVIDER_UNSPECIFIED/eu]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 75.13 | 0.22 | control | 6.95 | control | control | 11.62 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 500.00 W | QUALIFIED (>= 75%) |
| cutout4 | `{"cutout": 4}` | 8/8 | 75.20 | 0.16 | +0.08 +- 0.10 | 6.96 | +0.00 | -0.18 | 10.34 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 500.00 W | QUALIFIED (>= 75%) |
| cutout8 | `{"cutout": 8}` | 8/8 | 74.70 | 0.20 | -0.42 +- 0.11 | 6.96 | +0.01 | +0.96 | 10.28 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 500.00 W | BELOW 75% |
