# r3/acc-b

2026-10-03 18:03, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41B5T21QTYWAVC34QFM4MFR, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.445, container wall 381.2 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41B5T21QTYWAVC34QFM4MFR CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 74.98 | 0.22 | control | 6.09 | control | control | 23.26 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| bn0.7 | `{"bn_momentum": 0.7}` | 8/8 | 75.01 | 0.30 | +0.03 +- 0.14 | 6.08 | -0.01 | -0.08 | 45.09 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| jitter0.3 | `{"jitter": 0.3}` | 8/8 | 75.03 | 0.27 | +0.05 +- 0.13 | 6.11 | +0.02 | -0.09 | 21.03 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| mom0.8 | `{"momentum": 0.8}` | 8/8 | 74.83 | 0.33 | -0.15 +- 0.11 | 6.10 | +0.01 | +0.35 | 20.41 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
