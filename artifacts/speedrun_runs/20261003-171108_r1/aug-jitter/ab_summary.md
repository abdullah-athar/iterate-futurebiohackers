# r1/aug-jitter

2026-10-03 17:22, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M418QHJB2P0QRRFYJSCB2CSR, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.445, container wall 471.4 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M418QHJB2P0QRRFYJSCB2CSR CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 75.26 | 0.23 | control | 7.34 | control | control | 18.66 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| jitter0.1 | `{"jitter": 0.1}` | 8/8 | 75.31 | 0.32 | +0.04 +- 0.12 | 7.32 | -0.02 | -0.11 | 16.04 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| jitter0.2 | `{"jitter": 0.2}` | 8/8 | 75.34 | 0.23 | +0.08 +- 0.12 | 7.34 | +0.00 | -0.17 | 16.50 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| jitter0.3 | `{"jitter": 0.3}` | 8/8 | 75.38 | 0.19 | +0.11 +- 0.14 | 7.33 | -0.01 | -0.26 | 16.77 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
