# r1/aug-translate

2026-10-03 17:21, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M418QHJJK8TP4H1M3T1DMSXR, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.445, container wall 383.6 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M418QHJJK8TP4H1M3T1DMSXR CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 75.22 | 0.21 | control | 7.40 | control | control | 17.57 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| translate1 | `{"translate": 1}` | 8/8 | 75.26 | 0.18 | +0.04 +- 0.13 | 7.37 | -0.03 | -0.12 | 15.93 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| translate3 | `{"translate": 3}` | 8/8 | 75.22 | 0.23 | +0.01 +- 0.12 | 7.43 | +0.03 | +0.02 | 15.87 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
