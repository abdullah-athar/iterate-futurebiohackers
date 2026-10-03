# r1/opt-lr-mom

2026-10-03 17:22, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M418QHTXYMPDWJX2MXCSG9DR, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.445, container wall 472.6 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M418QHTXYMPDWJX2MXCSG9DR CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 75.13 | 0.22 | control | 7.31 | control | control | 18.44 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| lr7.2 | `{"lr": 7.2}` | 8/8 | 75.17 | 0.25 | +0.04 +- 0.12 | 7.33 | +0.02 | -0.08 | 15.79 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| lr10.8 | `{"lr": 10.8}` | 8/8 | 75.30 | 0.26 | +0.17 +- 0.14 | 7.33 | +0.01 | -0.37 | 16.11 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| mom0.8 | `{"momentum": 0.8}` | 8/8 | 75.24 | 0.23 | +0.11 +- 0.11 | 7.33 | +0.02 | -0.24 | 15.49 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
