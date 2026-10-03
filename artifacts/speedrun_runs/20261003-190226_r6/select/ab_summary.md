# r6/select

2026-10-03 19:18, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41EY94MSMZ4N47TJYX9HBHR, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 886.5 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M41EX85YP53E3WDP4QGHTMQR CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41EY94MSMZ4N47TJYX9HBHR CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hard0.75-offline | `{"hard_fraction": 0.75}` | 8/8 | 73.75 | 0.17 | -1.55 +- 0.11 | 5.09 | -0.56 | +0.98 | 405.06 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.29 | 0.24 | control | 5.65 | control | control | 24.83 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| hard0.75-online | `{"hard_fraction": 0.75, "proxy_mode": "online"}` | 8/8 | 73.91 | 0.24 | -1.38 +- 0.11 | 5.00 | -0.65 | +0.74 | 61.53 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| hard0.5-offline | `{"hard_fraction": 0.5}` | 8/8 | 67.81 | 0.53 | -7.49 +- 0.21 | 4.24 | -1.41 | +6.07 | 175.87 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
