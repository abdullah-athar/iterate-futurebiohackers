# r7/filter

2026-10-03 19:28, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41G1YJXEE3FBBD94V5T9HXR, CLOUD_PROVIDER_UNSPECIFIED/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 326.3 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41G1YJXEE3FBBD94V5T9HXR CLOUD_PROVIDER_UNSPECIFIED/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| filt6-0.75 | `{"filter_keep": 0.75, "filter_start": 6}` | 8/8 | 74.91 | 0.29 | -0.47 +- 0.10 | 5.21 | -0.46 | +0.20 | 24.59 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.38 | 0.23 | control | 5.68 | control | control | 23.79 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| filt5-0.75 | `{"filter_keep": 0.75, "filter_start": 5}` | 8/8 | 74.66 | 0.18 | -0.72 +- 0.10 | 5.09 | -0.59 | +0.44 | 23.27 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| filt6-0.5 | `{"filter_keep": 0.5, "filter_start": 6}` | 8/8 | 74.12 | 0.16 | -1.26 +- 0.07 | 4.82 | -0.86 | +0.93 | 23.30 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
