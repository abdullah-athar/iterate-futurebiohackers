# r27/b1-a

2026-10-04 01:13, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M423AAGRSJ9F1W5YA60EP72R, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 858.1 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M423AAGRSJ9F1W5YA60EP72R CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| se-g3 | `{"se_groups": [2]}` | 8/8 | 75.05 | 0.22 | +0.04 +- 0.07 | 4.47 | +0.04 | -0.00 | 200.19 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.00 | 0.13 | control | 4.43 | control | control | 35.01 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| se-g23 | `{"se_groups": [1, 2]}` | 8/8 | 75.19 | 0.37 | +0.19 +- 0.13 | 4.62 | +0.19 | -0.00 | 198.09 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| multiscale | `{"multiscale_head": true}` | 8/8 | 74.70 | 0.17 | -0.31 +- 0.07 | 4.49 | +0.06 | +0.36 | 185.86 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
