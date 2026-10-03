# r21/g3pair-b

2026-10-03 23:13, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41WTQRB85ZNDRAQM5V8Z46R, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 445.3 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M41WTFKF9V99504F8XPSWWNR CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41WTQRB85ZNDRAQM5V8Z46R CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g3-bottleneck384 | `{"g3_pair": "bottleneck384"}` | 8/8 | 73.77 | 0.30 | -1.44 +- 0.10 | 3.80 | -0.73 | +0.71 | 251.99 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.20 | 0.27 | control | 4.53 | control | control | 43.85 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
