# r14/d332

2026-10-03 20:35, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41KWK6TMC28JQHEZ5FYWZ7R, CLOUD_PROVIDER_UNSPECIFIED/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 340.2 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41KWK6TMC28JQHEZ5FYWZ7R CLOUD_PROVIDER_UNSPECIFIED/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| d332 | `{"depths": [3, 3, 2]}` | 8/8 | 74.16 | 0.34 | -1.37 +- 0.14 | 4.53 | -0.53 | +1.43 | 85.68 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.53 | 0.22 | control | 5.05 | control | control | 28.13 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| d332-e10.5 | `{"depths": [3, 3, 2], "epochs": 10.5}` | 8/8 | 74.55 | 0.22 | -0.98 +- 0.09 | 4.98 | -0.07 | +1.33 | 31.49 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
