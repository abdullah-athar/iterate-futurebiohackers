# r18/h24-a

2026-10-03 21:52, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41RD40MSS98A0BGDYPB06SR, CLOUD_PROVIDER_UNSPECIFIED/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 257.6 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41RD40MSS98A0BGDYPB06SR CLOUD_PROVIDER_UNSPECIFIED/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r24h-e9.75 | `{"epochs": 9.75, "resolution_schedule": [[24, 0.5]]}` | 8/8 | 75.00 | 0.36 | -0.04 +- 0.13 | 4.28 | -0.43 | -0.39 | 25.99 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.04 | 0.24 | control | 4.70 | control | control | 24.02 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| r24h-e10.0 | `{"epochs": 10.0, "resolution_schedule": [[24, 0.5]]}` | 8/8 | 75.10 | 0.30 | +0.06 +- 0.10 | 4.39 | -0.31 | -0.37 | 25.10 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
