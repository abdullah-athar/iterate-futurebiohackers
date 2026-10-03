# r6/batch

2026-10-03 19:12, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41EX85YZS63XN0C3TPRCHPR, CLOUD_PROVIDER_UNSPECIFIED/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 591.3 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41EX85YZS63XN0C3TPRCHPR CLOUD_PROVIDER_UNSPECIFIED/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bs1536 | `{"batch_size": 1536}` | 8/8 | 74.86 | 0.30 | -0.43 +- 0.17 | 5.46 | -0.21 | +0.22 | 154.13 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.29 | 0.24 | control | 5.67 | control | control | 24.56 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| bs2048 | `{"batch_size": 2048}` | 8/8 | 74.32 | 0.25 | -0.97 +- 0.12 | 5.41 | -0.26 | +0.71 | 148.91 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| bs1536-e9.25 | `{"batch_size": 1536, "epochs": 9.25}` | 8/8 | 75.12 | 0.13 | -0.18 +- 0.11 | 5.77 | +0.10 | +0.27 | 23.68 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
