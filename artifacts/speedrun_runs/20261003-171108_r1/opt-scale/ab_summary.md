# r1/opt-scale

2026-10-03 17:32, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41976CEQCTNVYTFH4KZ5XER, CLOUD_PROVIDER_AWS/us-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.445, container wall 565.0 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41976CEQCTNVYTFH4KZ5XER CLOUD_PROVIDER_AWS/us-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 75.13 | 0.22 | control | 7.34 | control | control | 17.00 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| scale0.0889 | `{"scaling_factor": 0.08888888888888889}` | 8/8 | 75.02 | 0.20 | -0.10 +- 0.07 | 7.32 | -0.02 | +0.21 | 35.55 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| scale0.1389 | `{"scaling_factor": 0.1388888888888889}` | 8/8 | 75.23 | 0.31 | +0.10 +- 0.16 | 7.30 | -0.04 | -0.26 | 34.79 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
