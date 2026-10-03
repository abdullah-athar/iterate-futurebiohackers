# r21/g3pair-a

2026-10-03 23:14, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41WTQ2GQATNT5Q5J6VHHKNR, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 524.5 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M41WTFKFCH2AYJAV8BP19GXR CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41WTQ2GQATNT5Q5J6VHHKNR CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g3-inner512 | `{"g3_pair": "inner512"}` | 8/8 | 74.86 | 0.24 | -0.35 +- 0.10 | 4.23 | -0.31 | +0.04 | 165.61 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.20 | 0.27 | control | 4.54 | control | control | 36.93 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| g3-inner640 | `{"g3_pair": "inner640"}` | 8/8 | 75.04 | 0.26 | -0.16 +- 0.16 | 4.43 | -0.11 | +0.06 | 138.51 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
