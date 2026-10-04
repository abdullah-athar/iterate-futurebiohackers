# r29/pool-a

2026-10-04 01:30, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M424T44C54Q6D6C03C36Q9WR, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 327.6 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M424T44C54Q6D6C03C36Q9WR CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fullpool | `{"global_pool": "fullpool"}` | 8/8 | 75.06 | 0.19 | +0.02 +- 0.08 | 4.27 | -0.13 | -0.15 | 134.06 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.04 | 0.27 | control | 4.40 | control | control | 39.40 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
