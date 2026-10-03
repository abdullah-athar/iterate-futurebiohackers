# r1/opt-warmup-final

2026-10-03 17:32, NVIDIA A100-SXM4-80GB @ 500.00 W (task ta-01M41975JD45A67NSJG3PMDEGR, CLOUD_PROVIDER_UNSPECIFIED/eu), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.445, container wall 561.4 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M41975JD45A67NSJG3PMDEGR CLOUD_PROVIDER_UNSPECIFIED/eu]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 75.13 | 0.22 | control | 7.00 | control | control | 11.06 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 500.00 W | QUALIFIED (>= 75%) |
| warmup0.3 | `{"warmup": 0.3}` | 8/8 | 75.05 | 0.36 | -0.08 +- 0.18 | 6.98 | -0.01 | +0.17 | 10.14 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 500.00 W | QUALIFIED (>= 75%) |
| final0.03 | `{"final_lr": 0.03}` | 8/8 | 75.07 | 0.34 | -0.05 +- 0.09 | 6.99 | -0.01 | +0.11 | 10.24 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 500.00 W | QUALIFIED (>= 75%) |
| final0.15 | `{"final_lr": 0.15}` | 8/8 | 75.13 | 0.14 | +0.00 +- 0.12 | 6.99 | -0.01 | -0.01 | 10.14 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 500.00 W | QUALIFIED (>= 75%) |
