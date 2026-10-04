# r27/b1-b

2026-10-04 01:08, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M423AAGR5704QDWCRQS4246R, CLOUD_PROVIDER_UNSPECIFIED/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 530.8 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M423AAGR5704QDWCRQS4246R CLOUD_PROVIDER_UNSPECIFIED/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pool-maxmean-cat | `{"global_pool": "maxmean_cat"}` | 8/8 | 75.21 | 0.14 | +0.18 +- 0.09 | 4.52 | +0.00 | -0.17 | 175.97 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.04 | 0.27 | control | 4.52 | control | control | 33.91 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| pool-maxmean-sum | `{"global_pool": "maxmean_sum"}` | 8/8 | 75.33 | 0.25 | +0.30 +- 0.14 | 4.52 | -0.00 | -0.30 | 116.24 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
