# r39/kaudit-a

2026-10-04 02:50, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M429A28WHWXHXTJ8R23JR1QR, CLOUD_PROVIDER_AZURE/uk), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 408.8 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M429A28WHWXHXTJ8R23JR1QR CLOUD_PROVIDER_AZURE/uk]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contig-batches | `{"batch_order": "contiguous", "g2_pair": "inner192", "global_pool": "fullpool_avgsum"}` | 8/8 | 75.06 | 0.35 | -0.08 +- 0.18 | 4.17 | -0.03 | +0.05 | 85.29 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.15 | 0.24 | control | 4.20 | control | control | 54.08 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| contig-batches-r24x0.35 | `{"batch_order": "contiguous", "g2_pair": "inner192", "global_pool": "fullpool_avgsum", "resolution_schedule": [[24, 0.35], [28, 0.5]]}` | 8/8 | 75.05 | 0.28 | -0.10 +- 0.14 | 4.06 | -0.14 | -0.04 | 52.42 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
