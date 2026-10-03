# r16/port-c

2026-10-03 21:35, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41PQQ6FQZH20MY3PYJE4JMR, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 926.7 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41PQQ6FQZH20MY3PYJE4JMR CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| res24q-28h | `{"resolution_schedule": [[24, 0.25], [28, 0.5]]}` | 8/8 | 75.08 | 0.34 | -0.04 +- 0.12 | 4.36 | -0.38 | -0.34 | 260.29 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.12 | 0.19 | control | 4.74 | control | control | 58.18 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| bs512-lr16-e8 | `{"batch_size": 512, "epochs": 8.0, "lr": 16.0}` | 8/8 | 75.04 | 0.25 | -0.07 +- 0.13 | 4.61 | -0.13 | -0.06 | 181.65 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| bs768-lr14-e8.5 | `{"batch_size": 768, "epochs": 8.5, "lr": 14.0}` | 8/8 | 74.96 | 0.19 | -0.16 +- 0.07 | 4.63 | -0.10 | +0.05 | 174.17 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
