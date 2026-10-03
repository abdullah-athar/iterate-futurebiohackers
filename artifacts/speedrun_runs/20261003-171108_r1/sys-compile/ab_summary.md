# r1/sys-compile

2026-10-03 17:39, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M419AJBN6P94MZ3PMVHBASZR, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.445, container wall 864.5 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M419AJBN6P94MZ3PMVHBASZR CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 75.13 | 0.22 | control | 7.34 | control | control | 27.15 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| no-cudagraphs | `{"compile": "max-autotune-no-cudagraphs"}` | 8/8 | 75.13 | 0.22 | +0.00 +- 0.00 | 7.39 | +0.05 | +0.05 | 44.13 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| silu-gather | `{"activation": "silu", "crop_gather": true}` | 8/8 | 74.95 | 0.25 | -0.18 +- 0.13 | 7.11 | -0.23 | +0.16 | 37.42 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
