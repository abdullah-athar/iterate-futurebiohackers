# r1/arch-widths-c

2026-10-03 17:25, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M418QHJBSP7EJXJW46MC2NVR, CLOUD_PROVIDER_AZURE/uk), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.445, container wall 666.1 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M418QHJBSP7EJXJW46MC2NVR CLOUD_PROVIDER_AZURE/uk]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 75.20 | 0.36 | control | 7.34 | control | control | 23.91 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| w128-256-768 | `{"widths": [128, 256, 768]}` | 8/8 | 75.23 | 0.25 | +0.03 +- 0.15 | 6.18 | -1.16 | -1.24 | 125.92 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| w128-384-512 | `{"widths": [128, 384, 512]}` | 8/8 | 75.01 | 0.23 | -0.19 +- 0.14 | 6.94 | -0.39 | +0.02 | 110.42 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
