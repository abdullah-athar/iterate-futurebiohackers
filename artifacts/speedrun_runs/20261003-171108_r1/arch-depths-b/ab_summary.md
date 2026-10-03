# r1/arch-depths-b

2026-10-03 17:24, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M418RHA1Q48VJ7T7SQNDEG2R, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.445, container wall 531.3 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M418QHJTX5KH6XGES7CWH1QR CLOUD_PROVIDER_AZURE/eu-west]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M418RHA1Q48VJ7T7SQNDEG2R CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 75.13 | 0.22 | control | 7.44 | control | control | 24.70 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| d3-3-2 | `{"depths": [3, 3, 2]}` | 8/8 | 74.19 | 0.13 | -0.94 +- 0.08 | 7.11 | -0.33 | +1.78 | 56.81 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| silu | `{"activation": "silu"}` | 8/8 | 75.00 | 0.38 | -0.13 +- 0.18 | 7.24 | -0.21 | +0.09 | 68.77 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
