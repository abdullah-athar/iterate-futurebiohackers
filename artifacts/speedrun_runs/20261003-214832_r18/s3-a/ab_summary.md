# r18/s3-a

2026-10-03 21:56, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41RD86NWVXX4VBQ1AHWMMMR, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 491.2 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M41RD49T3PGRF1BS9PDAEVWR CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41RD86NWVXX4VBQ1AHWMMMR CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r20-24-28-e9.5 | `{"epochs": 9.5, "resolution_schedule": [[20, 0.15], [24, 0.35], [28, 0.55]]}` | 8/8 | 74.88 | 0.28 | -0.23 +- 0.12 | 4.32 | -0.44 | -0.20 | 100.34 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.12 | 0.19 | control | 4.75 | control | control | 49.71 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| r20-24-28-e9.75 | `{"epochs": 9.75, "resolution_schedule": [[20, 0.15], [24, 0.35], [28, 0.55]]}` | 8/8 | 74.91 | 0.28 | -0.21 +- 0.14 | 4.49 | -0.26 | -0.05 | 91.85 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
