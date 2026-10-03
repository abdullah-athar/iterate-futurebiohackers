# r20/stackB-a

2026-10-03 22:55, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41VVX4HXBN7RC57EYRFR5MR, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 400.8 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41VT5GWY3N10EWHHR4T4NMR CLOUD_PROVIDER_UNSPECIFIED/us-east]', 'NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41VTAVZCHEXT699YYVY029R CLOUD_PROVIDER_AZURE/eu-west]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41VVX4HXBN7RC57EYRFR5MR CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B-augoff0.5 | `{"aug_off_last": 0.5, "epochs": 9.5, "resolution_schedule": [[24, 0.25], [28, 0.5]]}` | 8/8 | 75.28 | 0.27 | +0.08 +- 0.04 | 4.53 | -0.01 | -0.09 | 35.98 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.20 | 0.27 | control | 4.55 | control | control | 35.32 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| B-augoff0.5-e9.375 | `{"aug_off_last": 0.5, "epochs": 9.375, "resolution_schedule": [[24, 0.25], [28, 0.5]]}` | 8/8 | 75.18 | 0.38 | -0.03 +- 0.13 | 4.59 | +0.05 | +0.08 | 34.96 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| B-augoff0.5-e9.25 | `{"aug_off_last": 0.5, "epochs": 9.25, "resolution_schedule": [[24, 0.25], [28, 0.5]]}` | 8/8 | 75.17 | 0.24 | -0.04 +- 0.12 | 4.42 | -0.13 | -0.09 | 49.16 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
