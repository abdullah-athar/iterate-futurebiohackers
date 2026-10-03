# r20/method-a

2026-10-03 22:43, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41V45EZZPCN3TVG5Z2Z0SRR, CLOUD_PROVIDER_UNSPECIFIED/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 469.8 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41V3F2FNK16D951XBDW1X9R CLOUD_PROVIDER_UNSPECIFIED/us-east]', 'NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M41V3ME1E1KFPZ4QX0TPP83R CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M41V3SAZ425HVW3K2R48S8XR CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41V45EZZPCN3TVG5Z2Z0SRR CLOUD_PROVIDER_UNSPECIFIED/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A-augoff0.5 | `{"aug_off_last": 0.5, "epochs": 9.25, "resolution_schedule": [[24, 0.25], [28, 0.5]]}` | 8/8 | 75.23 | 0.29 | +0.09 +- 0.13 | 4.46 | -0.01 | -0.10 | 38.95 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.14 | 0.24 | control | 4.47 | control | control | 36.56 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| A-augoff1.0 | `{"aug_off_last": 1.0, "epochs": 9.25, "resolution_schedule": [[24, 0.25], [28, 0.5]]}` | 8/8 | 75.19 | 0.20 | +0.05 +- 0.12 | 4.47 | -0.01 | -0.06 | 37.93 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| A-ls0.35to0.15 | `{"epochs": 9.25, "label_smoothing": 0.35, "label_smoothing_end": 0.15, "resolution_schedule": [[24, 0.25], [28, 0.5]]}` | 8/8 | 75.02 | 0.40 | -0.12 +- 0.15 | 4.47 | -0.00 | +0.12 | 115.50 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
