# r20/schedB-b

2026-10-03 22:52, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41VVX4ACWZMMK9CZ5HXH8VR, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 207.3 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41VT52B2DK3Y8HD761NCXGR CLOUD_PROVIDER_AZURE/eu-west]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41VVX4ACWZMMK9CZ5HXH8VR CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B-28to80 | `{"epochs": 9.5, "resolution_schedule": [[28, 0.8]]}` | 8/8 | 74.52 | 0.23 | -0.68 +- 0.16 | 4.65 | +0.11 | +0.79 | 29.44 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.20 | 0.27 | control | 4.54 | control | control | 37.76 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
