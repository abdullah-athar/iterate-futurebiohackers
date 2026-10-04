# r35/width-a

2026-10-04 02:41, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M428PR63E769SP6RQXG4WS6R, CLOUD_PROVIDER_AWS/us-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 495.0 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M428PKQCQ2XG5QTMTD6J7NHR CLOUD_PROVIDER_UNSPECIFIED/us-east]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M428PR63E769SP6RQXG4WS6R CLOUD_PROVIDER_AWS/us-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w896-e10.0 | `{"g2_pair": "inner192", "global_pool": "fullpool_avgsum", "widths": [64, 256, 896]}` | 8/8 | 75.25 | 0.26 | +0.10 +- 0.12 | 4.56 | +0.37 | +0.27 | 147.71 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.15 | 0.24 | control | 4.19 | control | control | 109.22 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| w896-e9.5 | `{"epochs": 9.5, "g2_pair": "inner192", "global_pool": "fullpool_avgsum", "widths": [64, 256, 896]}` | 8/8 | 74.97 | 0.24 | -0.18 +- 0.08 | 4.35 | +0.16 | +0.34 | 36.60 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
