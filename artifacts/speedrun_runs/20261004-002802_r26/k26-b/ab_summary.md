# r26/k26-b

2026-10-04 00:35, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M421HGY9KJAHM1SV0JMDTGER, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 432.7 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M421H5RBAK1XG778A7JTY03R CLOUD_PROVIDER_AZURE/eu-west]', 'NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M421HBD4XCYVVE9PDY2X7F1R CLOUD_PROVIDER_AZURE/us-west]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M421HGY9KJAHM1SV0JMDTGER CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wd0.024 | `{"weight_decay": 0.024}` | 8/8 | 74.89 | 0.11 | -0.18 +- 0.11 | 4.44 | +0.01 | +0.19 | 42.44 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.07 | 0.28 | control | 4.43 | control | control | 34.31 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| bias16 | `{"bias_scaler": 16.0}` | 8/8 | 75.04 | 0.22 | -0.04 +- 0.10 | 4.43 | +0.01 | +0.04 | 35.82 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| bn0.7 | `{"bn_momentum": 0.7}` | 8/8 | 75.00 | 0.40 | -0.08 +- 0.18 | 4.43 | +0.01 | +0.08 | 81.50 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
