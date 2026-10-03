# r11/d2-k

2026-10-03 19:56, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41HKZP2THAQ7GMZ9WPTZM8R, CLOUD_PROVIDER_GCP/us-central), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 401.5 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41HGEANJF5SHSV1HMEA9NGR CLOUD_PROVIDER_AZURE/eu-south]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41HKZP2THAQ7GMZ9WPTZM8R CLOUD_PROVIDER_GCP/us-central]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| d222-e9.5 | `{"depths": [2, 2, 2], "epochs": 9.5}` | 8/8 | 73.79 | 0.25 | -1.50 +- 0.13 | 4.61 | -0.99 | +1.15 | 72.67 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.29 | 0.24 | control | 5.60 | control | control | 29.04 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| d222-e10.0 | `{"depths": [2, 2, 2], "epochs": 10.0}` | 8/8 | 74.00 | 0.10 | -1.30 +- 0.07 | 4.83 | -0.77 | +1.08 | 23.30 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| d222-e10.5 | `{"depths": [2, 2, 2], "epochs": 10.5}` | 8/8 | 74.28 | 0.23 | -1.02 +- 0.13 | 5.06 | -0.54 | +0.91 | 23.53 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
