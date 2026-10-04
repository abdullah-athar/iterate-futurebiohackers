# r35/width-c

2026-10-04 02:43, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M428PSXNYQ9KKN2AATV1MF4R, CLOUD_PROVIDER_AWS/us-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 620.1 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M428PKQM0ZZ74MNX2F56E1GR CLOUD_PROVIDER_AZURE/eu-south]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M428PSXNYQ9KKN2AATV1MF4R CLOUD_PROVIDER_AWS/us-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w288-e9.5 | `{"epochs": 9.5, "g2_pair": "inner192", "global_pool": "fullpool_avgsum", "widths": [64, 288, 768]}` | 8/8 | 75.22 | 0.28 | -0.02 +- 0.10 | 4.48 | +0.29 | +0.31 | 172.25 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.23 | 0.32 | control | 4.19 | control | control | 105.01 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| w288-g896-e9.5 | `{"epochs": 9.5, "g2_pair": "inner192", "global_pool": "fullpool_avgsum", "widths": [64, 288, 896]}` | 8/8 | 75.14 | 0.25 | -0.09 +- 0.16 | 4.89 | +0.70 | +0.79 | 141.59 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
