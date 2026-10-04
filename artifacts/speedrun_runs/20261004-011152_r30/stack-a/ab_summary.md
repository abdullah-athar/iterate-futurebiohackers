# r30/stack-a

2026-10-04 01:23, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M4243JN3SHCM7PQY8T5E1HMR, CLOUD_PROVIDER_GCP/us-central), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 623.9 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M4241E68BBXBQGN5EMMC07YR CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M424357WBM8Z69BWP0KE2V0R CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M4243CZFZSZVBH5D3ZRF0VKR CLOUD_PROVIDER_AZURE/eu-west]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M4243JN3SHCM7PQY8T5E1HMR CLOUD_PROVIDER_GCP/us-central]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g2-192+maxmeansum-e10.0 | `{"g2_pair": "inner192", "global_pool": "maxmean_sum"}` | 8/8 | 75.24 | 0.24 | +0.21 +- 0.12 | 4.33 | -0.11 | -0.32 | 212.33 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.03 | 0.26 | control | 4.43 | control | control | 44.82 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| g2-192+maxmeansum-e9.5 | `{"epochs": 9.5, "g2_pair": "inner192", "global_pool": "maxmean_sum"}` | 8/8 | 74.90 | 0.20 | -0.13 +- 0.07 | 4.14 | -0.30 | -0.16 | 46.26 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| g2-192+maxmeansum-e9.25 | `{"epochs": 9.25, "g2_pair": "inner192", "global_pool": "maxmean_sum"}` | 8/8 | 74.80 | 0.21 | -0.23 +- 0.11 | 4.02 | -0.42 | -0.18 | 46.14 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
