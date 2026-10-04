# r30/stack-b

2026-10-04 01:24, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M4243VVWVGYQ3MZGBQZEKH5R, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 695.6 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M4241E68B4AMQZ7XDV090JBR CLOUD_PROVIDER_AZURE/eu-west]', 'NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M424357WKZYX2K7VZAT6VM7R CLOUD_PROVIDER_AZURE/us-west]', 'NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M4243DFAGWWE0FEMJV464V5R CLOUD_PROVIDER_AZURE/us-west]', 'NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M4243K11VHM2D4YMYZJBNM9R CLOUD_PROVIDER_AZURE/us-west]', 'NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M4243R4Z5ZHMSFZAG7Z43VYR CLOUD_PROVIDER_AZURE/eu-west]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M4243VVWVGYQ3MZGBQZEKH5R CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g2-192+maxmeansum-e9.75 | `{"epochs": 9.75, "g2_pair": "inner192", "global_pool": "maxmean_sum"}` | 8/8 | 74.98 | 0.21 | -0.07 +- 0.11 | 4.24 | -0.21 | -0.14 | 167.21 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.05 | 0.20 | control | 4.45 | control | control | 33.70 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| maxmeansum-e9.5 | `{"epochs": 9.5, "global_pool": "maxmean_sum"}` | 8/8 | 75.03 | 0.22 | -0.02 +- 0.12 | 4.25 | -0.20 | -0.18 | 115.64 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| g2-192+maxmeancat-e9.5 | `{"epochs": 9.5, "g2_pair": "inner192", "global_pool": "maxmean_cat"}` | 8/8 | 74.97 | 0.24 | -0.08 +- 0.10 | 4.13 | -0.32 | -0.24 | 142.90 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
