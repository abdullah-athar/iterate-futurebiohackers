# r34/paired40-s40-stack

2026-10-04 02:19, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M427B3SDVDD2SFMH7RAV419R, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 40 trial(s) per run, seeds from 40, warm compile cache, k = 1.0, container wall 576.4 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M427AZ6E998P5F8S38BHF99R CLOUD_PROVIDER_AZURE/eu-south]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M427B3SDVDD2SFMH7RAV419R CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g2-192-avgsum-e10.0-s40 | `{"epochs": 10.0, "g2_pair": "inner192", "global_pool": "fullpool_avgsum"}` | 40/40 | 75.19 | 0.28 | -0.02 +- 0.06 | 4.15 | -0.27 | -0.24 | 110.15 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 40/40 | 75.22 | 0.25 | control | 4.42 | control | control | 34.38 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
