# r40/confirm16-r24

2026-10-04 02:59, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M429SAV70QCN4VYZPG58RCWR, CLOUD_PROVIDER_AWS/us-west), one container, sequential, 16 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 403.4 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M429S5J5S9TMYV50N44Z5VRR CLOUD_PROVIDER_AZURE/us-central]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M429SAV70QCN4VYZPG58RCWR CLOUD_PROVIDER_AWS/us-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r24x0.35-e10.0-n16 | `{"g2_pair": "inner192", "global_pool": "fullpool_avgsum", "resolution_schedule": [[24, 0.35], [28, 0.5]]}` | 16/16 | 75.14 | 0.21 | -0.01 +- 0.06 | 4.04 | -0.13 | -0.12 | 38.51 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 16/16 | 75.16 | 0.19 | control | 4.17 | control | control | 34.48 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| r24x0.35-contig-e10.0-n16 | `{"batch_order": "contiguous", "g2_pair": "inner192", "global_pool": "fullpool_avgsum", "resolution_schedule": [[24, 0.35], [28, 0.5]]}` | 16/16 | 75.10 | 0.21 | -0.06 +- 0.05 | 4.03 | -0.15 | -0.09 | 34.51 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
