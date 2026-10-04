# r37/g1-a

2026-10-04 02:49, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M429684BZHZG11WC2YPKSQ6R, CLOUD_PROVIDER_AWS/us-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 428.6 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M4292DG2RC885YFP71S1093R CLOUD_PROVIDER_UNSPECIFIED/us-east]', 'NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M429589S20RTSD7C8NZD2TER CLOUD_PROVIDER_UNSPECIFIED/us-east]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M429684BZHZG11WC2YPKSQ6R CLOUD_PROVIDER_AWS/us-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g1-48-e10.0 | `{"g1_pair": "inner48", "g2_pair": "inner192", "global_pool": "fullpool_avgsum"}` | 8/8 | 75.08 | 0.14 | -0.18 +- 0.10 | 4.18 | -0.06 | +0.13 | 161.87 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.26 | 0.27 | control | 4.24 | control | control | 37.36 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| g1-48-e9.75 | `{"epochs": 9.75, "g1_pair": "inner48", "g2_pair": "inner192", "global_pool": "fullpool_avgsum"}` | 8/8 | 74.92 | 0.19 | -0.34 +- 0.12 | 4.07 | -0.17 | +0.17 | 36.66 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
