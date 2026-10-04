# r27/d-b

2026-10-04 01:09, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M423D8JNYBBEC464H7HZD02R, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 501.9 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M423BB8Y67DYZTMTC89PTEAR CLOUD_PROVIDER_AZURE/eu-south]', 'NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M423D21G5ZKHAFZ90CEYZ6NR CLOUD_PROVIDER_AZURE/us-west]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M423D8JNYBBEC464H7HZD02R CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g3-inner448-e10.25 | `{"epochs": 10.25, "g3_pair": "inner448"}` | 8/8 | 75.13 | 0.25 | +0.06 +- 0.12 | 4.51 | +0.07 | +0.01 | 182.85 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.07 | 0.28 | control | 4.44 | control | control | 50.49 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| g3-inner448-e10.5 | `{"epochs": 10.5, "g3_pair": "inner448"}` | 8/8 | 75.31 | 0.14 | +0.24 +- 0.10 | 4.62 | +0.18 | -0.06 | 47.41 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
