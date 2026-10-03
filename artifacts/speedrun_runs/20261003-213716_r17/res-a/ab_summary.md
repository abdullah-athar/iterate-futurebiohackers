# r17/res-a

2026-10-03 21:42, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41QRVDNZNBW16FAFSBWFZAR, CLOUD_PROVIDER_GCP/eu-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 319.8 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41QRFZB3EMYS6X6H99E7FBR CLOUD_PROVIDER_AZURE/eu-south]', 'NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M41QRP8GZ5KE16CTBXQ17B3R CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41QRVDNZNBW16FAFSBWFZAR CLOUD_PROVIDER_GCP/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r24q33-28h | `{"resolution_schedule": [[24, 0.33], [28, 0.5]]}` | 8/8 | 74.81 | 0.29 | -0.23 +- 0.10 | 4.20 | -0.50 | -0.27 | 45.81 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.04 | 0.24 | control | 4.70 | control | control | 34.79 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| r24q40-28q60 | `{"resolution_schedule": [[24, 0.4], [28, 0.6]]}` | 8/8 | 74.58 | 0.22 | -0.46 +- 0.11 | 4.03 | -0.67 | -0.21 | 45.03 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
