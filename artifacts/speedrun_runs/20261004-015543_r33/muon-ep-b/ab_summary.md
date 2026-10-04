# r33/muon-ep-b

2026-10-04 02:03, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M426M3CK0ZHYRRE0QHN3QAPR, CLOUD_PROVIDER_GCP/us-central), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 371.8 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M426HQY6XA6DAAD6EF80CYTR CLOUD_PROVIDER_AZURE/us-west]', 'NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M426KHC9Q9TDPJHA4A212E6R CLOUD_PROVIDER_AZURE/eu-south]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M426M3CK0ZHYRRE0QHN3QAPR CLOUD_PROVIDER_GCP/us-central]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| muon-lr0.24-e8.0 | `{"epochs": 8.0, "muon_lr": 0.24, "muon_momentum": 0.6, "muon_ns_steps": 3, "optimizer": "muon_airbench"}` | 8/8 | 75.25 | 0.34 | +0.18 +- 0.10 | 4.61 | +0.14 | -0.03 | 50.35 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.07 | 0.28 | control | 4.46 | control | control | 46.61 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| muon-lr0.24-e8.5 | `{"epochs": 8.5, "muon_lr": 0.24, "muon_momentum": 0.6, "muon_ns_steps": 3, "optimizer": "muon_airbench"}` | 8/8 | 75.43 | 0.38 | +0.36 +- 0.18 | 4.97 | +0.51 | +0.15 | 50.11 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
