# r33/muon-ep-a

2026-10-04 02:03, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M426M3CKS6EWWWWRZWCPH6CR, CLOUD_PROVIDER_GCP/us-central), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 363.7 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M426HR5MKQFNYEKEXC7ZA0XR CLOUD_PROVIDER_AZURE/us-west]', 'NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M426KHC9V74KRZ7V8EG9F7MR CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M426M3CKS6EWWWWRZWCPH6CR CLOUD_PROVIDER_GCP/us-central]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| muon-lr0.24-e7.0 | `{"epochs": 7.0, "muon_lr": 0.24, "muon_momentum": 0.6, "muon_ns_steps": 3, "optimizer": "muon_airbench"}` | 8/8 | 74.70 | 0.20 | -0.37 +- 0.15 | 4.09 | -0.39 | -0.02 | 51.95 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.07 | 0.28 | control | 4.48 | control | control | 46.61 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| muon-lr0.24-e7.5 | `{"epochs": 7.5, "muon_lr": 0.24, "muon_momentum": 0.6, "muon_ns_steps": 3, "optimizer": "muon_airbench"}` | 8/8 | 74.93 | 0.24 | -0.14 +- 0.15 | 4.48 | -0.01 | +0.13 | 47.88 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
