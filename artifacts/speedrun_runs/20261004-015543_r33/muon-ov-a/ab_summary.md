# r33/muon-ov-a

2026-10-04 02:04, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M426HR5MRWXEAVMHSKMN3M3R, CLOUD_PROVIDER_GCP/us-central), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 516.3 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M426HR5MRWXEAVMHSKMN3M3R CLOUD_PROVIDER_GCP/us-central]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| muon-ns2-e10.0 | `{"epochs": 10.0, "muon_lr": 0.24, "muon_momentum": 0.6, "muon_ns_steps": 2, "optimizer": "muon_airbench"}` | 8/8 | 75.13 | 0.27 | +0.08 +- 0.13 | 5.61 | +1.16 | +1.09 | 56.30 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.06 | 0.26 | control | 4.44 | control | control | 52.66 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| muon-g3only-e10.0 | `{"epochs": 10.0, "muon_groups": [2], "muon_lr": 0.24, "muon_momentum": 0.6, "muon_ns_steps": 3, "optimizer": "muon_airbench"}` | 8/8 | 75.63 | 0.20 | +0.57 +- 0.12 | 5.01 | +0.57 | +0.00 | 50.88 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| muon-ns3-e10.0-ref | `{"epochs": 10.0, "muon_lr": 0.24, "muon_momentum": 0.6, "muon_ns_steps": 3, "optimizer": "muon_airbench"}` | 8/8 | 75.96 | 0.33 | +0.91 +- 0.11 | 6.01 | +1.57 | +0.66 | 49.75 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
