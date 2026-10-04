# r33/muon-ov-b

2026-10-04 02:01, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M426KND0QRB05PQQ33VS2QHR, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 300.4 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M426KHCHPDFP6KTCWPGSR4QR CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M426KND0QRB05PQQ33VS2QHR CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| muon-g3only-ns2-e10.0 | `{"epochs": 10.0, "muon_groups": [2], "muon_lr": 0.24, "muon_momentum": 0.6, "muon_ns_steps": 2, "optimizer": "muon_airbench"}` | 8/8 | 75.13 | 0.17 | +0.07 +- 0.12 | 4.93 | +0.47 | +0.40 | 35.24 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.06 | 0.30 | control | 4.46 | control | control | 34.66 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| muon-g3only-ns2-e8.0 | `{"epochs": 8.0, "muon_groups": [2], "muon_lr": 0.24, "muon_momentum": 0.6, "muon_ns_steps": 2, "optimizer": "muon_airbench"}` | 8/8 | 73.87 | 0.16 | -1.19 +- 0.15 | 3.96 | -0.50 | +0.69 | 35.11 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
