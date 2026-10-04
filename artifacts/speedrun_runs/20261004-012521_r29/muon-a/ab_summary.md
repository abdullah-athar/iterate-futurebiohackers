# r29/muon-a

2026-10-04 01:32, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M424T9T3773RJMSFS94G2X8R, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 417.7 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M424T48672B1P2MHK0ANVS8R CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M424T9T3773RJMSFS94G2X8R CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| muon-ab-lr0.24 | `{"muon_lr": 0.24, "muon_momentum": 0.6, "muon_ns_steps": 3, "optimizer": "muon_airbench"}` | 8/8 | 76.00 | 0.25 | +0.93 +- 0.17 | 5.51 | +1.07 | +0.14 | 38.92 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.07 | 0.28 | control | 4.43 | control | control | 37.47 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| muon-ab-lr0.16 | `{"muon_lr": 0.16, "muon_momentum": 0.6, "muon_ns_steps": 3, "optimizer": "muon_airbench"}` | 8/8 | 75.87 | 0.28 | +0.80 +- 0.12 | 5.53 | +1.10 | +0.30 | 37.51 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| muon-ab-lr0.32 | `{"muon_lr": 0.32, "muon_momentum": 0.6, "muon_ns_steps": 3, "optimizer": "muon_airbench"}` | 8/8 | 75.37 | 0.28 | +0.29 +- 0.12 | 5.54 | +1.10 | +0.81 | 37.14 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
