# r19/k19-a

2026-10-03 22:03, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41RXJ0T36KC993S9EK31H7R, CLOUD_PROVIDER_AWS/us-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 329.2 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41RV6Q26GDWE9QE9J4Y9QWR CLOUD_PROVIDER_UNSPECIFIED/us-east]', 'NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M41RVDPV7V5Q6Y7YAKE2KN2R CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M41RXAT4S8V46XFNXDPEV0PR CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41RXE99GA21DH3CNPHDB37R CLOUD_PROVIDER_AZURE/eu-west]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41RXJ0T36KC993S9EK31H7R CLOUD_PROVIDER_AWS/us-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| warmup0.15 | `{"epochs": 9.5, "resolution_schedule": [[24, 0.5]], "warmup": 0.15}` | 8/8 | 75.00 | 0.17 | -0.13 +- 0.05 | 4.15 | +0.00 | +0.13 | 28.18 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.13 | 0.21 | control | 4.15 | control | control | 27.32 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| warmup0.3 | `{"epochs": 9.5, "resolution_schedule": [[24, 0.5]], "warmup": 0.3}` | 8/8 | 74.98 | 0.20 | -0.15 +- 0.08 | 4.17 | +0.01 | +0.16 | 26.48 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| final0.03 | `{"epochs": 9.5, "final_lr": 0.03, "resolution_schedule": [[24, 0.5]]}` | 8/8 | 74.86 | 0.26 | -0.27 +- 0.11 | 4.16 | +0.01 | +0.28 | 26.58 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
