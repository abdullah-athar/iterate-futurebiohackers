# r24/paired40-s40

2026-10-03 23:50, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41YWJM6R224VRZ3D2PGW8SR, CLOUD_PROVIDER_UNSPECIFIED/us-east), one container, sequential, 40 trial(s) per run, seeds from 40, warm compile cache, k = 1.0, container wall 524.9 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41YWDE3Y3YJCR8739M9449R CLOUD_PROVIDER_UNSPECIFIED/us-east]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41YWJM6R224VRZ3D2PGW8SR CLOUD_PROVIDER_UNSPECIFIED/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g3-inner512-e10.0-s40 | `{"epochs": 10.0, "g3_pair": "inner512"}` | 40/40 | 75.18 | 0.27 | +0.00 +- 0.05 | 4.50 | -0.11 | -0.11 | 38.57 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 40/40 | 75.18 | 0.22 | control | 4.61 | control | control | 36.61 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
