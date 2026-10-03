# r6/k2

2026-10-03 19:08, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41EY94T7D1S6V3F84DKGK2R, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 335.6 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41EX86GVMF7K1GMSZBT597R CLOUD_PROVIDER_UNSPECIFIED/us-east]', 'NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M41EXD3M16Y36DD7EE7EFH9R CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41EY94T7D1S6V3F84DKGK2R CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e8.25 | `{"epochs": 8.25}` | 8/8 | 75.13 | 0.27 | -0.23 +- 0.14 | 5.30 | -0.37 | -0.13 | 26.65 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.37 | 0.25 | control | 5.67 | control | control | 25.03 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| e9.25 | `{"epochs": 9.25}` | 8/8 | 75.54 | 0.25 | +0.17 +- 0.12 | 5.91 | +0.24 | +0.07 | 26.71 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| switch0.35 | `{"resolution_switch": 0.35}` | 8/8 | 75.05 | 0.14 | -0.32 +- 0.12 | 5.36 | -0.31 | +0.00 | 24.46 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
