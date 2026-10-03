# r9/stack-c

2026-10-03 19:47, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41H5MZEHY6WXXWCXZVAG51R, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 318.1 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41H3X70VRD8VVCD9QBQFZCR CLOUD_PROVIDER_AZURE/eu-south]', 'NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41H4204M01J3JHY46N8GSQR CLOUD_PROVIDER_AZURE/eu-south]', 'NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41H46FAPH6FBDCE5JK76N4R CLOUD_PROVIDER_AZURE/eu-west]', 'NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M41H4SVAQPK3KHGAXS47PPJR CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M41H5D90VGX5HHH917ZKBYQR CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41H5MZEHY6WXXWCXZVAG51R CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S2-e8.75 | `{"bias_scaler": 16.0, "lr": 12.0, "weight_decay": 0.0168, "widths": [64, 256, 768]}` | 8/8 | 75.07 | 0.11 | +0.24 +- 0.06 | 4.83 | -0.02 | -0.36 | 24.11 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 74.83 | 0.13 | control | 4.85 | control | control | 23.39 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| S2-e9.0 | `{"bias_scaler": 16.0, "epochs": 9.0, "lr": 12.0, "weight_decay": 0.0168, "widths": [64, 256, 768]}` | 8/8 | 75.12 | 0.26 | +0.28 +- 0.10 | 4.97 | +0.13 | -0.28 | 22.82 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| S2-e9.25 | `{"bias_scaler": 16.0, "epochs": 9.25, "lr": 12.0, "weight_decay": 0.0168, "widths": [64, 256, 768]}` | 8/8 | 75.43 | 0.29 | +0.60 +- 0.10 | 5.11 | +0.26 | -0.60 | 22.63 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
