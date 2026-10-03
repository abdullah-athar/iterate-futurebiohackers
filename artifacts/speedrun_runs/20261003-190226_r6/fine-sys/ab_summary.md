# r6/fine-sys

2026-10-03 19:12, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41EYPVA95ZXPZ8ARXQD0NER, CLOUD_PROVIDER_GCP/ap-southeast), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 527.4 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41EXCCP0APTCE8N0E1E7HYR CLOUD_PROVIDER_UNSPECIFIED/us-east]', 'NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M41EYA3MHH517PC43BT94SCR CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M41EYGBNSY933D9NSX2XPN2R CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41EYPVA95ZXPZ8ARXQD0NER CLOUD_PROVIDER_GCP/ap-southeast]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scale1.5 | `{"scaling_factor": 0.16666666666666666}` | 8/8 | 75.32 | 0.24 | +0.03 +- 0.13 | 5.85 | +0.11 | +0.08 | 115.58 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.29 | 0.24 | control | 5.75 | control | control | 32.76 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| jitter0.45 | `{"jitter": 0.45}` | 8/8 | 75.22 | 0.29 | -0.07 +- 0.14 | 5.80 | +0.06 | +0.13 | 31.55 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| crop-triton | `{"crop_mode": "triton"}` | 8/8 | 75.36 | 0.26 | +0.07 +- 0.08 | 5.71 | -0.04 | -0.11 | 79.54 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
