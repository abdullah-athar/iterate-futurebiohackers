# r15/knobs-2

2026-10-03 20:50, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41MPJHX0XVHPSPDW2BD4CKR, CLOUD_PROVIDER_AZURE/uk), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 418.4 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41MPJHX0XVHPSPDW2BD4CKR CLOUD_PROVIDER_AZURE/uk]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ls0.3 | `{"label_smoothing": 0.3}` | 8/8 | 75.33 | 0.16 | -0.19 +- 0.07 | 5.01 | -0.05 | +0.23 | 39.63 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.53 | 0.22 | control | 5.06 | control | control | 33.90 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| whiten4 | `{"whiten_bias_epochs": 4}` | 8/8 | 75.46 | 0.19 | -0.07 +- 0.12 | 5.07 | +0.01 | +0.12 | 39.30 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| translate1 | `{"translate": 1}` | 8/8 | 75.47 | 0.21 | -0.06 +- 0.10 | 5.03 | -0.03 | +0.05 | 46.20 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
