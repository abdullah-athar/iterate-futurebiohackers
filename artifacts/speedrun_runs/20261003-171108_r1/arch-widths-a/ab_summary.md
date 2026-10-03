# r1/arch-widths-a

2026-10-03 17:22, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M418QHJJ5WRNFMK20F5NJSRR, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.445, container wall 448.3 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M418QHJJ5WRNFMK20F5NJSRR CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 75.31 | 0.25 | control | 7.40 | control | control | 17.36 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| w96-384-576 | `{"widths": [96, 384, 576]}` | 8/8 | 75.05 | 0.20 | -0.26 +- 0.10 | 6.81 | -0.58 | +0.00 | 52.82 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| w64-384-576 | `{"widths": [64, 384, 576]}` | 8/8 | 74.37 | 0.22 | -0.94 +- 0.11 | 5.82 | -1.58 | +0.54 | 54.70 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
