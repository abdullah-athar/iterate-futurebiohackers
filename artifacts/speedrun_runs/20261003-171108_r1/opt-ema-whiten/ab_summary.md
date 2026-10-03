# r1/opt-ema-whiten

2026-10-03 17:33, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41976CEGZ683Y3XAWPX2JSR, CLOUD_PROVIDER_AWS/us-west), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.445, container wall 601.4 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41976CEGZ683Y3XAWPX2JSR CLOUD_PROVIDER_AWS/us-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 75.30 | 0.32 | control | 7.32 | control | control | 17.29 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| ema3 | `{"ema_every": 3}` | 8/8 | 75.16 | 0.22 | -0.14 +- 0.14 | 7.35 | +0.03 | +0.35 | 15.88 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| ema10 | `{"ema_every": 10}` | 8/8 | 75.11 | 0.14 | -0.18 +- 0.14 | 7.33 | +0.01 | +0.42 | 15.48 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| whiten2 | `{"whiten_bias_epochs": 2}` | 8/8 | 75.15 | 0.18 | -0.15 +- 0.12 | 7.29 | -0.03 | +0.30 | 15.50 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
