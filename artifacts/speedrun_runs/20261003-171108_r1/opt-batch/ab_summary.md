# r1/opt-batch

2026-10-03 17:42, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M419AJC2X0V1P73C4ER35MYR, CLOUD_PROVIDER_UNSPECIFIED/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.445, container wall 1033.0 s. GPU guard attempts: ['NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M419AJC2X0V1P73C4ER35MYR CLOUD_PROVIDER_UNSPECIFIED/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 75.27 | 0.20 | control | 7.43 | control | control | 18.26 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| bs768 | `{"batch_size": 768}` | 8/8 | 75.55 | 0.35 | +0.28 +- 0.17 | 7.77 | +0.34 | -0.29 | 117.86 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| bs1536 | `{"batch_size": 1536}` | 8/8 | 75.11 | 0.25 | -0.16 +- 0.09 | 7.25 | -0.18 | +0.17 | 114.93 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
