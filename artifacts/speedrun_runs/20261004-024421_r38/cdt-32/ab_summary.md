# r38/cdt-32

2026-10-04 02:49, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M429AYNNYV06MT5H9RMJ1ARR, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 296.4 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M429ASBRE9TTTY6YW9Q9WEDR CLOUD_PROVIDER_AZURE/eu-south]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M429AYNNYV06MT5H9RMJ1ARR CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cdt-32px-only | `{"g2_pair": "inner192", "global_pool": "fullpool_avgsum", "inductor_tuning": ["coordinate_descent_tuning"], "inductor_tuning_resolutions": [32]}` | 8/8 | 75.15 | 0.17 | +0.03 +- 0.10 | 4.15 | +0.00 | -0.03 | 114.47 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.12 | 0.18 | control | 4.14 | control | control | 35.07 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
