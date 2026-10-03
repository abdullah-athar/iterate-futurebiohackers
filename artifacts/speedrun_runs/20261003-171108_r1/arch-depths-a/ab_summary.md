# r1/arch-depths-a

2026-10-03 17:22, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M418V8GVEND2Z5NY2MHX2Q4R, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.445, container wall 322.1 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M418QHJTJXMRK51BEQEBG0ZR CLOUD_PROVIDER_AZURE/eu-west]', 'NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M418RHA1YVFN0556PHJQAXVR CLOUD_PROVIDER_UNSPECIFIED/us-east]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M418V8GVEND2Z5NY2MHX2Q4R CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 75.13 | 0.22 | control | 7.39 | control | control | 17.03 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| d2-3-3 | `{"depths": [2, 3, 3]}` | 8/8 | 74.96 | 0.24 | -0.17 +- 0.08 | 6.80 | -0.59 | -0.21 | 38.16 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| d3-2-3 | `{"depths": [3, 2, 3]}` | 8/8 | 75.01 | 0.22 | -0.11 +- 0.14 | 6.53 | -0.87 | -0.61 | 40.08 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
