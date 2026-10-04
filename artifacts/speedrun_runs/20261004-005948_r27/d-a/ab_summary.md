# r27/d-a

2026-10-04 01:10, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M423D2SR7ZW790K15SMYH3JR, CLOUD_PROVIDER_UNSPECIFIED/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 577.4 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M423BB4ZET85A5A934QVS9BR CLOUD_PROVIDER_UNSPECIFIED/us-east]', 'NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M423CXVG7CN6H5J38WZ7YGGR CLOUD_PROVIDER_AZURE/us-west]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M423D2SR7ZW790K15SMYH3JR CLOUD_PROVIDER_UNSPECIFIED/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g3-inner448 | `{"g3_pair": "inner448"}` | 8/8 | 74.99 | 0.36 | -0.09 +- 0.15 | 4.46 | -0.04 | +0.05 | 154.06 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.07 | 0.28 | control | 4.50 | control | control | 41.25 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| g3-inner384 | `{"g3_pair": "inner384"}` | 8/8 | 74.98 | 0.15 | -0.09 +- 0.13 | 4.34 | -0.16 | -0.07 | 158.02 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
