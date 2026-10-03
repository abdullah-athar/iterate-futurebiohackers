# r22/g3lad-b

2026-10-03 23:22, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41XDMDF6P2ZQPAJGPPEJ78R, CLOUD_PROVIDER_AZURE/uk), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 380.4 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41XDEMAMTKZG4NNB0V152JR CLOUD_PROVIDER_AZURE/uk]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41XDMDF6P2ZQPAJGPPEJ78R CLOUD_PROVIDER_AZURE/uk]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g3-inner512-e10.25 | `{"epochs": 10.25, "g3_pair": "inner512"}` | 8/8 | 75.20 | 0.22 | -0.00 +- 0.12 | 4.49 | +0.02 | +0.02 | 57.46 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 8/8 | 75.20 | 0.27 | control | 4.47 | control | control | 55.13 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| g3-inner512-e10.0-augoff0.5 | `{"aug_off_last": 0.5, "epochs": 10.0, "g3_pair": "inner512"}` | 8/8 | 75.14 | 0.21 | -0.06 +- 0.09 | 4.38 | -0.09 | -0.03 | 53.46 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
