# r7/staged-b-2

2026-10-03 19:29, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41G4B39PFCAD5XW1KQVT2NR, CLOUD_PROVIDER_AZURE/uk), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 342.3 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41G1YQ76JSY1PWYXDJ6VD8R CLOUD_PROVIDER_UNSPECIFIED/us-east]', 'NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41G3V3B53DJ0TE6MWQC5VSR CLOUD_PROVIDER_AZURE/eu-west]', 'NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41G4135NR2S13FSHWDAZ1YR CLOUD_PROVIDER_AZURE/eu-west]', 'NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41G471KJ14W9T3H2HG00MDR CLOUD_PROVIDER_AZURE/eu-west]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41G4B39PFCAD5XW1KQVT2NR CLOUD_PROVIDER_AZURE/uk]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s16x15-20x30-24x45-28x60 | `{"res_schedule": [[16, 0.15], [20, 0.3], [24, 0.45], [28, 0.6]]}` | 8/8 | 74.38 | 0.27 | -0.91 +- 0.14 | 5.09 | -0.49 | +0.81 | 170.57 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.29 | 0.24 | control | 5.58 | control | control | 33.92 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
