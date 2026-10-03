# r14/stems-a

2026-10-03 20:37, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41KX41HSX7VPNSBW13GE3GR, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 8 trial(s) per run, seeds from 0, warm compile cache, k = 0.7, container wall 474.7 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41KWJED338BNK4RTHRW1HYR CLOUD_PROVIDER_AZURE/eu-south]', 'NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41KWPTQNV5QYFHT7K5311JR CLOUD_PROVIDER_AZURE/eu-south]', 'NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M41KWYHYNPSDZT0WYV77RFDR CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41KX41HSX7VPNSBW13GE3GR CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stem-s2d | `{"stem": "space_to_depth"}` | 8/8 | 65.87 | 0.20 | -9.66 +- 0.10 | 2.15 | -2.88 | +10.92 | 106.58 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| control | `{}` | 8/8 | 75.53 | 0.22 | control | 5.03 | control | control | 25.70 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| stem-s2d-nopool | `{"stem": "space_to_depth_nopool"}` | 8/8 | 73.10 | 0.17 | -2.43 +- 0.09 | 4.84 | -0.19 | +3.28 | 61.54 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
| stem-conv1x1 | `{"stem": "conv1x1"}` | 8/8 | 74.49 | 0.34 | -1.04 +- 0.17 | 4.90 | -0.13 | +1.36 | 76.46 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | BELOW 75% |
