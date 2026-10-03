# r6/diag-recompile

2026-10-03 19:05, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M41EYPVAJNAQFVRAKQSS4FYR, CLOUD_PROVIDER_GCP/ap-southeast), one container, sequential, 3 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 94.6 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41EX13PSK1QSQWGY3EJP6MR CLOUD_PROVIDER_UNSPECIFIED/us-east]', 'NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M41EXBYVR04FD4B64BJB116R CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M41EY94M3EAHEPGM6FPR46RR CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M41EYDSH99JM9AF5KK9C0ZKR CLOUD_PROVIDER_UNSPECIFIED/us-east]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M41EYPVAJNAQFVRAKQSS4FYR CLOUD_PROVIDER_GCP/ap-southeast]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| defaults-torchlogs | `{}` | 3/3 | 75.51 | 0.21 | control | 5.99 | control | control | 44.19 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
