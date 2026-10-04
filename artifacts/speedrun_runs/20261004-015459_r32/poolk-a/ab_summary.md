# r32/poolk-a

2026-10-04 02:05, NVIDIA A100-SXM4-80GB @ 400.00 W (task ta-01M426H9Q6ATA6068YGPBKZVJR, CLOUD_PROVIDER_AWS/us-east), one container, sequential, 16 trial(s) per run, seeds from 0, warm compile cache, k = 1.0, container wall 589.4 s. GPU guard attempts: ['NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M426GC55VKR0A0RK61XA70FR CLOUD_PROVIDER_AZURE/eu-south]', 'NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M426GJGJX23CT1ED6ZMD9X8R CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M426GP051RB1XZZPMHZK4X7R CLOUD_PROVIDER_AZURE/us-west]', 'NVIDIA A100-SXM4-80GB @ 500.00 W [task ta-01M426GSDYS67065T2XYQD3CMR CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M426GYM456GDS6GA9A23EC7R CLOUD_PROVIDER_AZURE/us-west]', 'NVIDIA A100 80GB PCIe @ 300.00 W [task ta-01M426H3Y6YJ0T7PG1JVFVAT4R CLOUD_PROVIDER_AZURE/us-west]', 'NVIDIA A100-SXM4-80GB @ 400.00 W [task ta-01M426H9Q6ATA6068YGPBKZVJR CLOUD_PROVIDER_AWS/us-east]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fullpool-n16 | `{"global_pool": "fullpool"}` | 16/16 | 75.17 | 0.21 | -0.03 +- 0.08 | 4.28 | -0.14 | -0.10 | 130.23 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| control | `{}` | 16/16 | 75.20 | 0.27 | control | 4.41 | control | control | 34.47 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
| fullpool-sum-n16 | `{"global_pool": "fullpool_sum"}` | 16/16 | 75.21 | 0.28 | +0.01 +- 0.10 | 4.35 | -0.07 | -0.07 | 121.41 (warm) | 0 | NVIDIA A100-SXM4-80GB @ 400.00 W | QUALIFIED (>= 75%) |
