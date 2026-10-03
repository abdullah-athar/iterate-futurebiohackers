# A/B p1a-lowres28x4

2026-10-03 16:22, NVIDIA A100 80GB PCIe (power limit 300.00 W, task ta-01M41520XMYT7CMD0MRYJY6THR, CLOUD_PROVIDER_AZURE/eu-west), one container, sequential, 8 trial(s) per variant, seeds from 0, container wall 697.2 s. PCIe guard attempts: ['NVIDIA A100-SXM4-80GB [task ta-01M4151KP25CGXR18QPKPKJ0MR CLOUD_PROVIDER_GCP/eu-west]', 'NVIDIA A100-SXM4-80GB [task ta-01M4151SQN54VRKVAXVHDMFMAR CLOUD_PROVIDER_UNSPECIFIED/eu]', 'NVIDIA A100-SXM4-80GB [task ta-01M4151XH4BPSAKWXY3G1VFF1R CLOUD_PROVIDER_UNSPECIFIED/us-east]', 'NVIDIA A100 80GB PCIe [task ta-01M41520XMYT7CMD0MRYJY6THR CLOUD_PROVIDER_AZURE/eu-west]'].

| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | dtime s | prepare s | build s cold | GPU | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `{}` | 8/8 | 75.32 | 0.41 | control | 8.04 | control | 0.081 | 153.56 | NVIDIA A100 80GB PCIe | QUALIFIED (>= 75%) |
| lowres28x4 | `{"low_res": 28, "low_res_epochs": 4}` | 8/8 | 75.02 | 0.25 | -0.30 +- 0.17 | 7.34 | -0.70 | 0.080 | 387.23 | NVIDIA A100 80GB PCIe | QUALIFIED (>= 75%); BUILD > 300 s (387 s cold) |
