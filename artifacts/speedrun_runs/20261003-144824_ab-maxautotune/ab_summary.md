# A/B ab-maxautotune

2026-10-03 14:48, NVIDIA A100 80GB PCIe (power limit 300.00 W, task ta-01M4100Q0KXSNK4ATETQHE7XER), one container, sequential, 2 trial(s) per variant, container wall 359.5 s. PCIe guard attempts: ['NVIDIA A100 80GB PCIe [task ta-01M4100Q0KXSNK4ATETQHE7XER]'].

| variant | config | n | mean acc % | std | mean prep+train s | std | prepare s | train s | eval s | build s | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `futurebiohackers {"use_compile": true}` | 2/2 | 75.59 | 0.36 | 41.36 | 0.21 | 0.082 | 41.28 | 0.195 | 44.71 | QUALIFIED (>= 75%) |
| max-autotune-no-cg | `futurebiohackers {"compile_mode": "max-autotune-no-cudagraphs", "use_compile": true}` | 2/2 | 75.50 | 0.10 | 41.51 | 0.22 | 0.081 | 41.43 | 0.195 | 115.57 | QUALIFIED (>= 75%) |
