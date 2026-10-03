# A/B ab-epochs-a

2026-10-03 14:56, NVIDIA A100 80GB PCIe (power limit 300.00 W, task ta-01M410E0ABWQ217CHZTSX0CA8R), one container, sequential, 2 trial(s) per variant, container wall 402.3 s. PCIe guard attempts: ['NVIDIA A100 80GB PCIe [task ta-01M410E0ABWQ217CHZTSX0CA8R]'].

| variant | config | n | mean acc % | std | mean prep+train s | std | prepare s | train s | eval s | build s | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `futurebiohackers {"use_compile": true}` | 2/2 | 75.33 | 0.16 | 40.77 | 0.16 | 0.082 | 40.69 | 0.192 | 46.06 | QUALIFIED (>= 75%) |
| e36 | `futurebiohackers {"epochs": 36, "use_compile": true}` | 2/2 | 75.39 | 0.36 | 37.55 | 0.68 | 0.087 | 37.46 | 0.187 | 42.48 | QUALIFIED (>= 75%) |
| e32 | `futurebiohackers {"epochs": 32, "use_compile": true}` | 2/2 | 75.26 | 0.16 | 33.00 | 0.08 | 0.084 | 32.92 | 0.192 | 42.51 | QUALIFIED (>= 75%) |
