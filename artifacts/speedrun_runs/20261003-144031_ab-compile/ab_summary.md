# A/B ab-compile

2026-10-03 14:40, NVIDIA A100 80GB PCIe (power limit 300.00 W, task ta-01M40ZMP3SGZ9SPRJG0YMVT7ZR), one container, sequential, 2 trial(s) per variant, container wall 281.1 s. PCIe guard attempts: ['NVIDIA A100-SXM4-80GB [task ta-01M40ZKEABYKKX8VDZVWYN1JER]', 'NVIDIA A100 80GB PCIe [task ta-01M40ZMP3SGZ9SPRJG0YMVT7ZR]'].

| variant | config | n | mean acc % | std | mean prep+train s | std | prepare s | train s | eval s | build s | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `futurebiohackers` | 2/2 | 75.81 | 0.16 | 56.11 | 0.31 | 0.096 | 56.02 | 0.200 | 4.12 | QUALIFIED (>= 75%) |
| compile-default | `futurebiohackers {"use_compile": true}` | 2/2 | 75.92 | 0.04 | 41.78 | 0.00 | 0.082 | 41.70 | 0.193 | 44.97 | QUALIFIED (>= 75%) |
