# A/B ab-noise

2026-10-03 14:34, NVIDIA A100 80GB PCIe (power limit 300.00 W, task ta-01M40ZD1QA65WQ0N0GHGS8PR7R), one container, sequential, 1 trial(s) per variant, container wall 151.5 s. PCIe guard attempts: ['NVIDIA A100-SXM4-80GB [task ta-01M40ZCQSYP6JN48NQERA56ZCR]', 'NVIDIA A100 80GB PCIe [task ta-01M40ZD1QA65WQ0N0GHGS8PR7R]'].

| variant | config | n | mean acc % | std | mean prep+train s | std | prepare s | train s | eval s | build s | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `futurebiohackers` | 1/1 | 75.55 | n/a | 54.70 | n/a | 0.149 | 54.55 | 0.225 | 3.75 | QUALIFIED (>= 75%) |
| control-again | `futurebiohackers` | 1/1 | 75.66 | n/a | 55.73 | n/a | 0.148 | 55.58 | 0.218 | 3.52 | QUALIFIED (>= 75%) |
