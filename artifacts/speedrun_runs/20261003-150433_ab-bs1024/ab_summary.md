# A/B ab-bs1024

2026-10-03 15:04, NVIDIA A100 80GB PCIe (power limit 300.00 W, task ta-01M410WJAQTX5DDV3XFK155KQR), one container, sequential, 2 trial(s) per variant, container wall 413.8 s. PCIe guard attempts: ['NVIDIA A100 80GB PCIe [task ta-01M410WJAQTX5DDV3XFK155KQR]'].

| variant | config | n | mean acc % | std | mean prep+train s | std | prepare s | train s | eval s | build s | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | `futurebiohackers {"use_compile": true}` | 2/2 | 75.92 | 0.04 | 41.52 | 0.28 | 0.080 | 41.44 | 0.190 | 43.76 | QUALIFIED (>= 75%) |
| bs1024-lr0.8-linear | `futurebiohackers {"batch_size": 1024, "lr": 0.8, "use_compile": true}` | 2/2 | 74.75 | 0.37 | 38.59 | 0.16 | 0.083 | 38.50 | 0.203 | 41.92 | BELOW 75% |
| bs1024-lr0.566-sqrt | `futurebiohackers {"batch_size": 1024, "lr": 0.566, "use_compile": true}` | 2/2 | 74.81 | 0.01 | 38.75 | 0.05 | 0.082 | 38.67 | 0.191 | 42.93 | BELOW 75% |
