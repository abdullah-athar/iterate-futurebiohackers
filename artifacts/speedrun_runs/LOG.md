# CIFAR-100 speedrun run log

One row per run. `config` = submission folder (+ `--params` JSON, "(no target)" when `--no-accuracy-target`).
Accuracies in %, time = mean preparation + training time in seconds (what the judges rank).
`scripts/modal_speedrun.py::main` prints a ready-to-paste row after every run; the full
result folder is in `artifacts/speedrun_runs/<timestamp>_<tag>/`.

Reference (organizer README, A100 80GB PCIe, Docker, 2 trials): ResNet9-style baseline,
40 epochs, width 64: 75.36% mean accuracy, 59.30 s mean prep+train. That recipe is not
in the public repo (organizers keep calibration recipes in a gitignored `.local/`).

| date | tag | config | n | mean acc | std | mean time (s) | GPU | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-03 | wsl-smoke-template | submission_template, --device cpu --synthetic (local WSL, torch 2.4.0+cu124 on CPU) | 2 | 0.00 (synthetic) | 0.00 | 0.05 | none (CPU, WSL Ubuntu-24.04) | harness check only: complete=true, qualified=null, exit 0. run_id 20261003T110701Z-f4d2b3da |
| 2026-10-03 | wsl-smoke-team | futurebiohackers (= template copy), --device cpu --synthetic (local WSL) | 2 | 0.00 (synthetic) | 0.00 | 0.03 | none (CPU, WSL Ubuntu-24.04) | harness check only: complete=true, qualified=null, exit 0. run_id 20261003T110707Z-fd974805 |
