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
| 2026-10-03 | modal-env-check | `::env_check` (no training; Modal profile/workspace `comerossary`) | 0 | n/a | n/a | n/a | A100-80GB requested, NOT granted | Modal auth OK. uv image built from the organizer lock: 9 layers, 3 min 10 s wall end to end (CUDA base 81 s, apt 21 s, torch sync 55 s); torch 2.4.0+cu124 / torchvision 0.19.0+cu124 / CUDA 12.4 verified at build time. GPU function refused with "Please add a payment method to use A100-80GB GPU functions"; exit 1. No GPU info (name, power limit, CPUs) obtained. |
| 2026-10-03 | modal-download-data | `::download_data` (CPU only) | 0 | n/a | n/a | n/a | none requested | Refused with the same A100 payment-method error after 7 s (image cached): `modal run` validates every function in the app, so the CPU-only functions are blocked too until the workspace has a payment method/credits. Volume `cifar100-data` NOT filled. |
| 2026-10-03 | template-n1 | submission_template (no target) | 1 | not run | not run | not run | not run | Not attempted: blocked by the workspace billing error above. Rerun `::env_check`, `::download_data`, then this once the active Modal profile points at the workspace with the hackathon credits. |
