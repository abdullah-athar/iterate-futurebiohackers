# CIFAR-100 speedrun: reproduce the setup

Two separate environments, never mixed:

| | Team env (repo root) | Speedrun env (`cifar100-speedrun/`) |
| --- | --- | --- |
| Python | 3.11 (`.python-version`) | 3.12.10 (its own `.python-version`) |
| Lock | `uv.lock`, `uv sync --locked` | organizer `uv.lock`, `uv sync --frozen`, never edited by us |
| Holds | `modal` (dev group) | torch 2.4.0, torchvision 0.19.0, numpy 1.26.4, pytest, ruff |
| Runs | `uv run modal run scripts/modal_speedrun.py::...` | `uv run python -m benchmark.run ...` |

Do not `uv add` anything inside `cifar100-speedrun/` and do not add torch at the root.
The GPU work happens on Modal (A100 80GB); nothing here needs a local GPU.

## From scratch (Linux, macOS, or WSL on Windows)

```sh
git clone https://github.com/abdullah-athar/iterate-futurebiohackers.git && cd iterate-futurebiohackers
curl -LsSf https://astral.sh/uv/install.sh | sh          # Windows PowerShell: irm https://astral.sh/uv/install.ps1 | iex
uv sync --locked                                          # team env (Python 3.11 + modal)
uv run modal setup                                        # once, opens a browser; writes ~/.modal.toml
(cd cifar100-speedrun && uv sync --frozen)                # speedrun env (Python 3.12.10 + torch 2.4.0)
(cd cifar100-speedrun && uv run python -m benchmark.run --submission futurebiohackers --device cpu --synthetic --n 2)
uv run modal run scripts/modal_speedrun.py::env_check     # A100 + versions, no training
uv run modal run scripts/modal_speedrun.py::download_data # once per Modal workspace: CIFAR-100 -> Volume cifar100-data
uv run modal run scripts/modal_speedrun.py::main --tag first --n 1 --no-accuracy-target
```

The CPU smoke test must end with `"complete": true` and `"qualified": null`
(synthetic images, accuracy meaningless). Each Modal run copies
`results/<team>/<run_id>/` to `artifacts/speedrun_runs/<timestamp>_<tag>/` and prints
a verdict plus a row for `artifacts/speedrun_runs/LOG.md`.

## Windows hosts: run the speedrun env inside WSL

The team env works natively on Windows. The speedrun env does not: torch 2.4.0's Windows
wheel fails at import (`fbgemm.dll` needs `libomp140.x86_64.dll`, which only ships with
Visual Studio). Use WSL (Ubuntu), which resolves the same lockfile to the Linux CUDA 12.4
wheels the judges use. Keep the venv on the Linux filesystem so it stays fast, is not
synced by cloud-sync tools, and cannot collide with a Windows `.venv` in the same folder:

```sh
wsl -d Ubuntu-24.04
curl -LsSf https://astral.sh/uv/install.sh | sh                 # once, inside WSL
cd /mnt/c/dev/iterate-futurebiohackers
scripts/wsl_speedrun.sh python -m benchmark.run --submission futurebiohackers --device cpu --synthetic --n 2
```

`scripts/wsl_speedrun.sh` sets `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/cifar100-speedrun`
and runs `uv run --frozen` inside `cifar100-speedrun/` (the first call creates the env,
about 2.5 GB of CUDA wheels). With that variable exported, `just smoke` and `just check`
also work from WSL.

## Modal launcher: `scripts/modal_speedrun.py`

| Command | What it does |
| --- | --- |
| `::env_check` | nvidia-smi (name, memory, power limit, MIG), torch/torchvision/CUDA versions, CPUs |
| `::download_data` | fills the `cifar100-data` Volume (mounted at `/data`) |
| `::smoke` | CPU + synthetic smoke test inside the Modal image, no GPU |
| `::main --tag T --n N [--submission futurebiohackers or --submission-path P] [--params JSON] [--no-accuracy-target] [--seed S]` | one `benchmark.run`; results in `artifacts/speedrun_runs/<ts>_<T>/` |
| `::sweep --tag T --n N --params-list '[{...}, {...}]' [--no-accuracy-target]` | one A100 container per params entry, in parallel; `<ts>_<T>_00/`, `_01/`, ... |

- GPU is pinned to `A100-80GB` (plain `A100` can be a 40GB card), 4 CPUs, 8 h timeout.
- The image is built once (uv 0.10.8 + `uv sync --frozen` on the organizer lock); the
  speedrun code is mounted at start, so recipe edits never rebuild it.
- Exit code 1 from `benchmark.run` means below target or incomplete, not a crash: read
  `summary.json` (`complete`, `qualified`, `run_error`) and `error.txt`.
- The launcher prints a warning if the GPU is not `NVIDIA A100 80GB PCIe`; an SXM card
  (power limit about 400 W instead of 300 W) makes timings slightly optimistic compared
  with the judges' machine.
- `SPEEDRUN_IMAGE=dockerfile` switches to `modal.Image.from_dockerfile` on the organizer
  Dockerfile (experimental; may be rejected by Modal's builder, and bakes the code in).

## Submitting upstream

The judges only take `submissions/futurebiohackers/`. Fork
https://github.com/AIDDA-Institute/CIFAR-100-speedrun, copy that folder in, open a PR
before 4 October 2:45pm. Pull organizer changes into our subtree with `just sync`.
