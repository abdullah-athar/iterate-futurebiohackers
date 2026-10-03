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
| `::main --tag T --n N [--submission futurebiohackers or --submission-path P] [--params JSON] [--no-accuracy-target] [--seed S] [--build-timeout S] [--torch-logs recompiles]` | one `benchmark.run`; results in `artifacts/speedrun_runs/<ts>_<T>/` |
| `::ab --tag T --n N --variants '[{...}, {...}]' [--control-params JSON] [--labels a,b] [--submission-paths '["..."]']` | control + variants **sequentially in one container on one card**, each its own `benchmark.run` (fresh build, nothing shared); comparison table + `ab_summary.{md,json}` in `<ts>_<T>/NN_<label>/`. Variants are parameter deltas merged over the control. The default way to compare speed. |
| `::fetch --run-id ID` | copy `results/futurebiohackers/ID/` out of the `cifar100-results` Volume (no GPU), e.g. after a Modal timeout |
| `::sweep --tag T --n N --params-list '[{...}, {...}]'` | one A100 container per params entry, in parallel, no PCIe guard; gated behind `SPEEDRUN_ALLOW_SWEEP=1` |

- GPU is pinned to `A100-80GB` (plain `A100` can be a 40GB card), 4 CPUs. Every GPU function
  has a hard Modal timeout of 15 min (`SPEEDRUN_TIMEOUT_MIN`), so a hung run cannot burn more.
  The launcher estimates the container time from the specs and refuses launches that cannot
  fit; inside the container each variant has a deadline (the harness gets SIGINT and keeps
  its finished trials) and variants that no longer fit are skipped, so the payload always
  comes back. `--build-timeout` defaults to 300 s (the 5-minute rule for compile builds).
- PCIe guard (`--require-pcie`, default on for `main` and `ab`): Modal's `A100-80GB` pool mixes
  the judges' PCIe card (300 W) with SXM4 cards (500 W). The container checks nvidia-smi
  before build; on a non-PCIe card it returns at once and the launcher retries, up to 3
  retries, logging every attempt with the GPU name and Modal task id. GPU functions are
  single-use containers, so a retry is never served by the container that just failed (it
  can still land on the same host). `--no-require-pcie` disables the guard.
- Each `benchmark.run` in an A/B gets its own empty `TORCHINDUCTOR_CACHE_DIR`, so compiled
  variants report cold build times like the judges' container.
- GPU budget: `artifacts/speedrun_runs/gpu_ledger.jsonl` records every container attempt (wall
  time + 15 s start allowance, failed and timed-out calls included) and the launcher prints
  `GPU used: X/90 min` before and after every call. `SPEEDRUN_GPU_BUDGET_MIN` (90) is a hard
  stop; `SPEEDRUN_GPU_STOP_MIN` (60) is a checkpoint that needs `SPEEDRUN_GPU_CONTINUE=1`.
- The image is built once (uv 0.10.8 + `uv sync --frozen` on the organizer lock); the
  speedrun code is mounted at start, so recipe edits never rebuild it.
- Exit code 1 from `benchmark.run` means below target or incomplete, not a crash: read
  `summary.json` (`complete`, `qualified`, `run_error`) and `error.txt`.
- The launcher prints a warning if the GPU is not `NVIDIA A100 80GB PCIe`; an SXM card
  (power limit about 400 W instead of 300 W) makes timings slightly optimistic compared
  with the judges' machine.
- `SPEEDRUN_IMAGE=dockerfile` switches to `modal.Image.from_dockerfile` on the organizer
  Dockerfile (experimental; may be rejected by Modal's builder, and bakes the code in).
- First image build measured on 3 October 2026: about 3 min wall (CUDA base 81 s, apt 21 s,
  torch sync 55 s); later runs reuse the cached image and reach the function in a few seconds.
- Modal refuses `A100-80GB` functions until the workspace has a payment method on file
  ("Please add a payment method to use A100-80GB GPU functions"). Because `modal run`
  validates every function of the app, this also blocks `::download_data` and `::smoke`.
  Check `uv run modal profile list` points at the workspace holding the hackathon credits
  (`uv run modal token new` to add another workspace, `uv run modal profile activate NAME`),
  and its Billing page at modal.com/settings.
- Windows hosts: set `PYTHONUTF8=1` (`$env:PYTHONUTF8 = "1"`) before Modal commands, or the
  CLI can die with `'charmap' codec can't encode character` after an otherwise successful
  step (seen with `modal setup`, which had already written the token).

## Submitting upstream

The judges only take `submissions/futurebiohackers/`. Fork
https://github.com/AIDDA-Institute/CIFAR-100-speedrun, copy that folder in, open a PR
before 4 October 2:45pm. Pull organizer changes into our subtree with `just sync`.
