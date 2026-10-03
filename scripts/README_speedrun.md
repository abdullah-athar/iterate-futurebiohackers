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
uv run modal run scripts/modal_speedrun.py::env_check     # GPU name/power + placement, no training
uv run modal run scripts/modal_speedrun.py::main --tag first --n 1 --no-accuracy-target  # downloads CIFAR-100 into the Volume on first use
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
| `::env_check` | nvidia-smi (name, memory, power limit, MIG) and the container's task id, region and cloud; no training |
| `::main [--tag T] [--require-gpu sxm\|pcie\|any] [--require-power 400\|500\|any] [--warm] [--count-nonfinite] <harness flags>` | one `benchmark.run` with the harness flags passed straight through (`--n N`, `--params JSON`, `--no-accuracy-target`, `--seed S`, `--build-timeout S`, `--submission-path P`, ...; default `--n 1`, cold build). `just modal [n] [flags]` calls this. Results in `artifacts/speedrun_runs/<ts>_<T>/` and `cifar100-speedrun/results/` (for `just last`); a row goes to `registry.jsonl` and `LOG.md`. Use it for references and finalists (n=40, cold build). |
| `::ab --tag T --n N --variants '[{...}, {...}]' [--labels a,b] [--hypotheses "a;b"] [--control-params JSON] [--seed S] [--warm] [--profile]` | control + variants **sequentially in one container on one card**, each its own `benchmark.run`; table with mean acc, std, mean prepare+train, build time, GPU @ power limit, non-finite losses and the **paired** per-seed Δacc ± SE, Δtime and Δtime_adj vs the control, saved as `ab_summary.{md,json}`. Variants are parameter deltas merged over the control. |
| `::screen --jobs FILE [--parallel 10] [--only a,b] [--no-warm] [--allow-big]` | a whole round from a jobs file (`artifacts/speedrun_runs/jobs/*.json`: `{"round", "n", "seed", "control", "jobs": [{"name", "variants": [{"label", "params", "hypothesis"}]}]}`); one container per job (control + up to 3 variants), up to `--parallel` containers at once, warm compile cache, every variant appended to `registry.jsonl` and `LOG.md`, then `scripts/leaderboard.py` regenerates `leaderboard.md` and `pareto.svg`. |
| `::profile [--params JSON] [--epochs E]` | runs `scripts/profile_recipe.py` on the real data: prepare breakdown, per-step forward/backward/optimizer time, per-group compute split, top kernels |

- CIFAR-100 is downloaded into the `cifar100-data` Volume on first use; `TEAM` picks the
  submission folder and `MODAL_GPU` the GPU type (default `A100-80GB`; plain `A100` can be a
  40GB card). 4 CPUs per container, like the judges.
- Every GPU function has a hard Modal timeout of `MODAL_TIMEOUT_MIN` minutes (default 20),
  so a hung run cannot burn more. Inside the container each run has a deadline (the harness
  gets SIGINT and keeps its finished trials) and runs that no longer fit are skipped, so the
  payload always comes back. Before launching, every job's GPU time is estimated from its
  specs and jobs above `MODAL_JOB_LIMIT_MIN` (20) are refused unless `--allow-big` is given.
- GPU guard (default: require an A100-SXM4-80GB, any power limit): Modal's `A100-80GB` pool
  mixes SXM4 cards at 400 W and 500 W with PCIe cards (300 W), and the 500 W cards are about
  7% faster. The container checks nvidia-smi before build; on the wrong card it returns at
  once (about 0.3 GPU-min) and the launcher retries, up to `MODAL_GPU_ATTEMPTS` calls
  (default 11 = one try + 10 retries), logging every attempt with the GPU name, power limit,
  Modal task id, region and cloud. GPU functions are single-use containers, so a retry is
  never served by the container that just failed. There is no fallback: a job that never
  gets the card fails. `--require-power 400` pins the power limit (references, finalists).
  Timing comparisons are only valid within one container (same card, same power limit).
- Compile caches: `::main` builds cold (an empty `TORCHINDUCTOR_CACHE_DIR` per run, like the
  judges' container; builds over 300 s are flagged). `::screen` (and `--warm`) copies the
  persistent cache Volume `cifar100-inductor-cache` into the container, enables the inductor
  FX graph cache, and merges new files back at the end, so only new graphs pay a compile.
- Non-finite losses: with `count_nonfinite` the recipe prints `NONFINITE_LOSSES n` at the end
  of every trial; the launcher sums them per run (`nonfinite` column; any value above 0
  disqualifies a variant). `::screen` and `::ab` turn it on by default.
- Exchange rate: `artifacts/speedrun_runs/k.json` (`{"k": pp per second}`) comes from the
  epoch calibration (control at 8.5 / 8.0 / 7.5 epochs) and converts accuracy into time:
  `dtime_adj = dtime - dacc_pp / k`. The leaderboard recomputes it from the current k.
- GPU budget: `artifacts/speedrun_runs/gpu_ledger.jsonl` records every container attempt (wall
  time + 15 s start allowance, failed and timed-out calls included) and the launcher prints
  `GPU used: X/1263 min` before and after every call. `MODAL_GPU_BUDGET_MIN` (1263 = the 63
  minutes used when the 1200-minute cap was set, plus the cap) is a hard stop.
- The image is `debian_slim` + `uv_sync` on the organizer lock, built once; the speedrun code
  is mounted at start, so recipe edits never rebuild it.
- Exit code 1 from `benchmark.run` means below target or incomplete, not a crash: read
  `summary.json` (`complete`, `qualified`, `run_error`) and `error.txt`.
- The launcher prints a warning if the GPU is not `NVIDIA A100 80GB PCIe`; an SXM card
  (power limit about 400 W instead of 300 W) makes timings slightly optimistic compared
  with the judges' machine.
- Modal refuses `A100-80GB` functions until the workspace has a payment method on file
  ("Please add a payment method to use A100-80GB GPU functions"). Because `modal run`
  validates every function of the app, this blocks every entrypoint.
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
