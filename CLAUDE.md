AGENTS.md

# CIFAR-100 speedrun

## Layout and the two environments
- `cifar100-speedrun/` is the organizer repo, imported as a **git subtree**
  (upstream https://github.com/AIDDA-Institute/CIFAR-100-speedrun, split commit `25237e3`).
  Pull organizer updates with `just sync`. Never edit `benchmark/`, `tests/`,
  `pyproject.toml`, `uv.lock`, `Dockerfile` or `.python-version` in there.
- Our recipe: `cifar100-speedrun/submissions/futurebiohackers/`. Only that folder goes in
  the upstream PR (from a fork of the organizer repo). Everything else we add lives in the
  team repo: `scripts/modal_speedrun.py`, `scripts/wsl_speedrun.sh`, `artifacts/speedrun_runs/`.
- Team env (repo root): Python 3.11, `uv sync --locked`, holds `modal`. Runs `modal run`.
- Speedrun env (`cifar100-speedrun/`): Python 3.12.10, `uv sync --frozen`, torch 2.4.0 /
  torchvision 0.19.0 / numpy 1.26.4. No GPU locally. On Windows run it in WSL via
  `scripts/wsl_speedrun.sh` (torch 2.4.0's Windows wheel cannot load). Never mix the two
  envs: no `uv add` in the speedrun folder, no torch at the root.
- Reproduction guide and command reference: `scripts/README_speedrun.md`.

## Commands
- Local CPU smoke test (no GPU, no data; expect `"complete": true`, `"qualified": null`):
  `cd cifar100-speedrun && uv run python -m benchmark.run --submission futurebiohackers --device cpu --synthetic --n 2`
  (Windows: `scripts/wsl_speedrun.sh python -m benchmark.run --submission futurebiohackers --device cpu --synthetic --n 2`).
- Modal, from the repo root in the team env (`uv run modal setup` once):
  `uv run modal run scripts/modal_speedrun.py::env_check`,
  `::download_data` (once), `::smoke`,
  `::main --tag T --n N [--submission-path P] [--params '{"k": v}'] [--no-accuracy-target]`,
  `::sweep --tag T --n N --params-list '[{...}, {...}]'`.
- Results: `artifacts/speedrun_runs/<timestamp>_<tag>/` (summary.json, trials.jsonl,
  config.json, modal_run.json; `source/` is gitignored). Add a row to
  `artifacts/speedrun_runs/LOG.md` for every GPU run (the launcher prints the row).

## Non-negotiable rules (see cifar100-speedrun/RULES.md)
- Never modify `cifar100-speedrun/benchmark/` or `tests/`; organizers run their own copy.
- Train only on the CIFAR-100 training split: no pretrained weights, external data, or
  constants encoding learned tensors.
- No learned state carried between trials: reset params, buffers, optimizer, scheduler,
  scaler, EMA and custom RNGs in `prepare`. Reusing memory and compiled code is fine.
- No real data and no trial seed in module import or `build`; synthetic warmup only.
- No test-time augmentation, no fitting or state changes during evaluation; one view per
  image; the full test pass must finish within 5 s.
- The submission must run with its defaults (no `--params` needed) in the pinned
  PyTorch 2.4.0 env; no extra packages. The judges rank mean prepare + train time over
  40 trials, with mean accuracy >= 75%.

## Git workflow
- Never work on `main`: personal branches (e.g. `crossary`), PR into `main`.
- Small commits prefixed `speedrun:`. Never force-push. Never commit datasets, weights,
  checkpoints, `results/`, copied `source/` trees, or credentials (`~/.modal.toml`).
- Upstream submission: fork AIDDA-Institute/CIFAR-100-speedrun, add only
  `submissions/futurebiohackers/`, open the PR before 4 October 2:45pm.
