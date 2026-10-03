# Accuracy check on a free GPU (Kaggle or Colab)

Purpose: run the exact `submissions/futurebiohackers/` recipe, with the organizers' pinned
versions (Python 3.12.10, torch 2.4.0+cu124, torchvision 0.19.0+cu124), on a free GPU to
see whether it reaches the 75% mean accuracy. **Timings from these GPUs say nothing about
the A100 80GB PCIe used for judging**: a T4 is roughly 8 to 10 times slower and takes the
fp16 + GradScaler path (no bf16 on Turing/Pascal), a P100 has no tensor cores at all.
Only read `mean_accuracy` from these runs.

Nothing below has been run yet; it is written from the organizer README and the launcher
we use on Modal.

## 0. What you need

- Kaggle: a notebook with *Accelerator = GPU T4 x2* (or P100) and *Internet = On*
  (Settings panel on the right). Free quota is about 30 GPU hours per week.
- Colab: *Runtime > Change runtime type > T4 GPU*. Free sessions can be cut after an idle
  period; Ctrl-C or a disconnect keeps the partial `results/` (see step 4).
- Both machines are Linux x86_64 with an NVIDIA driver recent enough for the cu124 wheels
  (driver 550 or newer). `nvidia-smi` prints the driver version.

The notebook's own Python and PyTorch are irrelevant: `uv` installs Python 3.12.10 and the
locked packages in a private virtual environment.

## 1. Get the code onto the machine

Either clone the team repository (needs a GitHub token if the repository is private):

```bash
git clone https://github.com/abdullah-athar/iterate-futurebiohackers.git
cd iterate-futurebiohackers/cifar100-speedrun
```

or upload a zip of `cifar100-speedrun/` (without `.venv/`, `data/`, `results/`) through the
notebook's file panel and `unzip` it, then `cd cifar100-speedrun`.

In a notebook cell, prefix shell commands with `!`, and use `%cd` to change directory.

## 2. Install the pinned environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"       # in a notebook: use ~/.local/bin/uv explicitly
uv python install 3.12.10
uv sync --frozen                            # creates .venv from the organizer uv.lock (~2.5 GB)
uv run python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

Expected: `2.4.0+cu124 Tesla T4` (or `Tesla P100-PCIE-16GB`). Any other torch version means
the wrong interpreter is being used; always go through `uv run`.

## 3. Sanity check, then data, then the real run

```bash
uv run python -m benchmark.run --submission futurebiohackers --device cpu --synthetic --n 2
uv run python -m benchmark.data --root data
uv run python -m benchmark.run --submission futurebiohackers --n 2 --no-accuracy-target --eval-timeout 60
```

- The synthetic run must end with `"complete": true`, `"qualified": null`.
- The real run uses the recipe defaults (40 epochs, width 64). On a T4 expect roughly
  5 to 8 minutes per trial; on a P100 up to about 10. `--n 2` matches the organizers'
  pilot; raise it if the quota allows.
- `--no-accuracy-target` only changes the exit code and `qualified`; accuracy is reported
  either way. `--eval-timeout 60` relaxes the 5 s inference limit, which a T4 may miss in
  fp32; the A100 does not need it.
- The build log line `[futurebiohackers] device=cuda precision=fp16 ...` confirms the
  automatic fp16 path on these GPUs. Pass `--params '{"precision": "fp32"}'` to rule out
  mixed precision as a cause if accuracy looks wrong (about 3x slower on a T4).

## 4. Read and bring back the result

```bash
cat results/futurebiohackers/*/summary.json
```

`mean_accuracy` is a fraction (0.7536 = 75.36%). Download `summary.json`, `trials.jsonl`
and `config.json` from the newest `results/futurebiohackers/<run_id>/` folder (Colab: file
panel; Kaggle: `/kaggle/working` output) and add a row to
`artifacts/speedrun_runs/LOG.md` with the GPU name in the GPU column and "accuracy only,
free GPU" in the notes. Do not commit `results/`, `data/` or `.venv/`.

## Interpreting the number

- The A100 run uses bf16; the T4/P100 run uses fp16 with loss scaling. Expect the mean
  accuracy to agree within roughly one standard deviation of the trial spread, not exactly.
- Two trials have a spread of about 0.3 to 0.5 percentage points on this recipe
  (organizer pilot: 75.56% and 75.16%). A mean of 74.5% here does not rule the recipe out;
  anything under about 73% means the recipe, not the hardware, needs work.
- CUDA kernels are nondeterministic, so identical seeds will not reproduce bitwise.
