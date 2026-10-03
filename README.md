# iterate-futurebiohackers

Project for the [Iterate AI x Science Hackathon](https://iterate.inc/london-ai-science), London, 3–4 October 2026.

Uses Python 3.11 and [uv](https://docs.astral.sh/uv/) for dependency management.

```sh
uv sync --locked
uv add <package>
uv add --dev <package>
uv run python <script.py>
```

Declare dependencies in `pyproject.toml` and commit `uv.lock` to keep environments reproducible.

Working folders: `scripts/` for experiments, `notebooks/` for exploration, `data/` for local inputs, and `artifacts/` for generated results. Data and artifacts are ignored by Git. This lightweight layout draws on [enformer-pytorch](https://github.com/lucidrains/enformer-pytorch).

## CIFAR-100 speedrun

We are competing in the [CIFAR-100 speedrun](https://github.com/AIDDA-Institute/CIFAR-100-speedrun): reach 75% mean test accuracy in the lowest preparation + training time on an A100. The organizer repo lives in `cifar100-speedrun/` as a git subtree; read its `README.md` and `RULES.md`. Our recipe is `cifar100-speedrun/submissions/futurebiohackers/submission.py`.

Our recipe adapts [airbench](https://github.com/KellerJordan/cifar10-airbench) to CIFAR-100. Over 40 trials on a Modal A100-SXM4-80GB it averaged **75.29%** accuracy in **8.11 s** of prepare + train time. The organizer baseline is 75.36% in 59.30 s. The architecture, hyperparameters and results table are in `cifar100-speedrun/submissions/futurebiohackers/README.md`. The defaults in `submission.py` are the recipe we plan to submit. To experiment, override them with `--params`.

| Config | Accuracy | Time |
|---|---:|---:|
| Submitted defaults: 8.5 epochs, widths 128/384/576 | 75.48% | 8.11 s |
| `{"epochs": 9}` | 75.61% | 8.58 s |
| `{"epochs": 8, "widths": [128, 384, 768]}` | 75.62% | 8.40 s |

The table rows are 8 trials each, all run on the same GPU. Use the `{"epochs": 9}` setting if we need more accuracy margin.

Commands run through [just](https://just.systems) (`uv tool install rust-just`) from the repo root. On the GPU machine:

```sh
just setup          # install pinned torch 2.4.0 / CUDA 12.4 env
just data           # download CIFAR-100 once
just smoke          # CPU sanity check, no GPU or data needed
just run            # one real trial on the GPU
just run 10         # ten trials to check consistency
just diag 3 --params '{"epochs": 10}'   # ignore the 75% target, pass recipe settings
just last           # print the latest summary.json
just sync           # pull organizer updates
```

### Modal

No GPU machine needed: `just modal` runs the same harness on a [Modal](https://modal.com/apps/abdullahmuhammadathar786/main) A100-80GB from your laptop, using an image built from the harness's `uv.lock`. Log in once with `uv run modal token new`, then:

```sh
just modal                                  # one trial
just modal 3 --params '{"epochs": 10}'      # same flags as just run
just modal 40                               # full official-style run with the 75% target
just last                                   # results are copied back locally
```

Modal needs a payment method on the workspace before it will start any GPU function, even when credits cover the cost. Modal's `A100-80GB` is the SXM part (400 W), but official judging uses the A100 80GB PCIe (300 W), so official times will probably be slower than our Modal times. Set `MODAL_GPU` to pick another GPU type. An empty `MODAL_GPU` runs on CPU only.

CIFAR-100 is cached in the `cifar100-data` volume. Every run is also kept in the `cifar100-results` volume, so after a dropped connection you can fetch results with `uv run modal volume get --force cifar100-results futurebiohackers/ cifar100-speedrun/results/`. Script: `scripts/modal_speedrun.py`.

We have not submitted yet. To submit, fork the upstream repo and open a PR that adds only `submissions/futurebiohackers/`.

## Autoresearch optimizer

[`autoresearch-optimizer/`](autoresearch-optimizer/README.md) is the shared workspace for our autoresearch optimizer, alongside the CIFAR-100 speedrun. It has its own Python 3.11 environment and folders for experiments, notebooks, local data, and results.

```sh
just autoresearch-setup  # install the workspace environment
just autoresearch-smoke  # verify the Python environment
```

Start with the workspace [README](autoresearch-optimizer/README.md) and claim a workstream in [TASKS.md](autoresearch-optimizer/TASKS.md). Use feature branches and reviewed PRs to collaborate.

## Science skills

Includes 40 [Google DeepMind science skills](https://github.com/google-deepmind/science-skills), from commit `68832757cbbf941c620b71df5756cf6e5cc287b0`.

The source bundle lives in `.claude/science-skills/`; `.claude/skills/`, `.codex/skills/`, and `.agents/skills/` link to the same collection. Codex discovers the `.agents/skills/` link; Claude uses `.claude/skills/`.

Upstream licensing and data-source terms are preserved in `.claude/science-skills/LICENSE` and `.claude/science-skills/SKILL_LICENSES.md`.
