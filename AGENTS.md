# Development

- This is a fast-paced hackathon repo. Keep changes focused and scaffolding minimal.
- Use uv and `pyproject.toml` for dependencies: `uv add`, `uv sync`, and `uv run`.
- Keep `uv.lock` in Git. Use Python 3.11 for the local environment.
- Put runnable experiments in `scripts/`, exploration in `notebooks/`, local inputs in `data/`, and generated results in `artifacts/`.
- Load relevant science skills from `.agents/skills/` or `.claude/skills/` when useful.
- Preserve the bundled third-party skills and their licenses; keep project code separate.
- `cifar100-speedrun/` is the organizer's competition repo (git subtree, Python 3.12, its own uv env). Only edit `cifar100-speedrun/submissions/futurebiohackers/`; use `just` from the repo root to run it.
- `autoresearch-optimizer/` is our team's autoresearch workspace (Python 3.11, its own uv env). Run its commands with the `autoresearch-` prefix from the repo root; add its dependencies from inside that folder.
