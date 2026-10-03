# Development

- This is a fast-paced hackathon repo. Keep changes focused and scaffolding minimal.
- Use uv and `pyproject.toml` for dependencies: `uv add`, `uv sync`, and `uv run`.
- Keep `uv.lock` in Git. Use Python 3.11 for the local environment.
- Put runnable experiments in `scripts/`, exploration in `notebooks/`, local inputs in `data/`, and generated results in `artifacts/`.
- Load relevant science skills from `.agents/skills/` or `.claude/skills/` when useful.
- Preserve the bundled third-party skills and their licenses; keep project code separate.
