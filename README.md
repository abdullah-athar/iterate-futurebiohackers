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

## Science skills

Includes 40 [Google DeepMind science skills](https://github.com/google-deepmind/science-skills), from commit `68832757cbbf941c620b71df5756cf6e5cc287b0`.

The source bundle lives in `.claude/science-skills/`; `.claude/skills/`, `.codex/skills/`, and `.agents/skills/` link to the same collection. Codex discovers the `.agents/skills/` link; Claude uses `.claude/skills/`.

Upstream licensing and data-source terms are preserved in `.claude/science-skills/LICENSE` and `.claude/science-skills/SKILL_LICENSES.md`.
