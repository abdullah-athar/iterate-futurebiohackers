"""Proposers: produce the next candidate solver from the research context."""

from __future__ import annotations

import random
import re
from pathlib import Path

from .core import Candidate, ResearchContext
from .prompts import SYSTEM_PROMPT, build_prompt

TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "median_string" / "solvers" / "template_solver.py"


def seed_candidate() -> Candidate:
    """The starting point: the bundled template solver, rewritten to use absolute imports."""
    code = TEMPLATE_PATH.read_text().replace("from ..", "from median_string.")
    return Candidate(code=code, rationale="Seed: template greedy local search", id="seed")


def extract_code(text: str) -> str | None:
    blocks = re.findall(r"```python\s*\n(.*?)```", text, flags=re.DOTALL)
    return max(blocks, key=len) if blocks else None


class ClaudeProposer:
    """Asks Claude to rewrite the champion solver, given its scores and recent history."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "high", timeout_s: float = 60.0) -> None:
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.system = SYSTEM_PROMPT.format(timeout_s=timeout_s)

    def propose(self, ctx: ResearchContext) -> Candidate:
        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=64000,
            system=self.system,
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": build_prompt(ctx)}],
        ) as stream:
            message = stream.get_final_message()

        text = "".join(b.text for b in message.content if b.type == "text")
        code = extract_code(text)
        parent_id = ctx.parents[0].candidate.id if ctx.parents else None
        meta = {"model": message.model, "stop_reason": message.stop_reason, "usage": message.usage.to_dict()}
        if code is None:
            # An empty module fails evaluation cleanly and the failure is recorded in memory.
            code = f"# No code block in model reply (stop_reason={message.stop_reason})\n"
        rationale = text.split("```", 1)[0].strip()
        return Candidate(code=code, parent_id=parent_id, rationale=rationale, meta=meta)


class MockProposer:
    """Offline proposer: randomly retunes the template's max_iterations and seed. For tests and dry runs."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def propose(self, ctx: ResearchContext) -> Candidate:
        parent = ctx.parents[0].candidate
        iters, seed = self.rng.choice([10, 50, 200, 1000]), self.rng.randrange(1000)
        code = re.sub(r"max_iterations: int = \d+", f"max_iterations: int = {iters}", parent.code)
        code = re.sub(r"seed: int = \d+", f"seed: int = {seed}", code)
        return Candidate(code=code, parent_id=parent.id, rationale=f"max_iterations={iters}, seed={seed}")
