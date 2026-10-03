"""Hypothesis gate: deduplicate ideas before any agent writes code.

In hypothesis-first swarms each agent's idea is first stated as one line by a single tool-less
call over its STATUS.md (`ask_claude`). This gate compares those lines with each other (in worker
order) and with the hypotheses already in the ledger; an idea that repeats an earlier one stops
there, so the expensive coding session (reading parents, editing, running ./try) is only paid
for distinct ideas. The novelty gate on code still runs afterwards.

Two judges: `claude_judge` asks a small model whether two lines describe the same algorithmic
change (semantic); `lexical_judge` compares content words (free, deterministic) and is the
fallback when the model call fails.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass

_STOP = set("""a an and are as at be by for from in into is it its of on or so than that the then this to
using use with without via per each all more less better lower higher new only same also when where which
while instead replace add keep make so we our their them they""".split())


@dataclass
class GateVerdict:
    keep: bool
    similar_to: str = ""   # "#<ledger id>" or "w<worker>" of the earlier idea
    reason: str = ""

    def to_dict(self) -> dict:
        return {"keep": self.keep, "similar_to": self.similar_to, "reason": self.reason}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in _STOP}


def lexical_similarity(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def lexical_judge(threshold: float = 0.5):
    """Reject a hypothesis whose content-word Jaccard similarity to an earlier one reaches `threshold`."""

    def judge(new: list[tuple[str, str]], prior: list[tuple[str, str]]) -> tuple[list[GateVerdict], dict]:
        seen = list(prior)
        out = []
        for label, text in new:
            sim, near = max(((lexical_similarity(text, t), l) for l, t in seen), default=(0.0, ""))
            if sim >= threshold:
                out.append(GateVerdict(False, near, f"word overlap {sim:.2f}"))
            else:
                out.append(GateVerdict(True))
                seen.append((label, text))
        return out, {}

    return judge


JUDGE_PROMPT = """You deduplicate research ideas in an automated search for median-string solvers.

ALREADY TRIED (research ledger):
{prior}

NEW PROPOSALS (in this order):
{new}

A new proposal is a DUPLICATE if it would implement essentially the same algorithmic change as an
already-tried idea, or as an EARLIER new proposal in the list that you did not mark duplicate
(same technique applied the same way; wording does not matter). A different technique, or the same
technique applied to a clearly different part of the solver, is NOT a duplicate. When unsure, keep it.

Answer with JSON only, one decision per new proposal, in order:
{{"decisions": [{{"id": "w00", "duplicate_of": null, "reason": "short"}}]}}
where duplicate_of is null or the id ("#12" or "w01") of the earlier idea."""


def _parse_decisions(text: str, labels: list[str]) -> list[GateVerdict]:
    data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    by_id = {str(d.get("id")): d for d in data["decisions"]}
    out, kept = [], set()
    for label in labels:
        d = by_id[label]
        dup = d.get("duplicate_of")
        # a duplicate must point at the ledger or at an earlier kept proposal, never at itself or a later one
        if dup and (str(dup).startswith("#") or dup in kept):
            out.append(GateVerdict(False, str(dup), str(d.get("reason", ""))[:200]))
        else:
            out.append(GateVerdict(True))
            kept.add(label)
    return out


def ask_claude(prompt: str, system: str, model: str, timeout_s: int) -> tuple[str, dict]:
    """One tool-less headless Claude Code call -> (reply text, usage).

    No tools and a short system prompt: the call only reads what is in `prompt`, so it skips the
    coding-agent instructions that otherwise make up most of the input tokens (~30k -> a few k)."""
    cmd = ["claude", "-p", prompt, "--output-format", "json", "--model", model, "--max-turns", "1",
           "--tools", "", "--system-prompt", system]
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="autoresearch-ask-") as cwd:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout_s, stdin=subprocess.DEVNULL)
    res = json.loads(proc.stdout)
    u = res.get("usage", {})
    usage = {"model": model, "seconds": round(time.time() - t0, 1), "cost_usd": res.get("total_cost_usd", 0.0) or 0.0,
             "prompt_tokens": u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
             + u.get("cache_creation_input_tokens", 0),
             "completion_tokens": u.get("output_tokens", 0)}
    if res.get("is_error"):
        raise ValueError(f"claude error: {res.get('subtype') or res.get('result', '')[:200]}")
    return res.get("result", ""), usage


def claude_judge(model: str = "haiku", timeout_s: int = 120, fallback=None):
    """One tool-less Claude Code call per generation; falls back to `lexical_judge()` on any failure."""
    fallback = fallback or lexical_judge()

    def judge(new: list[tuple[str, str]], prior: list[tuple[str, str]]) -> tuple[list[GateVerdict], dict]:
        if not new:
            return [], {}
        prompt = JUDGE_PROMPT.format(prior="\n".join(f"- {l}: {t}" for l, t in prior) or "- (none)",
                                     new="\n".join(f"- {l}: {t}" for l, t in new))
        usage: dict = {"judge": f"claude-code:{model}"}
        try:
            text, u = ask_claude(prompt, "You judge whether research ideas are duplicates. Reply with JSON only.",
                                 model, timeout_s)
            usage.update(cost_usd=u["cost_usd"], prompt_tokens=u["prompt_tokens"], completion_tokens=u["completion_tokens"])
            return _parse_decisions(text, [l for l, _ in new]), usage
        except (subprocess.SubprocessError, OSError, ValueError, KeyError, TypeError) as e:
            verdicts, _ = fallback(new, prior)
            usage["fallback"] = f"lexical ({type(e).__name__})"
            return verdicts, usage

    return judge
