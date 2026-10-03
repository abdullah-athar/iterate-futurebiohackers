"""Variant-switch check for submissions/futurebiohackers. Not part of the submission.

1. The default parameters must give a trial that is bit-identical (weights and predictions)
   to a reference copy of the recipe (default: the committed HEAD version), so experimental
   switches never change the control; its step count and its silence on stderr too.
2. The vectorized crop must reproduce the masked crop exactly.
3. Every experimental switch must run end to end with finite predictions, and the switches
   with bookkeeping of their own must add up: the step schedule under data filtering, the
   non-finite loss counter, and the per-trial Muon state.

CPU, synthetic images, about 30 s. Run from the speedrun env with cwd cifar100-speedrun:
    uv run python ../scripts/check_variants.py [reference_submission_dir]
    scripts/wsl_speedrun.sh python ../scripts/check_variants.py          # Windows, via WSL
"""

from __future__ import annotations

import contextlib
import io
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmark.api import BuildContext
from benchmark.data import synthetic_split
from benchmark.worker import load_submission, seed_everything

TEAM_DIR = (
    Path(__file__).resolve().parents[1] / "cifar100-speedrun" / "submissions" / "futurebiohackers"
)
# The synthetic split has 64 images: batch 8 gives 8 steps per epoch, 16 in two epochs.
BASE = {"widths": [32, 64, 64], "epochs": 2.0, "batch_size": 8, "compile": ""}
# Epoch 0 trains on everything and scores every example; epoch 1 keeps the 32 hardest (4 steps).
FILTER = {"filter_start": 1, "filter_keep": 0.5}
VARIANTS = {
    "low_res26": {"low_res": 26, "low_res_epochs": 1},
    "low_res28": {"low_res": 28, "low_res_epochs": 1},
    # CPU never compiles; this exercises the per-resolution wrapper selection in _fit.
    "low_res_compile": {"low_res": 28, "low_res_epochs": 1, "low_res_compile": "reduce-overhead"},
    "crop_gather": {"crop_gather": True},
    "jitter": {"jitter": 0.2},
    "silu": {"activation": "silu"},
    "relu": {"activation": "relu"},
    "whiten_svd": {"whiten_svd": True},
    "depths324": {"depths": [3, 2, 4]},
    "depths332": {"depths": [3, 3, 2]},
    "count_nonfinite": {"count_nonfinite": True},
    "filter": FILTER,
    "muon": {"optimizer": "muon"},
    "low_res_filter": {"low_res": 28, "low_res_epochs": 1, **FILTER},
    "muon_filter_nonfinite": {"optimizer": "muon", "count_nonfinite": True, **FILTER},
}
FILTERED = ("filter", "low_res_filter", "muon_filter_nonfinite")
results: list[bool] = []


def check(name: str, ok: bool) -> None:
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)


def trial(module, params, seed=7):
    data = synthetic_split(train=True)
    state = module.build(BuildContext(torch.device("cpu"), params))
    seed_everything(seed)
    module.prepare(state, data, seed)
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        model = module.train(state)
    model.eval()
    with torch.inference_mode():
        out = model(synthetic_split(train=False).images.float().div(255))
    if not torch.isfinite(out).all():
        raise RuntimeError("non-finite predictions")
    weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return SimpleNamespace(weights=weights, out=out, state=state, stderr=stderr.getvalue())


def reference_dir() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()
    folder = Path(tempfile.mkdtemp(prefix="submission-head-"))
    source = subprocess.run(
        ["git", "show", "HEAD:cifar100-speedrun/submissions/futurebiohackers/submission.py"],
        cwd=TEAM_DIR,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    (folder / "submission.py").write_text(source, encoding="utf-8")
    return folder


def main() -> int:
    torch.set_num_threads(4)
    reference = load_submission(reference_dir())
    new = load_submission(TEAM_DIR)
    a = trial(reference, BASE)
    b = trial(new, BASE)
    same = (
        set(a.weights) == set(b.weights)
        and all(torch.equal(a.weights[k], b.weights[k]) for k in a.weights)
        and torch.equal(a.out, b.out)
    )
    check("control path bit-identical to the reference recipe", same)
    count = len(synthetic_split(train=True).labels)
    steps_per_epoch = count // BASE["batch_size"]
    control_steps = math.ceil(BASE["epochs"] * steps_per_epoch)
    check(
        f"control path: total_steps unchanged ({control_steps}) and equal to the steps taken",
        a.state.total_steps == control_steps == b.state.total_steps == b.state.steps_taken,
    )
    check("control path prints no NONFINITE_LOSSES line", "NONFINITE_LOSSES" not in b.stderr)

    images = torch.randn(64, 3, 36, 36)
    torch.manual_seed(1)
    masked = new.batch_crop(images, 32)
    torch.manual_seed(1)
    gathered = new.batch_crop_gather(images, 32)
    check("batch_crop_gather == batch_crop (same draws, same crops)", torch.equal(masked, gathered))

    runs = {}
    for name, delta in VARIANTS.items():
        t0 = time.perf_counter()
        try:
            runs[name] = trial(new, {**BASE, **delta})
            check(f"variant {name} runs ({time.perf_counter() - t0:.1f} s)", True)
        except Exception as exc:  # noqa: BLE001
            check(f"variant {name}: {type(exc).__name__}: {exc}", False)

    for name in ("count_nonfinite", "muon_filter_nonfinite"):
        if name in runs:
            run = runs[name]
            check(
                f"{name}: train printed 'NONFINITE_LOSSES 0' to stderr and stored 0",
                "NONFINITE_LOSSES 0" in run.stderr.splitlines() and run.state.nonfinite_losses == 0,
            )
    filtered_steps = steps_per_epoch + round(FILTER["filter_keep"] * count) // BASE["batch_size"]
    for name in FILTERED:
        if name in runs:
            state = runs[name].state
            check(
                f"{name}: total_steps from prepare ({state.total_steps}) == steps taken "
                f"({state.steps_taken}) == {filtered_steps}, and every example was scored",
                state.total_steps == state.steps_taken == filtered_steps > 0
                and bool(torch.isfinite(state.scores).all()),
            )
    if "muon" in runs:
        state = runs["muon"].state
        filters = [p for p in state.net.parameters() if p.ndim == 4 and p.requires_grad]
        muon_before = state.muon
        populated = state.muon is not None and len(state.muon.state) == len(filters) > 0
        new.prepare(state, synthetic_split(train=True), 8)
        check(
            "muon: one momentum buffer per conv filter after train; prepare re-creates it empty",
            populated and state.muon is not muon_before and len(state.muon.state) == 0,
        )
    print(f"\n{sum(results)}/{len(results)} checks passed", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
