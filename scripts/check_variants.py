"""Variant-switch check for submissions/futurebiohackers. Not part of the submission.

1. With default parameters the recipe must give a trial that is bit-identical (weights and
   predictions) to the reference recipe: Abdullah's PR #5 file (origin/runtime-optimization),
   the base our switches are ported onto. So experimental switches never change the control,
   its step count, or its silence on stderr.
2. Every experimental switch (ours and his) must run end to end on CPU with finite predictions;
   the non-finite counter must report 0.

CPU, synthetic images, about 20 s. Run from the speedrun env with cwd cifar100-speedrun:
    uv run python ../scripts/check_variants.py [reference_submission_dir]
    scripts/wsl_speedrun.sh python ../scripts/check_variants.py          # Windows, via WSL
The reference defaults to `git show origin/runtime-optimization:...submission.py`; set
CHECK_VARIANTS_REF=<git ref> to compare against another commit.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
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
REFERENCE_REF = os.environ.get("CHECK_VARIANTS_REF", "origin/runtime-optimization")
# The synthetic split has 64 images: batch 8 gives 8 steps per epoch, 16 in two epochs.
BASE = {"widths": [32, 64, 64], "epochs": 2.0, "batch_size": 8, "compile": ""}
VARIANTS = {
    # ours: accuracy recovery and progressive resizing
    "jitter": {"jitter": 0.2},
    "count_nonfinite": {"count_nonfinite": True},
    "res24": {"train_resolution": 24, "resolution_switch": 0.5},
    "res28_late": {"train_resolution": 28, "resolution_switch": 0.75},
    "res24_reduce_overhead": {
        "train_resolution": 24,
        "resolution_switch": 0.5,
        "low_res_compile": "reduce-overhead",
    },
    "cutout_translate": {"cutout": 4, "translate": 1},
    "jitter_res24_nonfinite": {
        "jitter": 0.3,
        "train_resolution": 24,
        "resolution_switch": 0.5,
        "count_nonfinite": True,
    },
    # his (must keep running after the port)
    "indexed_crop": {"crop_mode": "indexed"},
    "depths233": {"depths": [2, 3, 3]},
    "fused_sgd_cpu": {"fused_sgd": True},
    "compile_loss_cpu": {"compile_loss": True},
    "pool_first": {"pool_first": [False, False, True]},
    "gelu_tanh": {"gelu_approximate": "tanh"},
}
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
    folder = Path(tempfile.mkdtemp(prefix="submission-ref-"))
    for name in ("submission.py", "kernels.py"):
        source = subprocess.run(
            [
                "git",
                "show",
                f"{REFERENCE_REF}:cifar100-speedrun/submissions/futurebiohackers/{name}",
            ],
            cwd=TEAM_DIR,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        (folder / name).write_text(source, encoding="utf-8")
    return folder


def main() -> int:
    torch.set_num_threads(4)
    reference = load_submission(reference_dir())
    new = load_submission(TEAM_DIR)
    # Our DEFAULTS may differ from the reference's (promoted values) and add switches the
    # reference does not know (jitter, low_res_compile, count_nonfinite). The faithfulness
    # test: our file, with every shared parameter set to OUR default and our own switches
    # turned off, must match the reference file given the same shared parameters.
    shared = {
        k: v
        for k, v in new.DEFAULTS.items()
        if k in reference.DEFAULTS and reference.DEFAULTS[k] != v and k not in BASE
    }
    ours_only = {k: v for k, v in new.DEFAULTS.items() if k not in reference.DEFAULTS}
    off = {"jitter": 0.0, "count_nonfinite": False}
    if shared:
        print(f"promoted defaults vs {REFERENCE_REF}: {shared}", flush=True)
    if ours_only:
        print(f"switches only in our file: {ours_only}", flush=True)
    a = trial(reference, {**BASE, **shared})
    b = trial(new, {**BASE, **shared, **off})
    same = (
        set(a.weights) == set(b.weights)
        and all(torch.equal(a.weights[k], b.weights[k]) for k in a.weights)
        and torch.equal(a.out, b.out)
    )
    check(f"shared-parameter path bit-identical to the reference recipe ({REFERENCE_REF})", same)
    count = len(synthetic_split(train=True).labels)
    control_steps = math.ceil(BASE["epochs"] * (count // BASE["batch_size"]))
    check(
        f"shared-parameter path: total_steps unchanged ({control_steps})",
        a.state.total_steps == control_steps == b.state.total_steps,
    )
    check("shared-parameter path prints nothing on stderr", b.stderr.strip() == "")
    d = trial(new, BASE)  # our real defaults (plus the small BASE overrides)
    check("our defaults run with finite predictions and nothing on stderr", d.stderr.strip() == "")
    if new.DEFAULTS.get("jitter"):
        check(
            "our defaults differ from the shared path only through jitter (own generator)",
            d.state.aug_generator is not None and not torch.equal(d.out, b.out),
        )

    runs = {}
    for name, delta in VARIANTS.items():
        t0 = time.perf_counter()
        try:
            runs[name] = trial(new, {**BASE, **delta})
            check(f"variant {name} runs ({time.perf_counter() - t0:.1f} s)", True)
        except Exception as exc:  # noqa: BLE001
            check(f"variant {name}: {type(exc).__name__}: {exc}", False)

    for name in ("count_nonfinite", "jitter_res24_nonfinite"):
        if name in runs:
            run = runs[name]
            check(
                f"{name}: train printed 'NONFINITE_LOSSES 0' to stderr and stored 0",
                "NONFINITE_LOSSES 0" in run.stderr.splitlines() and run.state.nonfinite_losses == 0,
            )
    if "jitter" in runs:
        state = runs["jitter"].state
        check(
            "jitter: per-trial generator exists and is re-seeded by prepare",
            state.aug_generator is not None and state.aug_generator.initial_seed() == 7,
        )
    print(f"\n{sum(results)}/{len(results)} checks passed", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
