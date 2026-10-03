"""Variant-switch check for submissions/futurebiohackers. Not part of the submission.

1. The default parameters must give a trial that is bit-identical (weights and predictions)
   to a reference copy of the recipe (default: the committed HEAD version), so experimental
   switches never change the control.
2. The vectorized crop must reproduce the masked crop exactly.
3. Every experimental switch must run end to end with finite predictions.

CPU, synthetic images, about 30 s. Run from the speedrun env with cwd cifar100-speedrun:
    uv run python ../scripts/check_variants.py [reference_submission_dir]
    scripts/wsl_speedrun.sh python ../scripts/check_variants.py          # Windows, via WSL
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from benchmark.api import BuildContext
from benchmark.data import synthetic_split
from benchmark.worker import load_submission, seed_everything

TEAM_DIR = (
    Path(__file__).resolve().parents[1] / "cifar100-speedrun" / "submissions" / "futurebiohackers"
)
BASE = {"widths": [32, 64, 64], "epochs": 2.0, "batch_size": 8, "compile": ""}
VARIANTS = {
    "low_res26": {"low_res": 26, "low_res_epochs": 1},
    "low_res28": {"low_res": 28, "low_res_epochs": 1},
    "crop_gather": {"crop_gather": True},
    "jitter": {"jitter": 0.2},
    "silu": {"activation": "silu"},
    "whiten_svd": {"whiten_svd": True},
    "depths324": {"depths": [3, 2, 4]},
    "depths332": {"depths": [3, 3, 2]},
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
    model = module.train(state)
    model.eval()
    with torch.inference_mode():
        out = model(synthetic_split(train=False).images.float().div(255))
    if not torch.isfinite(out).all():
        raise RuntimeError("non-finite predictions")
    return {k: v.detach().clone() for k, v in model.state_dict().items()}, out


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
    a, out_a = trial(reference, BASE)
    b, out_b = trial(new, BASE)
    same = (
        set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a) and torch.equal(out_a, out_b)
    )
    check("control path bit-identical to the reference recipe", same)
    images = torch.randn(64, 3, 36, 36)
    torch.manual_seed(1)
    masked = new.batch_crop(images, 32)
    torch.manual_seed(1)
    gathered = new.batch_crop_gather(images, 32)
    check("batch_crop_gather == batch_crop (same draws, same crops)", torch.equal(masked, gathered))
    for name, delta in VARIANTS.items():
        t0 = time.perf_counter()
        try:
            trial(new, {**BASE, **delta})
            check(f"variant {name} runs ({time.perf_counter() - t0:.1f} s)", True)
        except Exception as exc:  # noqa: BLE001
            check(f"variant {name}: {type(exc).__name__}: {exc}", False)
    print(f"\n{sum(results)}/{len(results)} checks passed", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
