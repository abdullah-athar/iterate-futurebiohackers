"""Variant-switch check for submissions/futurebiohackers. Not part of the submission.

1. With default parameters the recipe must give a trial that is bit-identical (weights and
   predictions) to the reference recipe (default: the promoted file on origin/speedrun-accuracy-stack;
   the base our switches are ported onto. So experimental switches never change the control,
   its step count, or its silence on stderr. Switches the reference file does not know are
   turned off for that comparison; with the promoted file as reference (CHECK_VARIANTS_REF=
   origin/speedrun-accuracy-stack) it compares our defaults with the promoted defaults.
2. Every experimental switch (ours and his) must run end to end on CPU with finite predictions;
   the non-finite counter must report 0, filtered schedules take exactly total_steps steps
   with finite scores, and staged resizing keeps one downscaled copy per low resolution.

CPU, synthetic images, about 20 s. Run from the speedrun env with cwd cifar100-speedrun:
    uv run python ../scripts/check_variants.py [reference_submission_dir]
    scripts/wsl_speedrun.sh python ../scripts/check_variants.py          # Windows, via WSL
The reference defaults to `git show origin/speedrun-accuracy-stack:...submission.py`; set
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
REFERENCE_REF = os.environ.get("CHECK_VARIANTS_REF", "origin/speedrun-accuracy-stack")
# The synthetic split has 64 images: batch 8 gives 8 steps per epoch, 16 in two epochs.
BASE = {"widths": [32, 64, 64], "epochs": 2.0, "batch_size": 8, "compile": ""}
# Our switches with their "off" values; the control turns off those the reference file lacks.
OFF = {
    "jitter": 0.0,
    "count_nonfinite": False,
    "res_schedule": None,
    "filter_start": 0,
    "stem": "conv",
}
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
    # ours: staged resizing (replaces the train_resolution pair) and hard-example filtering
    "res_schedule_20_24": {"res_schedule": [[20, 0.25], [24, 0.5]]},
    "res_schedule_16_to_28": {"res_schedule": [[16, 0.2], [20, 0.4], [24, 0.6], [28, 0.8]]},
    "res_schedule_jitter": {"res_schedule": [[20, 0.3], [28, 0.6]], "jitter": 0.3},
    "res_schedule_indexed_crop": {"res_schedule": [[20, 0.5]], "crop_mode": "indexed"},
    # (the filter cannot run with the promoted compile_loss default, so it is turned off here)
    "filter": {"filter_start": 1, "filter_keep": 0.5, "compile_loss": False},
    "filter_nonfinite": {
        "filter_start": 1,
        "filter_keep": 0.5,
        "count_nonfinite": True,
        "compile_loss": False,
    },
    "filter_res_schedule": {
        "filter_start": 1,
        "filter_keep": 0.5,
        "res_schedule": [[20, 0.25], [24, 0.5]],
        "compile_loss": False,
    },
    # ours: stems that avoid group 1's 3x3 conv at 31x31 (whiten4s2 needs widths[0] == 96)
    "stem_space_to_depth": {"stem": "space_to_depth"},
    "stem_space_to_depth_res_schedule": {
        "stem": "space_to_depth",
        "res_schedule": [[20, 0.3], [24, 0.6]],
    },
    "stem_space_to_depth_nopool": {"stem": "space_to_depth_nopool"},
    "stem_whiten4s2": {"stem": "whiten4s2", "widths": [96, 64, 64]},
    "stem_whiten3s2": {"stem": "whiten3s2"},
    "stem_conv1x1": {"stem": "conv1x1"},
    "stem_conv1x1_depths223": {"stem": "conv1x1", "depths": [2, 2, 3]},
    # his (must keep running after the port)
    "indexed_crop": {"crop_mode": "indexed"},
    "depths233": {"depths": [2, 3, 3]},
    "fused_sgd_cpu": {"fused_sgd": True},
    "compile_loss_cpu": {"compile_loss": True},
    "pool_first": {"pool_first": [False, False, True]},
    "gelu_tanh": {"gelu_approximate": "tanh"},
    "depth2_residual": {"depths": [2, 2, 2], "depth2_residual": True},
    "depth2_residual_mixed": {"depths": [2, 3, 2], "depth2_residual": True},
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
    off = {k: v for k, v in OFF.items() if k not in reference.DEFAULTS}
    if shared:
        print(f"promoted defaults vs {REFERENCE_REF}: {shared}", flush=True)
    if ours_only:
        print(f"switches only in our file: {ours_only} (control turns off {off})", flush=True)
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
    if "jitter" in off and new.DEFAULTS["jitter"]:
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

    for name in ("count_nonfinite", "jitter_res24_nonfinite", "filter_nonfinite"):
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
    check(
        "every run of our file took exactly total_steps optimizer steps (state.steps_taken)",
        all(r.state.steps_taken == r.state.total_steps for r in (b, d, *runs.values())),
    )
    for name, run in runs.items():
        hyp, delta = run.state.hyp, VARIANTS[name]
        if "res_schedule" in delta or "train_resolution" in delta:
            if hyp["res_schedule"] is None:
                expected = sorted({hyp["train_resolution"], 32})
            else:
                expected = sorted({r for r, _ in hyp["res_schedule"]} | {32})
            low = [r for r in expected if r < 32]
            small = run.state.small_images
            side = {r: r + 2 * hyp["translate"] for r in low}
            check(
                f"{name}: _resolutions {expected}; small_images has one channels-last copy per "
                f"low resolution with sides {side}",
                new._resolutions(hyp) == expected
                and sorted(small) == low
                and all(
                    tuple(small[r].shape) == (count, 3, side[r], side[r])
                    and small[r].is_contiguous(memory_format=torch.channels_last)
                    for r in low
                ),
            )
        if hyp["filter_start"]:
            # BASE trains whole epochs: count // batch_size steps per epoch before filter_start,
            # round(filter_keep * count) // batch_size from then on.
            expected = sum(
                (round(hyp["filter_keep"] * count) if e >= hyp["filter_start"] else count)
                // BASE["batch_size"]
                for e in range(int(BASE["epochs"]))
            )
            check(
                f"{name}: total_steps == steps_taken == {expected} on the filtered schedule",
                run.state.total_steps == run.state.steps_taken == expected,
            )
            check(
                f"{name}: every training example has a finite loss score after train",
                tuple(run.state.scores.shape) == (count,)
                and bool(torch.isfinite(run.state.scores).all()),
            )

    # Stems: the map after the whitening stem on a 32x32 input, group 1's entry, and the
    # whitening filters (eigenvector filters followed by their negatives, frozen).
    expected_side = {
        "space_to_depth": 15,
        "space_to_depth_nopool": 15,
        "whiten4s2": 15,
        "whiten3s2": 15,
        "conv1x1": 31,
    }
    whiten_shape = {
        "space_to_depth": (96, 12, 2, 2),
        "space_to_depth_nopool": (96, 12, 2, 2),
        "whiten4s2": (96, 3, 4, 4),
        "whiten3s2": (54, 3, 3, 3),
        "conv1x1": (24, 3, 2, 2),
    }
    for name, run in runs.items():
        stem, net = run.state.hyp["stem"], run.state.net
        if stem == "conv":
            continue
        with torch.inference_mode():
            side = net.stem(torch.zeros(1, 3, 32, 32)).shape[-1]
        check(
            f"{name}: {side}x{side} map after the stem on a 32x32 input (expected "
            f"{expected_side[stem]}; sizes after the stem and each group {new._stem_sizes(stem, 32)})",
            side == expected_side[stem] == new._stem_sizes(stem, 32)[0],
        )
        group = net.layers[1]
        if stem == "whiten4s2":
            entry = group.conv1 is None and group.pool is None and group.norm1 is None
        else:
            kernel = (3, 3) if stem.startswith("space_to_depth") else (1, 1)
            no_pool = stem in ("whiten3s2", "space_to_depth_nopool")
            pool = torch.nn.Identity if no_pool else torch.nn.MaxPool2d
            entry = group.conv1.kernel_size == kernel and isinstance(group.pool, pool)
        w, half = net.whiten.weight, whiten_shape[stem][0] // 2
        check(
            f"{name}: group 1 entry matches the stem; frozen whitening weight "
            f"{whiten_shape[stem]} whose filters 0-{half - 1} are the negatives of "
            f"{half}-{2 * half - 1}",
            entry
            and tuple(w.shape) == whiten_shape[stem]
            and torch.equal(w[:half], -w[half:])
            and not w.requires_grad
            and net.whiten.bias.requires_grad,
        )

    # Spatial sizes per stem and resolution (after the stem, then after each group), and the
    # combinations build must reject: no 2x2 pool may see a map smaller than 2x2.
    table = {}
    for stem in new.STEMS:
        table[stem] = {}
        for resolution in (32, 28, 24, 20, 16):
            try:
                table[stem][resolution] = new._stem_sizes(stem, resolution)
            except ValueError:
                table[stem][resolution] = None
        print(f"stem {stem}: sizes after the stem and each group {table[stem]}", flush=True)
    expected_32 = {
        "conv": [31, 15, 7, 3],
        "space_to_depth": [15, 7, 3, 1],
        "space_to_depth_nopool": [15, 15, 7, 3],
        "whiten4s2": [15, 15, 7, 3],
        "whiten3s2": [15, 15, 7, 3],
        "conv1x1": [31, 15, 7, 3],
    }
    check(
        f"32 px sizes after the stem and each group: {expected_32}",
        {stem: sizes[32] for stem, sizes in table.items()} == expected_32,
    )
    check(
        "every stem accepts 20, 24, 28 and 32 px; 16 px is rejected by space_to_depth only",
        all(table[s][r] for s in table for r in (20, 24, 28, 32))
        and [s for s in table if table[s][16] is None] == ["space_to_depth"],
    )
    rejected = {
        "space_to_depth at 16 px (a 1x1 map before group 3's pool)": {
            "stem": "space_to_depth",
            "res_schedule": [[16, 0.5]],
        },
        "whiten4s2 with widths[0] != 96": {"stem": "whiten4s2"},
        "whiten4s2 with pool_first[0]": {
            "stem": "whiten4s2",
            "widths": [96, 64, 64],
            "pool_first": [True, False, False],
        },
        "an unknown stem": {"stem": "pool"},
    }
    for name, delta in rejected.items():
        try:
            new.build(BuildContext(torch.device("cpu"), {**BASE, **delta}))
        except ValueError as exc:
            check(f"build rejects {name}: {exc}", True)
        else:
            check(f"build rejects {name}", False)
    print(f"\n{sum(results)}/{len(results)} checks passed", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
