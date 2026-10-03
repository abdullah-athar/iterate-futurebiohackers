"""Hyperparameter-variant check for submissions/futurebiohackers. Not part of the submission.

1. Default-path faithfulness: with default parameters the working-tree recipe must give a trial
   that is bit-identical (weights and predictions) to the reference recipe, by default the team
   baseline on origin/speedrun-g3-512-10ep (PR #23, the current record). Shared parameters whose defaults differ are set to OUR defaults on
   both sides (and printed) and parameters only our file knows are turned off, so the comparison
   tests the code path rather than promoted values. The control also keeps the reference's step
   count and prints nothing on stderr.
2. Every hyperparameter variant we plan to screen (optimizer, schedule shape, augmentation,
   resolution and batch schedules, systems switches; the architecture is out of scope) and every
   default-off switch of our own (the un-augmented finish, progressive depth, the label-smoothing
   schedule, the non-finite counter) must run end to end on CPU with finite predictions, taking
   exactly the optimizer steps the recipe's own loop arithmetic implies, with the learning rate,
   weight decay, whitening-bias gradient, training resolution, input layout, skip phase and label
   smoothing of every step following the recipe's formulas on example progress.
   Structural checks cover the downscaled copies, the lookahead EMA, the whitening-bias freeze,
   the optimizer parametrization, BatchNorm momentum and recalibration, the un-augmented view,
   the skipped convolutions, the smoothing tensor and the combinations build must reject.

CPU, synthetic images, well under a minute. Run from the speedrun env with cwd cifar100-speedrun:
    uv run python ../scripts/check_variants.py [reference_submission_dir]
    scripts/wsl_speedrun.sh python ../scripts/check_variants.py          # Windows, via WSL
The reference defaults to `git show origin/speedrun-g3-512-10ep:...submission.py`; set CHECK_VARIANTS_REF=<git
ref> to compare against another commit (it must share the baseline's state layout).
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
import torch.nn.functional as F

from benchmark.api import BuildContext
from benchmark.data import synthetic_split
from benchmark.worker import load_submission, seed_everything

TEAM_DIR = (
    Path(__file__).resolve().parents[1] / "cifar100-speedrun" / "submissions" / "futurebiohackers"
)
REFERENCE_REF = os.environ.get("CHECK_VARIANTS_REF", "origin/speedrun-g3-512-10ep")
# The synthetic split has 64 images: batch 8 gives 8 steps per epoch, 16 in two epochs.
BASE = {"widths": [32, 64, 64], "epochs": 2.0, "batch_size": 8, "compile": ""}
# "Off" values for parameters only our file knows. The control turns every such parameter off
# for the comparison; the defaults are the off values today, the entries keep the control clean
# if one of them is promoted later.
OFF: dict[str, object] = {
    "aug_off_last": 0.0,
    "skip_residual_until": 0.0,
    "skip_residual_groups": [],
    "label_smoothing_end": None,
    "g3_pair": "full",
}
# --params deltas on top of BASE; every other value is the recipe's default (28 px first half).
VARIANTS = {
    # optimizer and schedule shape
    "bias_scaler_16": {"bias_scaler": 16.0},
    "lr_13.8": {"lr": 13.8},
    "weight_decay_0.0238": {"weight_decay": 0.0238},
    "momentum_0.9": {"momentum": 0.9},
    "label_smoothing_0.3": {"label_smoothing": 0.3},
    "warmup_0.15": {"warmup": 0.15},
    "final_lr_0.03": {"final_lr": 0.03},
    "scaling_factor_1.5": {"scaling_factor": 1.5 / 9},
    "bn_momentum_0.7": {"bn_momentum": 0.7},
    # whitening bias and lookahead EMA. BASE trains 2 epochs, so 2 and 4 keep the bias training
    # throughout exactly like the default 3; only 1 and 0 exercise the freeze.
    "whiten_bias_epochs_2": {"whiten_bias_epochs": 2},
    "whiten_bias_epochs_4": {"whiten_bias_epochs": 4},
    "whiten_bias_epochs_1": {"whiten_bias_epochs": 1},
    "whiten_bias_epochs_1_no_ema": {"whiten_bias_epochs": 1, "ema_every": 0},
    "whiten_bias_epochs_0": {"whiten_bias_epochs": 0},
    "ema_every_0": {"ema_every": 0},
    "ema_every_4": {"ema_every": 4},
    # augmentation
    "translate_1": {"translate": 1},
    "translate_3": {"translate": 3},
    "translate_0": {"translate": 0},
    "cutout_4": {"cutout": 4},
    "color_jitter_0.3": {"color_jitter": [0.3, 0.3]},
    "color_jitter_off": {"color_jitter": [0, 0]},
    "masked_crop": {"crop_mode": "masked"},
    "masked_crop_translate_3": {"crop_mode": "masked", "translate": 3},
    # length, batch and resolution schedules (fractions run on example progress)
    "epochs_1.5": {"epochs": 1.5},
    "batch_4_lr_16": {"batch_size": 4, "lr": 16.0},
    "batch_schedule_4_half": {"batch_schedule": [[4, 0.5]]},
    "res_24_28": {"resolution_schedule": [[24, 0.25], [28, 0.5]]},
    "res_28_late": {"resolution_schedule": [[28, 0.6]]},
    "res_none": {"resolution_schedule": []},
    "res_20": {"resolution_schedule": [[20, 0.3]]},
    # a stack of candidate winners
    "stack": {
        "bias_scaler": 16.0,
        "lr": 13.8,
        "weight_decay": 0.0238,
        "bn_momentum": 0.7,
        "color_jitter": [0.3, 0.3],
        "resolution_schedule": [[24, 0.25], [28, 0.5]],
        "compile_loss": True,
    },
    # systems switches that keep the network (no effect on CPU, but they must run)
    "compile_loss": {"compile_loss": True},
    "no_fused_sgd": {"fused_sgd": False},
    "bn_recal_2": {"bn_recal_batches": 2},
    "gelu_tanh": {"gelu_approximate": "tanh"},
    # build sets this flag process-wide on torch._inductor.config; main resets it afterwards
    "coordinate_descent": {"inductor_tuning": ["coordinate_descent_tuning"]},
}
# Our own default-off switches, run only when the working-tree file has them (the reference
# may not): name -> delta. count_nonfinite reports "NONFINITE_LOSSES n" on stderr after train.
OWN_SWITCHES = {
    "count_nonfinite": {"count_nonfinite": True},
    # cheaper residual pair in group 3 (residual and output stay at widths[2])
    "g3_pair_inner512": {"g3_pair": "inner512"},
    "g3_pair_inner640": {"g3_pair": "inner640"},
    "g3_pair_bottleneck384": {"g3_pair": "bottleneck384"},
    # un-augmented finish over the last 1.0 / 0.5 of BASE's 2 epochs (steps 8-15 / 12-15), in
    # every crop layout: indexed (channels-last), masked and cutout (NCHW), translate 0 (source)
    "aug_off_last_1.0": {"aug_off_last": 1.0},
    "aug_off_last_0.5": {"aug_off_last": 0.5},
    "aug_off_last_0.5_masked": {"aug_off_last": 0.5, "crop_mode": "masked"},
    "aug_off_last_0.5_cutout": {"aug_off_last": 0.5, "cutout": 4},
    "aug_off_last_0.5_translate_0": {"aug_off_last": 0.5, "translate": 0},
    # progressive depth: the first quarter (steps 0-3) without conv2/conv3 of one group; the
    # eager skip phase; all groups for half the steps with the whitening bias frozen midway
    "skip_g0_0.25": {"skip_residual_until": 0.25, "skip_residual_groups": [0]},
    "skip_g1_0.25_eager": {
        "skip_residual_until": 0.25,
        "skip_residual_groups": [1],
        "skip_compile": "",
    },
    "skip_g012_0.5_whiten_1": {
        "skip_residual_until": 0.5,
        "skip_residual_groups": [0, 1, 2],
        "whiten_bias_epochs": 1,
    },
    # label smoothing 0.35 -> 0.15 linearly over example progress
    "label_smoothing_0.35_to_0.15": {"label_smoothing": 0.35, "label_smoothing_end": 0.15},
}
results: list[bool] = []


def check(name: str, ok: bool) -> None:
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)


def check_all(name: str, runs: dict, predicate) -> None:
    """One PASS/FAIL line for a property every run must have; failures name the runs."""
    bad = []
    for label, run in runs.items():
        try:
            ok = predicate(run)
        except Exception as exc:  # noqa: BLE001
            ok, label = False, f"{label} ({type(exc).__name__}: {exc})"
        if not ok:
            bad.append(label)
    check(name + (f" -- failing: {bad}" if bad else ""), not bad)


def instrument(state) -> SimpleNamespace:
    """Record what training does: per optimizer step the group lrs and weight decays and the
    whitening bias (index 0 is the value after prepare); per forward call the whiten_bias_grad
    flag, the input side (the training resolution), the input strides, a copy of the inputs,
    whether the progressive-depth forward ran and the label smoothing in force; the conv2 calls
    of every group; plus the lookahead EMA after prepare."""
    rec = SimpleNamespace(
        steps=[0] * len(state.optimizers),
        lrs=[],
        wds=[],
        bias=[state.net.whiten.bias.detach().clone()],
        flags=[],
        sides=[],
        strides=[],
        inputs=[],
        skips=[],
        eps=[],
        conv2_calls=[0, 0, 0],
        ema0=[t.clone() for t in state.ema],
    )
    for index, optimizer in enumerate(state.optimizers):

        def hook(optimizer, args, kwargs, index=index):
            rec.steps[index] += 1
            if index == 0:
                rec.lrs.append([g["lr"] for g in optimizer.param_groups])
                rec.wds.append([g["weight_decay"] for g in optimizer.param_groups])
                rec.bias.append(state.net.whiten.bias.detach().clone())

        optimizer.register_step_post_hook(hook)

    def wrap(inner, skip):
        def forward_loss(inputs, targets, whiten_bias_grad):
            rec.flags.append(bool(whiten_bias_grad))
            rec.sides.append(inputs.shape[-1])
            rec.strides.append(tuple(inputs.stride()))
            rec.inputs.append(inputs.detach().clone())
            rec.skips.append(skip)
            smoothing = getattr(state, "smoothing", None)  # the reference has no such tensor
            rec.eps.append(None if smoothing is None else float(smoothing))
            return inner(inputs, targets, whiten_bias_grad)

        return forward_loss

    state.forward_loss = wrap(state.forward_loss, False)
    if getattr(state, "forward_loss_skip", None) is not None:
        state.forward_loss_skip = wrap(state.forward_loss_skip, True)
    for index, group in enumerate(state.net.layers):

        def conv2_hook(module, args, output, index=index):
            rec.conv2_calls[index] += 1

        group.conv2.register_forward_hook(conv2_hook)
    return rec


def trial(module, params, seed=7):
    data = synthetic_split(train=True)
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        state = module.build(BuildContext(torch.device("cpu"), params))
        seed_everything(seed)
        module.prepare(state, data, seed)
        rec = instrument(state)
        model = module.train(state)
        rec.conv2_train = list(rec.conv2_calls)  # before the evaluation forward below
    model.eval()
    with torch.inference_mode():
        out = model(synthetic_split(train=False).images.float().div(255))
    if not torch.isfinite(out).all():
        raise RuntimeError("non-finite predictions")
    weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return SimpleNamespace(weights=weights, out=out, state=state, stderr=stderr.getvalue(), rec=rec)


def schedule(hyp, count):
    """Replay the recipe's loop arithmetic on `count` training examples: the batch of an epoch
    comes from batch_schedule by progress at the epoch start, steps_per_epoch = count // batch,
    progress = examples seen / total examples. Returns total_steps and (progress, batch) per step.
    """
    base = min(hyp["batch_size"], count)
    total_steps = math.ceil(hyp["epochs"] * (count // base))
    total_examples = total_steps * base
    steps, seen = [], 0
    while seen < total_examples:
        progress = seen / total_examples
        batch = min(next((b for b, end in hyp["batch_schedule"] if progress < end), base), count)
        for _ in range(count // batch):
            if seen >= total_examples:
                break
            steps.append((seen / total_examples, batch))
            seen += batch
    return total_steps, steps


def lr_scale(hyp, total_steps, progress):
    """The recipe's multiplier: 0.2 -> 1 over int(total_steps * warmup) steps, then linearly
    towards final_lr at progress 1 (never reached: the last step sits at (T - 1) / T)."""
    warmup_frac = int(total_steps * hyp["warmup"]) / total_steps
    if progress < warmup_frac:
        frac = progress / warmup_frac
        return 0.2 * (1 - frac) + frac
    frac = (progress - warmup_frac) / max(1e-9, 1 - warmup_frac)
    return (1 - frac) + hyp["final_lr"] * frac


def whiten_steps(hyp, count):
    return math.ceil(hyp["whiten_bias_epochs"] * (count // min(hyp["batch_size"], count)))


def resolution_at(hyp, progress):
    return next((r for r, end in hyp["resolution_schedule"] if progress < end), 32)


def batchnorms(net):
    return [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)]


def reference_dir() -> tuple[Path, str]:
    """The reference submission folder and its label: argv[1], or REFERENCE_REF via git show."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve(), sys.argv[1]
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
    return folder, REFERENCE_REF


def main() -> int:
    torch.set_num_threads(4)
    import torch._inductor.config as inductor_config

    saved_flag = inductor_config.coordinate_descent_tuning
    folder, label = reference_dir()
    reference = load_submission(folder)
    new = load_submission(TEAM_DIR)
    count = len(synthetic_split(train=True).labels)

    # 1. Faithfulness: shared parameters at OUR defaults on both sides, our own parameters off.
    shared = {
        k: v
        for k, v in new.DEFAULTS.items()
        if k in reference.DEFAULTS and reference.DEFAULTS[k] != v and k not in BASE
    }
    ours_only = {k: OFF.get(k, v) for k, v in new.DEFAULTS.items() if k not in reference.DEFAULTS}
    if shared:
        print(f"promoted defaults vs {label}: {shared}", flush=True)
    if ours_only:
        print(f"parameters only in our file, turned off for the control: {ours_only}", flush=True)
    a = trial(reference, {**BASE, **shared})
    b = trial(new, {**BASE, **shared, **ours_only})
    same = (
        set(a.weights) == set(b.weights)
        and all(torch.equal(a.weights[k], b.weights[k]) for k in a.weights)
        and torch.equal(a.out, b.out)
    )
    check(f"default path bit-identical to the reference recipe ({label})", same)
    control_steps = math.ceil(BASE["epochs"] * (count // BASE["batch_size"]))
    check(
        f"default path: total_steps {control_steps} on both files, {control_steps} optimizer steps",
        a.state.total_steps == control_steps == b.state.total_steps
        and a.rec.steps == b.rec.steps == [control_steps],
    )
    check(
        "default path prints nothing on stderr (build, prepare, train)",
        a.stderr.strip() == "" == b.stderr.strip(),
    )
    d = trial(new, BASE)  # our real defaults (plus the small BASE overrides)
    check("our defaults run with finite predictions and nothing on stderr", d.stderr.strip() == "")
    if not shared and all(new.DEFAULTS[k] == v for k, v in ours_only.items()):
        # Every switch the reference lacks is off by default, so our defaults must reproduce
        # the control trial exactly (weights and predictions).
        check(
            "our defaults equal the control (own switches off by default): bit-identical trial",
            set(d.weights) == set(b.weights)
            and all(torch.equal(d.weights[k], b.weights[k]) for k in d.weights)
            and torch.equal(d.out, b.out),
        )

    # 2. Every planned variant runs on CPU with finite predictions.
    variants = dict(VARIANTS)
    for name, delta in OWN_SWITCHES.items():
        if all(k in new.DEFAULTS for k in delta):
            variants[name] = delta
    runs = {}
    for name, delta in variants.items():
        t0 = time.perf_counter()
        try:
            runs[name] = trial(new, {**BASE, **delta})
            check(f"variant {name} runs ({time.perf_counter() - t0:.1f} s)", True)
        except Exception as exc:  # noqa: BLE001
            check(f"variant {name}: {type(exc).__name__}: {exc}", False)
        if "inductor_tuning" in delta:
            check(
                f"{name}: build set torch._inductor.config.coordinate_descent_tuning for the whole "
                "process (reset here; harmless for the harness: one benchmark.run per process)",
                inductor_config.coordinate_descent_tuning is True,
            )
            inductor_config.coordinate_descent_tuning = saved_flag
    every = {"control": b, "defaults": d, **runs}

    def stderr_ok(run):
        if run.state.hyp.get("count_nonfinite"):
            return run.stderr.splitlines() == ["NONFINITE_LOSSES 0"]
        return run.stderr.strip() == ""

    check_all(
        "every run: nothing on stderr (count_nonfinite: exactly one line, NONFINITE_LOSSES 0)",
        every,
        stderr_ok,
    )
    if "count_nonfinite" in runs:
        check(
            "count_nonfinite: the device counter stays 0 and is reported once after train; the "
            "default leaves state.nonfinite None and prints nothing",
            int(runs["count_nonfinite"].state.nonfinite) == 0
            and b.state.nonfinite is None
            and b.stderr == "",
        )

    # 3. Step counts and per-step schedules from the recipe's own arithmetic.
    detail = ", ".join(
        f"{name} {len(schedule(runs[name].state.hyp, count)[1])} of total_steps "
        f"{schedule(runs[name].state.hyp, count)[0]}"
        for name in ("batch_schedule_4_half", "epochs_1.5", "batch_4_lr_16")
        if name in runs
    )

    def steps_ok(run):
        total, steps = schedule(run.state.hyp, count)
        return (
            run.state.total_steps == total
            and run.rec.steps == [len(steps)] * len(run.rec.steps)
            and len(run.rec.flags) == len(run.rec.lrs) == len(steps)
        )

    check_all(
        "every run: state.total_steps == ceil(epochs x steps_per_epoch) and every optimizer took "
        f"exactly the steps the loop implies ({detail})",
        every,
        steps_ok,
    )

    def lr_ok(run):
        hyp = run.state.hyp
        total, steps = schedule(hyp, count)
        base = min(hyp["batch_size"], count)
        groups = run.state.optimizers[0].param_groups
        for (progress, batch), lrs, wds in zip(steps, run.rec.lrs, run.rec.wds, strict=True):
            scale = lr_scale(hyp, total, progress)
            for group, lr, wd in zip(groups, lrs, wds, strict=True):
                if "whiten" in group:
                    continue  # the Muon path's ramped whitening-bias group; not used by SGD
                if not math.isclose(lr, group["initial_lr"] * scale, rel_tol=1e-9):
                    return False
                expected_wd = group["initial_weight_decay"] * batch / base
                if not math.isclose(wd, expected_wd, rel_tol=1e-9):
                    return False
        return True

    check_all(
        "every run: the lr of every step follows warmup/final_lr on example progress and the "
        "weight decay follows batch / base batch (halved while batch_schedule_4_half runs at 4)",
        every,
        lr_ok,
    )

    def flags_ok(run):
        hyp = run.state.hyp
        total, steps = schedule(hyp, count)
        frac = whiten_steps(hyp, count) / total
        return run.state.whiten_bias_steps == whiten_steps(hyp, count) and run.rec.flags == [
            progress < frac for progress, _ in steps
        ]

    check_all(
        "every run: whiten_bias_grad per step == progress < ceil(whiten_bias_epochs x "
        "steps_per_epoch) / total_steps",
        every,
        flags_ok,
    )
    check_all(
        "every run: the training resolution of every step follows resolution_schedule on example "
        "progress (32 px once past the last fraction)",
        every,
        lambda run: run.rec.sides
        == [resolution_at(run.state.hyp, p) for p, _ in schedule(run.state.hyp, count)[1]],
    )

    def copies_ok(run):
        hyp, state = run.state.hyp, run.state
        t = hyp["translate"]
        low = sorted({r for r, _ in hyp["resolution_schedule"]})
        return (
            sorted(state.small_images) == low == sorted(state.resize)
            and all(tuple(state.resize[r].shape) == (r, 32) for r in low)
            and all(
                tuple(state.small_images[r].shape) == (count, 3, r + 2 * t, r + 2 * t)
                and state.small_images[r].is_contiguous(memory_format=torch.channels_last)
                for r in low
            )
            and tuple(state.images.shape) == (count, 3, 32 + 2 * t, 32 + 2 * t)
        )

    check_all(
        "every run: one channels-last downscaled copy per low resolution with side "
        "r + 2 x translate, one (r, 32) resize matrix, 32 px images with side 32 + 2 x translate",
        every,
        copies_ok,
    )

    def strides_ok(run):
        by_side = {}
        for side, stride in zip(run.rec.sides, run.rec.strides, strict=True):
            by_side.setdefault(side, set()).add(stride)
        return all(len(strides) == 1 for strides in by_side.values())

    check_all(
        "every run: all steps at one resolution feed inputs with one stride tuple, the "
        "un-augmented finish included (a static-shape graph never meets a new layout)",
        every,
        strides_ok,
    )

    # 4. Lookahead EMA, whitening-bias freeze, optimizer parametrization, BatchNorm.
    ema_runs = {
        k: r
        for k, r in every.items()
        if r.state.hyp["ema_every"] and not r.state.hyp["bn_recal_batches"]
    }
    check_all(
        "every run with ema_every > 0 (no BN recalibration): train ends by copying the lookahead "
        "EMA into the weights (state.ema == state.float_state)",
        ema_runs,
        lambda run: all(
            torch.equal(e, p) for e, p in zip(run.state.ema, run.state.float_state, strict=True)
        ),
    )
    if "ema_every_0" in runs:
        run = runs["ema_every_0"]
        check(
            "ema_every_0: train leaves state.ema at the post-prepare copy while the weights moved",
            all(torch.equal(e, e0) for e, e0 in zip(run.state.ema, run.rec.ema0, strict=True))
            and not all(
                torch.equal(e, p) for e, p in zip(run.state.ema, run.state.float_state, strict=True)
            ),
        )
    if "whiten_bias_epochs_0" in runs:
        run = runs["whiten_bias_epochs_0"]
        check(
            "whiten_bias_epochs_0: the whitening bias never gets a gradient and stays zero",
            not any(run.rec.flags) and int(torch.count_nonzero(run.state.net.whiten.bias)) == 0,
        )
    if "whiten_bias_epochs_1_no_ema" in runs:
        run = runs["whiten_bias_epochs_1_no_ema"]
        ws, bias = whiten_steps(run.state.hyp, count), run.rec.bias
        check(
            f"whiten_bias_epochs_1_no_ema: the bias moves during the first {ws} steps and is "
            f"constant over the last {len(bias) - 1 - ws} (SGD skips the detached bias)",
            not torch.equal(bias[ws], bias[0])
            and all(torch.equal(bias[k], bias[ws]) for k in range(ws, len(bias)))
            and torch.equal(run.weights["net.whiten.bias"], bias[-1]),
        )
    if "whiten_bias_epochs_1" in runs:
        run = runs["whiten_bias_epochs_1"]
        ws, bias = whiten_steps(run.state.hyp, count), run.rec.bias
        period = run.state.hyp["ema_every"]
        moves = [k for k in range(ws + 1, len(bias)) if not torch.equal(bias[k], bias[k - 1])]
        check(
            "whiten_bias_epochs_1: after the freeze the bias changes only right after a lookahead "
            f"(moved after steps {[k - 1 for k in moves]}; the EMA still pulls the frozen bias once)",
            not torch.equal(bias[ws], bias[0]) and all((k - 1) % period == 0 for k in moves),
        )

    def groups_ok(run):
        hyp, net = run.state.hyp, run.state.net
        kilostep = 1024 * (1 + 1 / (1 - hyp["momentum"]))
        lr = hyp["lr"] / kilostep
        wd = hyp["weight_decay"] * min(hyp["batch_size"], count) / kilostep
        biases, others = run.state.optimizers[0].param_groups
        bn_biases = sorted(
            id(p) for n, p in net.named_parameters() if "norm" in n and p.requires_grad
        )
        close = lambda x, y: math.isclose(x, y, rel_tol=1e-12)  # noqa: E731
        return (
            close(others["initial_lr"], lr)
            and close(biases["initial_lr"], lr * hyp["bias_scaler"])
            and close(others["initial_lr"] * others["initial_weight_decay"], wd)
            and close(biases["initial_lr"] * biases["initial_weight_decay"], wd)
            and sorted(id(p) for p in biases["params"]) == bn_biases
            and any(p is net.whiten.bias for p in others["params"])
            and not biases["fused"]
            and not others["fused"]
        )

    check_all(
        "every run: lr = lr / (1024 (1 + 1 / (1 - momentum))), the BatchNorm-bias group runs at "
        "bias_scaler x that, the decay lr x wd is the same for both groups (decoupled, per example "
        "x batch), the whitening bias sits in the base group, fused SGD is never requested on CPU",
        every,
        groups_ok,
    )
    if "bn_momentum_0.7" in runs:
        check(
            "bn_momentum_0.7: every BatchNorm has torch momentum 1 - 0.7 (slower running stats)",
            all(m.momentum == 1 - 0.7 for m in batchnorms(runs["bn_momentum_0.7"].state.net)),
        )
    check(
        f"control: every BatchNorm tracked exactly total_steps ({control_steps}) batches",
        all(int(m.num_batches_tracked) == control_steps for m in batchnorms(b.state.net)),
    )
    if "bn_recal_2" in runs:
        run = runs["bn_recal_2"]
        check(
            "bn_recal_2: after train every BatchNorm tracked exactly the 2 recalibration batches "
            "(cumulative average) and has its momentum 1 - bn_momentum back",
            all(
                int(m.num_batches_tracked) == 2 and m.momentum == 1 - run.state.hyp["bn_momentum"]
                for m in batchnorms(run.state.net)
            ),
        )
    if "gelu_tanh" in runs:
        x = torch.linspace(-3, 3, 13)
        act = runs["gelu_tanh"].state.net.act
        check(
            "gelu_tanh: the network's activation is the tanh approximation of GELU",
            torch.equal(act(x), F.gelu(x, approximate="tanh"))
            and not torch.equal(act(x), F.gelu(x)),
        )
    if "scaling_factor_1.5" in runs:
        check(
            "scaling_factor_1.5: the head logits are scaled by 1.5 / 9",
            runs["scaling_factor_1.5"].state.net.scaling_factor == 1.5 / 9,
        )
    if "label_smoothing_0.3" in runs:
        generator = torch.Generator().manual_seed(0)
        logits = torch.randn(5, 100, generator=generator)
        labels = torch.randint(0, 100, (5,), generator=generator)
        check(
            "label_smoothing_0.3: loss_fn is the summed cross-entropy with label smoothing 0.3",
            torch.equal(
                runs["label_smoothing_0.3"].state.loss_fn(logits, labels),
                F.cross_entropy(logits, labels, label_smoothing=0.3, reduction="sum"),
            ),
        )

    # 5. Our default-off switches: un-augmented finish, progressive depth, smoothing schedule.
    def center_views(run):
        """Per forward call: is every input image (after undoing the odd-epoch mirror) the
        center crop of a stored training image? True exactly on un-augmented steps, since the
        color jitter of the augmented ones never reproduces a stored image."""
        state, hyp = run.state, run.state.hyp
        t = hyp["translate"]
        steps_per_epoch = count // min(hyp["batch_size"], count)
        views = []
        for k, (inputs, side) in enumerate(zip(run.rec.inputs, run.rec.sides, strict=True)):
            source = state.small_images[side] if side < 32 else state.images
            centers = (source[:, :, t : t + side, t : t + side] if t else source).flatten(1)
            batch = inputs.flip(-1) if (k // steps_per_epoch) % 2 == 1 else inputs
            views.append(all((centers == image.flatten()).all(1).any() for image in batch))
        return views

    def aug_off_ok(run):
        hyp = run.state.hyp
        _, steps = schedule(hyp, count)
        off_from = 1 - hyp["aug_off_last"] / hyp["epochs"]
        return center_views(run) == [progress >= off_from for progress, _ in steps]

    jittered = {
        k: r
        for k, r in every.items()
        if k in ("control", "defaults") or k.startswith("aug_off_last")
    }
    check_all(
        "control, defaults and every aug_off_last run: the inputs are un-augmented center crops "
        "(odd epochs mirrored) exactly for progress >= 1 - aug_off_last / epochs, never before",
        jittered,
        aug_off_ok,
    )

    def skips_ok(run):
        hyp = run.state.hyp
        _, steps = schedule(hyp, count)
        has_skip = getattr(run.state, "forward_loss_skip", None) is not None
        return has_skip == (hyp["skip_residual_until"] > 0) and run.rec.skips == [
            progress < hyp["skip_residual_until"] for progress, _ in steps
        ]

    check_all(
        "every run: forward_loss_skip exists exactly when skip_residual_until > 0 and is used "
        "exactly for progress < skip_residual_until",
        every,
        skips_ok,
    )

    def conv2_ok(run):
        hyp = run.state.hyp
        _, steps = schedule(hyp, count)
        skipped = sum(progress < hyp["skip_residual_until"] for progress, _ in steps)
        recal = min(hyp["bn_recal_batches"], count // min(hyp["batch_size"], count))
        expected = [
            len(steps) + recal - (skipped if g in hyp["skip_residual_groups"] else 0)
            for g in range(3)
        ]
        return run.rec.conv2_train == expected

    check_all(
        "every run: conv2 of a skipped group ran only on the non-skip steps, every other conv2 "
        "on every step (plus the BN recalibration batches)",
        every,
        conv2_ok,
    )
    plan = getattr(new, "_skip_warmup", None)
    if plan is not None:
        skip_hyp = {
            **new.DEFAULTS,
            "resolution_schedule": [[28, 0.5]],
            "epochs": 9.0,
            "skip_residual_until": 0.25,
            "skip_residual_groups": [0],
        }
        check(
            "_skip_warmup: with main's schedule (28 px to 0.5, batch 1024, 3 of 9 whitening "
            "epochs) the skip graph is warmed for (1024, 28) with the trained whitening bias only "
            "(whiten_bias_steps 6 of 6); 24 px to 0.25 then 28 px: (1024, 24) only; batch "
            "schedule [[512, 0.1]]: (512, 28) and (1024, 28); whiten_bias_epochs 1: both flags "
            "(3); whiten_bias_epochs 0: the frozen flag only (0); off: nothing",
            plan(skip_hyp) == {(1024, 28): 6}
            and plan({**skip_hyp, "resolution_schedule": [[24, 0.25], [28, 0.5]]})
            == {(1024, 24): 6}
            and plan({**skip_hyp, "batch_schedule": [[512, 0.1]]}) == {(512, 28): 6, (1024, 28): 6}
            and plan({**skip_hyp, "whiten_bias_epochs": 1}) == {(1024, 28): 3}
            and plan({**skip_hyp, "whiten_bias_epochs": 0}) == {(1024, 28): 0}
            and plan(new.DEFAULTS) == {},
        )

    def eps_ok(run):
        hyp = run.state.hyp
        _, steps = schedule(hyp, count)
        start, end = hyp["label_smoothing"], hyp["label_smoothing_end"]
        smoothing = getattr(run.state, "smoothing", None)
        if end is None:
            return smoothing is None and all(e is None for e in run.rec.eps)
        return smoothing is not None and all(
            math.isclose(e, start + (end - start) * progress, abs_tol=1e-6)
            for e, (progress, _) in zip(run.rec.eps, steps, strict=True)
        )

    check_all(
        "every run: the label smoothing of every step is label_smoothing + (label_smoothing_end "
        "- label_smoothing) x progress, read from the device tensor (None without the schedule)",
        every,
        eps_ok,
    )
    generator = torch.Generator().manual_seed(1)
    logits = torch.randn(64, 100, generator=generator) * 3
    labels = torch.randint(0, 100, (64,), generator=generator)
    check(
        "control: loss_fn is F.cross_entropy(label_smoothing=0.25, reduction='sum') bit for bit "
        "and state.smoothing is None",
        torch.equal(
            b.state.loss_fn(logits, labels),
            F.cross_entropy(logits, labels, label_smoothing=0.25, reduction="sum"),
        )
        and b.state.smoothing is None,
    )
    if "label_smoothing_0.35_to_0.15" in runs:
        run = runs["label_smoothing_0.35_to_0.15"]
        worst = 0.0
        for eps in (0.25, 0.35, 0.15):
            run.state.smoothing.fill_(eps)
            x = logits.clone().requires_grad_()
            ours = run.state.loss_fn(x, labels)
            (grad_ours,) = torch.autograd.grad(ours, x)
            ref = F.cross_entropy(x, labels, label_smoothing=eps, reduction="sum")
            (grad_ref,) = torch.autograd.grad(ref, x)
            worst = max(
                worst,
                float(abs(ours - ref) / ref),
                float((grad_ours - grad_ref).abs().max() / grad_ref.abs().max()),
            )
        last = run.rec.eps[-1]
        new.prepare(run.state, synthetic_split(train=True), 7)  # the last use of this run
        check(
            "label_smoothing_0.35_to_0.15: the tensor-eps loss matches F.cross_entropy(label_"
            f"smoothing=eps, reduction='sum') in value and gradient (worst rel diff {worst:.1e}, "
            f"limit 1e-5); the last step used eps {last:.4f} (0.35 - 0.2 x 15/16); prepare "
            "resets the tensor to 0.35",
            worst < 1e-5
            and math.isclose(last, 0.35 - 0.2 * 15 / 16, abs_tol=1e-6)
            and math.isclose(float(run.state.smoothing), 0.35, abs_tol=1e-7),
        )

    # 6. Combinations build must reject.
    # g3_pair: conv shapes of group 3's pair, the residual/output width, and a probe forward.
    for name, run in runs.items():
        hyp, net = run.state.hyp, run.state.net
        pair = hyp["g3_pair"]
        if pair == "full":
            continue
        g, w3 = net.layers[2], hyp["widths"][2]
        kind, inner = new.G3_PAIRS[pair]
        shapes = {
            "conv2": tuple(g.conv2.weight.shape),
            "conv3": tuple(g.conv3.weight.shape),
            "conv2a": tuple(g.conv2a.weight.shape)
            if getattr(g, "conv2a", None) is not None
            else None,
        }
        if kind == "bottleneck":
            expected = {
                "conv2": (inner, inner, 3, 3),
                "conv3": (w3, inner, 1, 1),
                "conv2a": (inner, w3, 1, 1),
            }
        else:
            expected = {"conv2": (inner, w3, 3, 3), "conv3": (w3, inner, 3, 3), "conv2a": None}
        with torch.inference_mode():
            probe = g(torch.randn(2, hyp["widths"][1], 6, 6))
        check(
            f"{name}: group 3 pair shapes {shapes} == {expected}; output width {w3}; "
            f"norm2 over {g.norm2.num_features} channels",
            shapes == expected
            and tuple(probe.shape) == (2, w3, 3, 3)
            and g.norm2.num_features == inner
            and g.norm3.num_features == w3,
        )
    check(
        "g3_pair full (default) registers no conv2a and the same state_dict keys as the control",
        getattr(d.state.net.layers[2], "conv2a", None) is None and set(d.weights) == set(b.weights),
    )

    rejected = {
        "g3_pair inner128 (unknown variant)": {"g3_pair": "inner128"},
        "g3_pair inner512 with 1x1 inner kernels in group 3": {
            "g3_pair": "inner512",
            "inner_kernels": [3, 3, 1],
        },
        "resolution_schedule [[30, 0.5]] (not 16/20/24/28)": {"resolution_schedule": [[30, 0.5]]},
        "batch_schedule [[0, 0.5]] (batch must be positive)": {"batch_schedule": [[0, 0.5]]},
        "an unknown parameter (our old 'jitter' switch)": {"jitter": 0.3},
        "aug_off_last 2.0 with 2 epochs (must be < epochs)": {"aug_off_last": 2.0},
        "skip_residual_until 0.25 without groups": {"skip_residual_until": 0.25},
        "skip_residual_groups [0] with skip_residual_until 0": {"skip_residual_groups": [0]},
        "skip_residual_groups [0, 0] (duplicate)": {
            "skip_residual_until": 0.25,
            "skip_residual_groups": [0, 0],
        },
        "skip_residual_groups [3] (not 0/1/2)": {
            "skip_residual_until": 0.25,
            "skip_residual_groups": [3],
        },
        "skip_compile turbo": {
            "skip_residual_until": 0.25,
            "skip_residual_groups": [0],
            "skip_compile": "turbo",
        },
        "label_smoothing_end 1.0 (must be < 1)": {"label_smoothing_end": 1.0},
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
