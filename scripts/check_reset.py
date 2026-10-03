"""Throwaway reset check for submissions/futurebiohackers. Not part of the submission.

Builds once, then runs prepare + train for two different seeds on the SAME state object
and checks that prepare really starts from scratch (weights, BatchNorm statistics,
gradients, optimizer state), that a reused state behaves exactly like a fresh build,
and that repeating a seed reproduces a trial bit for bit on CPU.

Run from the speedrun env with cwd cifar100-speedrun (synthetic images, CPU, ~1 min):
    uv run python ../scripts/check_reset.py                     # Linux/macOS
    scripts/wsl_speedrun.sh python ../scripts/check_reset.py    # Windows, via WSL
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from benchmark.api import BuildContext
from benchmark.data import synthetic_split
from benchmark.worker import load_submission, seed_everything

# Speed only. The recipe's own defaults (40 epochs, width 64) are not touched.
PARAMS = {"epochs": 2, "width": 16}
RESET_BUFFERS = ("mean", "std")  # recomputed from the data each trial, so identical across trials

results: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    results.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)


def snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def differing(a: dict, b: dict) -> list[str]:
    return [k for k in a if not torch.equal(a[k], b[k])]


def learnable(model: torch.nn.Module) -> list[str]:
    return [name for name, _ in model.named_parameters()]


def randomly_initialised(model: torch.nn.Module) -> list[str]:
    """Conv/Linear weights: seed-dependent init. BatchNorm affine params start at 1/0 always."""
    return [
        f"{prefix}.{name}" if prefix else name
        for prefix, module in model.named_modules()
        if isinstance(module, torch.nn.Conv2d | torch.nn.Linear)
        for name, _ in module.named_parameters(recurse=False)
    ]


def batchnorm_is_fresh(model: torch.nn.Module) -> bool:
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            if not (
                torch.equal(module.running_mean, torch.zeros_like(module.running_mean))
                and torch.equal(module.running_var, torch.ones_like(module.running_var))
                and int(module.num_batches_tracked) == 0
                and torch.equal(module.weight, torch.ones_like(module.weight))
                and torch.equal(module.bias, torch.zeros_like(module.bias))
            ):
                return False
    return True


def main() -> int:
    torch.set_num_threads(4)
    speedrun = Path(__file__).resolve().parents[1] / "cifar100-speedrun"
    module = load_submission(speedrun / "submissions" / "futurebiohackers")
    data = synthetic_split(train=True)
    original_images, original_labels = data.images.clone(), data.labels.clone()
    context = BuildContext(torch.device("cpu"), PARAMS)

    state = module.build(context)
    after_build = snapshot(state.model)
    params = learnable(state.model)
    check(
        "build ran a synthetic warmup (optimizer state is non-empty)",
        len(state.optimizer.state) > 0,
    )
    check(
        "build left BatchNorm statistics changed by the warmup", not batchnorm_is_fresh(state.model)
    )

    # Trial 1: seed 42 -------------------------------------------------------------
    seed_everything(42)
    module.prepare(state, data, 42)
    after_prepare_1 = snapshot(state.model)
    changed = differing(after_build, after_prepare_1)
    check(
        "prepare #1 re-initialised every learnable parameter touched by the warmup",
        all(p in changed for p in params),
    )
    check(
        "prepare #1 reset BatchNorm (running stats, counters, affine params)",
        batchnorm_is_fresh(state.model),
    )
    check("prepare #1 created an optimizer with EMPTY state", len(state.optimizer.state) == 0)
    check("prepare #1 cleared gradients", all(p.grad is None for p in state.model.parameters()))
    optimizer_1 = state.optimizer
    model = module.train(state)
    check("train returned the eager nn.Module held in state", model is state.model)
    after_train_1 = snapshot(model)
    check(
        "train #1 changed every learnable parameter",
        all(p in differing(after_prepare_1, after_train_1) for p in params),
    )
    check(
        "train #1 populated the optimizer state (momentum buffers)", len(state.optimizer.state) > 0
    )

    # Trial 2: seed 43, same state object --------------------------------------------
    seed_everything(43)
    module.prepare(state, data, 43)
    after_prepare_2 = snapshot(state.model)
    check(
        "prepare #2: every learnable parameter differs from the weights after train #1",
        all(p in differing(after_train_1, after_prepare_2) for p in params),
    )
    check(
        "prepare #2: every Conv/Linear weight differs from prepare #1 (different seed)",
        all(p in differing(after_prepare_1, after_prepare_2) for p in randomly_initialised(model)),
    )
    check(
        "prepare #2: normalization buffers equal prepare #1 (same data, recomputed)",
        all(
            f"{b}" in after_prepare_1 and torch.equal(after_prepare_1[b], after_prepare_2[b])
            for b in RESET_BUFFERS
        ),
    )
    check("prepare #2 built a NEW optimizer object", state.optimizer is not optimizer_1)
    check("prepare #2 optimizer state is EMPTY", len(state.optimizer.state) == 0)
    check("prepare #2 reset BatchNorm", batchnorm_is_fresh(state.model))
    check("prepare #2 cleared gradients", all(p.grad is None for p in state.model.parameters()))
    after_train_2 = snapshot(module.train(state))

    # History independence: a brand-new build with the same seed must match the reused state.
    fresh = module.build(context)
    seed_everything(43)
    module.prepare(fresh, data, 43)
    check(
        "prepare on the reused state == prepare on a fresh build (seed 43)",
        not differing(snapshot(fresh.model), after_prepare_2),
    )
    check(
        "full trial on the reused state == full trial on a fresh build (seed 43)",
        not differing(snapshot(module.train(fresh)), after_train_2),
    )

    # Determinism: repeating seed 42 on the twice-used state reproduces trial 1 exactly.
    seed_everything(42)
    module.prepare(state, data, 42)
    check(
        "repeating seed 42 reproduces trial #1 bit for bit (CPU)",
        not differing(snapshot(module.train(state)), after_train_1),
    )

    check(
        "training tensors were not modified",
        torch.equal(data.images, original_images) and torch.equal(data.labels, original_labels),
    )

    failed = [name for name, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
