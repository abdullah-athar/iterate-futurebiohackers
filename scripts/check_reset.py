"""Reset check for submissions/futurebiohackers (airbench-style recipe). Not part of the submission.

Builds once, then runs prepare + train for two different seeds on the SAME state object and
checks that prepare really starts from scratch (every parameter, BatchNorm statistics,
optimizer state, lookahead EMA buffers), that a reused state behaves exactly like a fresh
build, and that repeating a seed reproduces a trial bit for bit on CPU.

Run from the speedrun env with cwd cifar100-speedrun (synthetic images, CPU, about a minute):
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

# Speed only (narrow network, two epochs, eager). The recipe's own defaults are not touched.
# The synthetic split has 64 images: batch 8 gives 16 steps, so the lookahead EMA (every 5
# steps) really updates and the final EMA copy does not just restore the initial weights.
PARAMS = {"widths": [32, 64, 64], "epochs": 2.0, "batch_size": 8, "compile": ""}

results: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail: list[str] | None = None) -> None:
    results.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)
    if not ok and detail:
        print(f"      offending: {detail}", flush=True)


def snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def differing(a: dict, b: dict) -> list[str]:
    return [k for k in a if not torch.equal(a[k], b[k])]


def trainable(model: torch.nn.Module) -> list[str]:
    return [name for name, p in model.named_parameters() if p.requires_grad]


def random_init(model: torch.nn.Module) -> list[str]:
    """Weights with a seed-dependent init: Linear layers and convs with more output than input
    channels. The recipe's Conv sets the first `in` output filters to a dirac (identity)
    kernel, so a square conv is fully deterministic; BatchNorm biases start at zero."""
    names = []
    for prefix, module in model.named_modules():
        linear = isinstance(module, torch.nn.Linear)
        tall_conv = isinstance(module, torch.nn.Conv2d) and (
            module.out_channels > module.in_channels
        )
        if (linear or tall_conv) and module.weight.requires_grad:
            names.append(f"{prefix}.weight" if prefix else "weight")
    return names


def batchnorm_is_fresh(model: torch.nn.Module) -> bool:
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            if not (
                torch.equal(module.running_mean, torch.zeros_like(module.running_mean))
                and torch.equal(module.running_var, torch.ones_like(module.running_var))
                and int(module.num_batches_tracked) == 0
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
    net = state.net

    # Trial 1: seed 42 -------------------------------------------------------------
    seed_everything(42)
    module.prepare(state, data, 42)
    model = module.train(state)
    check("train returned an nn.Module", isinstance(model, torch.nn.Module))
    after_train_1 = snapshot(model)
    check(
        "train #1 populated the optimizer state (momentum buffers)",
        len(state.optimizers[0].state) > 0,
    )
    optimizer_1 = state.optimizers[0]

    # Trial 2: seed 43, same state object --------------------------------------------
    seed_everything(43)
    module.prepare(state, data, 43)
    after_prepare_2 = snapshot(model)
    params = trainable(model)
    changed = differing(after_train_1, after_prepare_2)
    check(
        "prepare #2: every trainable parameter differs from the weights after train #1",
        all(p in changed for p in params),
        [p for p in params if p not in changed],
    )
    check("prepare #2 reset BatchNorm (running stats, counters, biases)", batchnorm_is_fresh(model))
    check(
        "prepare #2 built a NEW optimizer with EMPTY state",
        state.optimizers[0] is not optimizer_1 and len(state.optimizers[0].state) == 0,
    )
    # Gradients are not reset in prepare, and need not be: _fit zeroes them before every
    # backward, and the "reused state == fresh build" check below proves they cannot leak.
    check(
        "prepare #2: lookahead EMA buffers equal the fresh parameters (no carry-over)",
        all(torch.equal(e, p) for e, p in zip(state.ema, state.float_state, strict=True)),
    )
    check(
        "prepare #2: whitening bias is zero again",
        torch.equal(net.whiten.bias, torch.zeros_like(net.whiten.bias)),
    )
    after_train_2 = snapshot(module.train(state))

    # History independence: a brand-new build with the same seed must match the reused state.
    fresh = module.build(context)
    seed_everything(43)
    module.prepare(fresh, data, 43)
    fresh_model = module.train(fresh)
    check(
        "full trial on the reused state == full trial on a fresh build (seed 43)",
        not differing(snapshot(fresh_model), after_train_2),
    )

    # Different seeds give different inits; repeating a seed reproduces a trial exactly.
    seed_everything(42)
    module.prepare(state, data, 42)
    after_prepare_1b = snapshot(model)
    seed_dependent = random_init(model)
    seed_changed = differing(after_prepare_1b, after_prepare_2)
    check(
        "different seeds give different inits for every seed-dependent Conv/Linear weight",
        bool(seed_dependent) and all(w in seed_changed for w in seed_dependent),
        [w for w in seed_dependent if w not in seed_changed],
    )
    check(
        "repeating seed 42 reproduces trial #1 bit for bit (CPU)",
        not differing(snapshot(module.train(state)), after_train_1),
    )

    # Whitening statistics come from the training data passed to prepare, nothing else.
    check(
        "whitening weights are recomputed from the data in prepare (same data -> identical)",
        torch.equal(after_prepare_2["net.whiten.weight"], after_prepare_1b["net.whiten.weight"]),
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
