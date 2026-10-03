"""Profile the futurebiohackers recipe on the real dataset. Development only, runs on Modal.

Answers "where does the prepare+train time go, and how is the compute split across the
network?" without changing the recipe. It measures the EAGER fp16 network ("compile": ""
is forced: the per-block split needs profiler ranges, which torch.compile would fuse
away), so absolute times are a little slower than the compiled harness runs; the
proportions are what matter.

- build time and a prepare breakdown (reset, host-to-device copy + normalization,
  whitening init, flip + pad, EMA copy);
- per-epoch time of the data path (crop, flip, randperm) and of the steps;
- per step: forward, backward, optimizer step and lookahead EMA (CUDA events);
- forward device time per block (whitening stem, the three conv groups, head) from
  torch.profiler record_function ranges, plus the total backward kernel time;
- GPU busy fraction over a window of steps and the top kernels by device time;
- the evaluator-style forward over the test split.

Launched by scripts/modal_speedrun.py::profile or ::ab --profile (runs with PYTHONPATH set
to the harness checkout so `benchmark` imports). Prints a table and one line
"PROFILE_JSON {...}" for the launcher.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark.api import BuildContext
from benchmark.data import load_split
from benchmark.worker import load_submission, seed_everything

BLOCKS = ("stem_whiten_gelu", "group1", "group2", "group3", "head")


def sync() -> None:
    torch.cuda.synchronize()


def timed(fn) -> float:
    sync()
    start = time.perf_counter()
    fn()
    sync()
    return time.perf_counter() - start


def device_us(row) -> float:
    value = getattr(row, "self_device_time_total", None)
    if value is None:
        value = getattr(row, "self_cuda_time_total", 0.0)
    return float(value)


def range_us(row) -> float:
    value = getattr(row, "device_time_total", None)
    if value is None:
        value = getattr(row, "cuda_time_total", 0.0)
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/data")
    parser.add_argument("--submission-path", default="submissions/futurebiohackers")
    parser.add_argument("--params", default="{}")
    parser.add_argument("--epochs", type=int, default=2, help="epochs to time (schedule unchanged)")
    parser.add_argument("--profile-steps", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    params = {**json.loads(args.params), "compile": ""}  # eager: see the module docstring
    report: dict = {"params": params, "gpu": torch.cuda.get_device_name(0), "eager": True}

    module = load_submission(Path(args.submission_path).resolve())
    train = load_split(Path(args.data_root), train=True)
    test = load_split(Path(args.data_root), train=False)

    # ---- build and prepare, as the harness does -----------------------------------------
    holder: dict = {}

    def do_build() -> None:
        holder["state"] = module.build(BuildContext(device, params))

    report["build_s"] = round(timed(do_build), 3)
    state = holder["state"]
    seed_everything(0)
    report["prepare_s"] = round(timed(lambda: module.prepare(state, train, 0)), 3)
    hyp, net = state.hyp, state.net

    # Prepare breakdown: each piece re-run on its own, outside the timed prepare.
    pieces: dict = {}
    pieces["reset"] = timed(net.reset)

    def stage() -> None:
        raw = train.images.to(device, non_blocking=True).float().div_(255)
        mean = raw.mean(dim=(0, 2, 3), keepdim=True)
        std = raw.std(dim=(0, 2, 3), keepdim=True)
        holder["images"] = ((raw - mean) / std).to(state.dtype, memory_format=torch.channels_last)

    pieces["h2d_normalize"] = timed(stage)
    pieces["whitening_init"] = timed(lambda: net.init_whiten(holder["images"][:5000]))
    pieces["flip_pad"] = timed(
        lambda: F.pad(module.batch_flip_lr(holder["images"]), (hyp["translate"],) * 4, "reflect")
    )
    pieces["ema_copy"] = timed(lambda: torch._foreach_copy_(state.ema, state.float_state))
    report["prepare_pieces_s"] = {k: round(v, 4) for k, v in pieces.items()}
    seed_everything(0)
    module.prepare(state, train, 0)  # back to a clean trial state

    # ---- the training loop, instrumented (mirrors module._fit) ---------------------------------
    labels, batch_size = state.labels, state.batch_size
    steps_per_epoch, total_steps = state.steps_per_epoch, state.total_steps
    warmup_steps = int(total_steps * hyp["warmup"])
    ema_decay = 0.95**5 * (torch.arange(total_steps + 1) / total_steps) ** 3
    report.update(
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        epochs=hyp["epochs"],
    )
    optimizer = state.optimizer
    net.train()
    step = 0
    epoch_s: list[float] = []
    data_epoch_s: list[float] = []
    fwd_ms: list[float] = []
    bwd_ms: list[float] = []
    opt_ms: list[float] = []
    ema_ms: list[float] = []
    order = torch.randperm(len(labels), device=labels.device)

    def lr_scale(s: int) -> float:
        if s < warmup_steps:
            frac = s / warmup_steps
            return 0.2 * (1 - frac) + frac
        frac = (s - warmup_steps) / max(1, total_steps - warmup_steps)
        return (1 - frac) + hyp["final_lr"] * frac

    def one_step(images, step: int):
        e = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        i = step % steps_per_epoch
        idx = order[i * batch_size : (i + 1) * batch_size]
        e[0].record()
        outputs = state.train_net(images[idx], step < state.whiten_bias_steps)
        loss = F.cross_entropy(
            outputs.float(), labels[idx], label_smoothing=hyp["label_smoothing"], reduction="sum"
        )
        e[1].record()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        e[2].record()
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lr_scale(step)
        optimizer.step()
        e[3].record()
        if hyp["ema_every"] and (step + 1) % hyp["ema_every"] == 0:
            module._lookahead(state, ema_decay[min(step + 1, total_steps)].item())
        e[4].record()
        return e

    for epoch in range(args.epochs):
        sync()
        t0 = time.perf_counter()
        images = module.batch_crop(state.images, 32) if hyp["translate"] else state.images
        if epoch % 2 == 1:
            images = images.flip(-1)
        if hyp["cutout"]:
            images = module.batch_cutout(images, hyp["cutout"])
        order = torch.randperm(len(labels), device=labels.device)
        sync()
        data_epoch_s.append(time.perf_counter() - t0)
        events = []
        for _ in range(steps_per_epoch):
            events.append(one_step(images, step))
            step += 1
        sync()
        epoch_s.append(time.perf_counter() - t0)
        fwd_ms += [e[0].elapsed_time(e[1]) for e in events]
        bwd_ms += [e[1].elapsed_time(e[2]) for e in events]
        opt_ms += [e[2].elapsed_time(e[3]) for e in events]
        ema_ms += [e[3].elapsed_time(e[4]) for e in events]
    steady = statistics.mean(epoch_s[1:]) if len(epoch_s) > 1 else epoch_s[0]
    report.update(
        epoch_s=[round(e, 3) for e in epoch_s],
        data_epoch_s=[round(e, 4) for e in data_epoch_s],
        steady_epoch_s=round(steady, 3),
        projected_train_s=round(epoch_s[0] + steady * (hyp["epochs"] - 1), 2),
        fwd_ms_per_step=round(statistics.median(fwd_ms), 3),
        bwd_ms_per_step=round(statistics.median(bwd_ms), 3),
        opt_ms_per_step=round(statistics.median(opt_ms), 3),
        ema_ms_per_ema_step=round(max(ema_ms), 3),
        step_ms=round(
            statistics.median(fwd_ms) + statistics.median(bwd_ms) + statistics.median(opt_ms), 3
        ),
    )

    # ---- per-block forward split + GPU busy fraction (torch.profiler) ------------------------------
    active: dict = {}
    handles = []

    def hooks(name: str):
        def pre(_m, _inp) -> None:
            rf = torch.profiler.record_function(name)
            rf.__enter__()
            active[name] = rf

        def post(_m, _inp, _out) -> None:
            active.pop(name).__exit__(None, None, None)

        return pre, post

    targets = {
        "stem_whiten_gelu": net.layers[0],
        "group1": net.layers[1],
        "group2": net.layers[2],
        "group3": net.layers[3],
        "head": net.head,
    }
    for name, mod in targets.items():
        pre, post = hooks(name)
        handles += [mod.register_forward_pre_hook(pre), mod.register_forward_hook(post)]
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    images = module.batch_crop(state.images, 32) if hyp["translate"] else state.images
    order = torch.randperm(len(labels), device=labels.device)
    sync()
    with torch.profiler.profile(activities=activities) as prof:
        w0 = time.perf_counter()
        for _ in range(min(args.profile_steps, steps_per_epoch)):
            one_step(images, step)
            step += 1
        sync()
        wall_s = time.perf_counter() - w0
    for h in handles:
        h.remove()
    rows = prof.key_averages()
    busy_s = sum(device_us(r) for r in rows) / 1e6
    report.update(
        profiled_steps=min(args.profile_steps, steps_per_epoch),
        window_wall_s=round(wall_s, 3),
        gpu_busy_s=round(busy_s, 3),
        gpu_busy_fraction=round(busy_s / wall_s, 3) if wall_s else None,
    )
    split = {r.key: round(range_us(r) / 1e6, 4) for r in rows if r.key in BLOCKS}
    fwd_total = sum(split.values())
    report["forward_split_s"] = split
    report["forward_split_fraction"] = (
        {k: round(v / fwd_total, 3) for k, v in split.items()} if fwd_total else {}
    )
    # The whitening conv is called directly in forward() (not a submodule call), so the
    # stem range covers only the GELU; its conv shows up in the top kernels instead.
    bwd = sum(device_us(r) for r in rows if "ackward" in r.key) / 1e6
    report["backward_kernels_s"] = round(bwd, 4)
    top = sorted(rows, key=device_us, reverse=True)[:12]
    report["top_kernels"] = [
        {
            "name": r.key[:70],
            "calls": r.count,
            "device_ms": round(device_us(r) / 1e3, 2),
            "share": round(device_us(r) / 1e6 / busy_s, 3) if busy_s else None,
        }
        for r in top
    ]

    # ---- evaluator-style forward over the test split ---------------------------------------------
    classifier = state.classifier
    classifier.eval()
    test_images = test.images

    def evaluate() -> None:
        with torch.inference_mode():
            for batch in test_images.split(1024):
                classifier(batch.to(device=device, dtype=torch.float32).div_(255))

    report["eval_s"] = round(timed(evaluate), 3)
    classifier.train()

    # ---- print -------------------------------------------------------------------------------
    frac = report["forward_split_fraction"]
    print("\n=== profile (eager fp16) ===")
    print(f"GPU {report['gpu']}  params {params}")
    print(
        f"build {report['build_s']:.2f} s   prepare {report['prepare_s']:.3f} s  (pieces: "
        + ", ".join(f"{k} {v:.3f}" for k, v in pieces.items())
        + ")"
    )
    print(
        f"epochs timed {report['epoch_s']} s (data path per epoch {report['data_epoch_s']} s); "
        f"steady {steady:.3f} s/epoch; projected train {report['projected_train_s']} s for "
        f"{hyp['epochs']} epochs"
    )
    print(
        f"per step ({steps_per_epoch} steps/epoch, batch {batch_size}): forward "
        f"{report['fwd_ms_per_step']} ms + backward {report['bwd_ms_per_step']} ms + optimizer "
        f"{report['opt_ms_per_step']} ms = {report['step_ms']} ms; EMA step "
        f"{report['ema_ms_per_ema_step']} ms"
    )
    print(
        f"profiler window ({report['profiled_steps']} steps): wall {wall_s:.3f} s, GPU busy "
        f"{busy_s:.3f} s -> busy fraction {report['gpu_busy_fraction']}"
    )
    print(
        "forward device time per block: "
        + ", ".join(f"{k} {v:.4f} s ({frac.get(k, 0):.1%})" for k, v in split.items())
    )
    print(
        f"backward kernels total {bwd:.4f} s;  eval over {len(test_images)} test images: "
        f"{report['eval_s']} s"
    )
    print("top kernels by device time:")
    for k in report["top_kernels"]:
        print(f"  {k['device_ms']:9.2f} ms  {k['share']:6.1%}  x{k['calls']:<5d} {k['name']}")
    print("PROFILE_JSON " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
