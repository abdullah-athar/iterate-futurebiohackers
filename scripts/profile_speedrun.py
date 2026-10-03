"""Diagnostic only: profiles a selected recipe; never produces a scored result."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from benchmark.api import BuildContext
from benchmark.data import load_split
from benchmark.worker import load_submission, seed_everything


def main():
    output = Path(sys.argv[1])
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    source = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path("/root/speedrun/submissions/futurebiohackers")
    )
    parameters = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    module = load_submission(source)
    state = module.build(BuildContext(torch.device("cuda"), parameters))
    data = load_split(Path("/data"), train=True)
    seed_everything(42)
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(activities=activities) as prof:
        with torch.profiler.record_function("prepare"):
            module.prepare(state, data, 42)
            torch.cuda.synchronize()
        with torch.profiler.record_function("train"):
            module.train(state)
            torch.cuda.synchronize()
    prof.export_chrome_trace(str(output / "profile.json"))
    averages = prof.key_averages()
    # CUDA events count each actual kernel once; parent CPU scopes overlap them.
    kernels = defaultdict(lambda: {"calls": 0, "cuda_us": 0})
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA and event.name not in (
            "prepare",
            "train",
        ):
            kernels[event.name]["calls"] += 1
            kernels[event.name]["cuda_us"] += event.self_device_time_total
    kernel_rows = sorted(
        ({"key": key, **value} for key, value in kernels.items()),
        key=lambda row: row["cuda_us"],
        reverse=True,
    )
    (output / "profile-kernels.json").write_text(json.dumps(kernel_rows, indent=2))
    print(
        json.dumps(
            {
                "profile_source": str(source),
                "parameters": parameters,
                "total_kernel_ms": sum(r["cuda_us"] for r in kernel_rows) / 1000,
                "top_kernels": kernel_rows[:8],
            }
        ),
        flush=True,
    )
    (output / "profile.txt").write_text(
        averages.table(sort_by="self_cuda_time_total", row_limit=35)
    )
    (output / "profile-summary.json").write_text(
        json.dumps(
            [
                {
                    "key": e.key,
                    "calls": e.count,
                    "self_cuda_us": e.self_device_time_total,
                    "self_cpu_us": e.self_cpu_time_total,
                }
                for e in averages
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
