"""Diagnostic only: profiles the frozen control; never produces a scored result."""

import json
import sys
from pathlib import Path

import torch
from benchmark.api import BuildContext
from benchmark.data import load_split
from benchmark.worker import load_submission, seed_everything


def main():
    output = Path(sys.argv[1])
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    module = load_submission(Path("/control"))
    state = module.build(BuildContext(torch.device("cuda"), {}))
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
