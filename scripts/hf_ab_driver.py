"""Run inside an HF Job: one benchmark.run per variant, same seeds, same card."""

import json
import os
import subprocess
import sys
from pathlib import Path

team, n, variants = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
rows = []
for i, params in enumerate([{}] + variants):
    params = dict(params)
    sub = params.pop("_team", team)
    root = Path(f"/app/ab/{i:02d}")
    cmd = [
        "/opt/venv/bin/python", "-m", "benchmark.run", "--submission", sub, "--n", n,
        "--seed", os.environ.get("SEED", "0"), "--no-accuracy-target", "--results-root", str(root),
        "--params", json.dumps(params),
    ]
    print(f"\n### variant {i}: {json.dumps(params)}", flush=True)
    subprocess.run(cmd, cwd="/app", check=False)
    summaries = sorted(root.glob("*/*/summary.json"))
    s = json.loads(summaries[-1].read_text()) if summaries else {}
    label = {"_team": sub, **params} if sub != team else params
    rows.append((label, s.get("mean_accuracy"), s.get("accuracy_std"),
                 s.get("mean_training_time"), s.get("run_error")))
print("\n=== AB RESULTS (variant 0 = control) ===")
for i, (p, acc, sd, t, err) in enumerate(rows):
    acc_s = f"{acc * 100:.2f}% ± {sd * 100:.2f}" if acc is not None else "n/a"
    t_s = f"{t:.3f}s" if t is not None else "n/a"
    print(f"AB {i:02d} | {acc_s} | {t_s} | {json.dumps(p)} | {err or ''}")
