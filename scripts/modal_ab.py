"""Same-container A/B on one Modal A100: control, then each variant, same seeds.

    uv run modal run scripts/modal_ab.py --team devin --n 16 --seed 123 \
        --variants '[{"_team": "futurebiohackers"}, {"optimizer": "muon"}]'

`_team` in a variant picks another folder under cifar100-speedrun/submissions/.
Rows starting with "AB" summarise mean accuracy, sd and mean prep + training time.
"""

import json
import subprocess
import sys
from pathlib import Path

import modal

SPEEDRUN = Path(__file__).resolve().parents[1] / "cifar100-speedrun"
REMOTE = "/root/speedrun"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(SPEEDRUN))
    .env({"PYTHONPATH": REMOTE})
    .add_local_dir(
        SPEEDRUN, REMOTE, ignore=["**/.venv", "**/__pycache__", "data", "results", "**/.DS_Store"]
    )
)
app = modal.App("cifar100-ab", image=image)
data = modal.Volume.from_name("cifar100-data", create_if_missing=True)


@app.function(gpu="A100-80GB", cpu=(4, 4), timeout=6 * 60 * 60, volumes={"/data": data})
def ab(team: str, n: int, seed: int, variants: list[dict]) -> list[tuple]:
    if not Path("/data/cifar-100-python").exists():
        subprocess.run(
            [sys.executable, "-m", "benchmark.data", "--root", "/data"], cwd=REMOTE, check=True
        )
        data.commit()
    rows = []
    for i, params in enumerate([{}, *variants]):
        params = dict(params)
        sub = params.pop("_team", team)
        root = Path(f"/tmp/ab/{i:02d}")
        cmd = [
            sys.executable, "-m", "benchmark.run", "--data-root", "/data",
            "--submission", sub, "--n", str(n), "--seed", str(seed),
            "--no-accuracy-target", "--results-root", str(root), "--params", json.dumps(params),
        ]  # fmt: skip
        print(f"\n### variant {i}: {sub} {json.dumps(params)}", flush=True)
        subprocess.run(cmd, cwd=REMOTE, check=False)
        summaries = sorted(root.glob("*/*/summary.json"))
        s = json.loads(summaries[-1].read_text()) if summaries else {}
        label = {"_team": sub, **params} if sub != team else params
        rows.append(
            (
                label,
                s.get("mean_accuracy"),
                s.get("accuracy_std"),
                s.get("mean_training_time"),
                s.get("run_error"),
            )
        )
    return rows


@app.local_entrypoint()
def main(team: str = "devin", n: int = 16, seed: int = 0, variants: str = "[]"):
    rows = ab.remote(team, n, seed, json.loads(variants))
    print(f"\n=== AB RESULTS (variant 0 = {team} control, n={n}, seed={seed}) ===")
    for i, (p, acc, sd, t, err) in enumerate(rows):
        acc_s = f"{acc * 100:.2f}% ± {sd * 100:.2f}" if acc is not None else "n/a"
        t_s = f"{t:.3f}s" if t is not None else "n/a"
        print(f"AB {i:02d} | {acc_s} | {t_s} | {json.dumps(p)} | {err or ''}")
