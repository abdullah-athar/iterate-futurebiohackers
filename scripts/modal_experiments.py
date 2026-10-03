"""Bounded, sequential experiments on one Modal GPU using the unchanged harness.

uv run modal run scripts/modal_experiments.py --stage screen --budget 20
uv run modal run scripts/modal_experiments.py --manifest artifacts/runtime-optimization/tune.json --budget 10
Each manifest entry is {"name": str, "params": dict, "n": int, "seed": int}.
The control is the frozen source from the reproduced pre-optimization baseline.
"""

import io
import itertools
import json
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
SPEEDRUN = ROOT / "cifar100-speedrun"
REMOTE = "/root/speedrun"
data = modal.Volume.from_name("cifar100-data", create_if_missing=True)
results = modal.Volume.from_name("cifar100-results", create_if_missing=True)
OUTPUT = ROOT / "artifacts/runtime-optimization"
CONTROL = OUTPUT / "control"
PR5_CONTROL = OUTPUT / "pr5_control"
PR10_CONTROL = OUTPUT / "pr10_control"
if modal.is_local() and not CONTROL.exists():
    CONTROL.mkdir(parents=True)
    baseline = subprocess.check_output(
        [
            "git",
            "show",
            "36f534b:cifar100-speedrun/submissions/futurebiohackers/submission.py",
        ],
        cwd=ROOT,
    )
    (CONTROL / "submission.py").write_bytes(baseline)
if modal.is_local() and not PR5_CONTROL.exists():
    PR5_CONTROL.mkdir(parents=True)
    baseline = subprocess.check_output(
        [
            "git",
            "show",
            "55d7931:cifar100-speedrun/submissions/futurebiohackers/submission.py",
        ],
        cwd=ROOT,
    )
    (PR5_CONTROL / "submission.py").write_bytes(baseline)
if modal.is_local() and not PR10_CONTROL.exists():
    PR10_CONTROL.mkdir(parents=True)
    baseline = subprocess.check_output(
        [
            "git",
            "show",
            "bcf5a0e:cifar100-speedrun/submissions/futurebiohackers/submission.py",
        ],
        cwd=ROOT,
    )
    (PR10_CONTROL / "submission.py").write_bytes(baseline)
# $5/hour conservatively exceeds current A100 + four CPUs + 32 GiB RAM pricing.
RATE = 5 / 3600
cache = modal.Volume.from_name("cifar100-compile-cache", create_if_missing=True)
experiment_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(SPEEDRUN))
    .env(
        {
            "PYTHONPATH": REMOTE,
            "TORCHINDUCTOR_CACHE_DIR": "/compile/inductor",
            "TRITON_CACHE_DIR": "/compile/triton",
        }
    )
    .add_local_dir(
        SPEEDRUN,
        REMOTE,
        ignore=["**/.venv", "**/__pycache__", "data", "results", "**/.DS_Store"],
    )
    .add_local_file(CONTROL / "submission.py", "/control/submission.py")
    .add_local_file(PR5_CONTROL / "submission.py", "/pr5_control/submission.py")
    .add_local_file(PR10_CONTROL / "submission.py", "/pr10_control/submission.py")
    .add_local_file(ROOT / "scripts/profile_speedrun.py", "/profile_speedrun.py")
    .add_local_file(
        ROOT / "scripts/diagnose_speedrun_muon.py", "/diagnose_speedrun_muon.py"
    )
    .add_local_file(
        ROOT / "scripts/check_speedrun_kernels.py", "/check_speedrun_kernels.py"
    )
)
app = modal.App("cifar100-experiments", image=experiment_image)


def screen_manifest():
    items = [
        {"name": "control", "control": True, "n": 3},
        {"name": "indexed-baseline", "params": {"crop_mode": "indexed"}, "n": 3},
    ]
    for widths, depths, resolution in itertools.product(
        ([96, 256, 576], [128, 320, 576]), ([2, 3, 3], [3, 3, 3]), (32, 24, 28)
    ):
        items.append(
            {
                "name": f"w{widths[0]}-{widths[1]}-d{depths[0]}-r{resolution}",
                "params": {
                    "widths": widths,
                    "depths": depths,
                    "train_resolution": resolution,
                    "crop_mode": "indexed",
                },
                "n": 1,
            }
        )
    return items


@app.function(
    gpu="A100-80GB",
    cpu=(4, 4),
    memory=(16384, 32768),
    timeout=14400,
    scaledown_window=2,
    volumes={"/data": data, "/results": results, "/compile": cache},
)
def experiment(items: list[dict], budget: float, profile: bool, diagnose_muon: bool):
    started = time.monotonic()
    deadline = started + budget / RATE - 120
    session = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    folder = Path("/results/experiments") / session
    folder.mkdir(parents=True)
    completed = []
    # Isolate summaries from concurrent stages sharing the results volume.
    run_root = folder / "runs"
    run_root.mkdir()
    checks = [
        sys.executable,
        "/check_speedrun_kernels.py",
        str(folder / "kernel-checks.txt"),
    ]
    subprocess.run(
        checks,
        cwd=REMOTE,
        check=True,
        timeout=120,
    )
    if profile:
        subprocess.run(
            [
                sys.executable,
                "/profile_speedrun.py",
                str(folder),
                f"{REMOTE}/submissions/futurebiohackers",
            ],
            cwd=REMOTE,
            timeout=min(1000, deadline - time.monotonic()),
            check=False,
        )
    if diagnose_muon:
        subprocess.run(
            [sys.executable, "/diagnose_speedrun_muon.py", str(folder)],
            cwd=REMOTE,
            timeout=min(1000, deadline - time.monotonic()),
            check=False,
        )
    for item in items:
        # Reserve a full build plus trial; never start work near the stage cap.
        remaining = deadline - time.monotonic()
        if remaining < 750:
            print(
                "Budget guard: remaining stage allowance is too small for another build.",
                flush=True,
            )
            break
        args = [
            sys.executable,
            "-m",
            "benchmark.run",
            "--data-root",
            "/data",
            "--results-root",
            str(run_root),
            "--n",
            str(item.get("n", 1)),
            "--seed",
            str(item.get("seed", 100)),
            "--params",
            json.dumps(item.get("params", {})),
        ]
        reference = {"pr5": "pr5_control", "pr10": "pr10_control"}.get(
            item.get("reference"), "control"
        )
        args += (
            ["--submission-path", f"/{reference}"]
            if item.get("control")
            else ["--submission", "futurebiohackers"]
        )
        team_results = run_root / (
            reference if item.get("control") else "futurebiohackers"
        )
        old = set(team_results.glob("*"))
        print(f"EXPERIMENT {item['name']} {item.get('params', {})}", flush=True)
        with (folder / f"{item['name']}.log").open("w") as log:
            process = subprocess.Popen(
                args,
                cwd=REMOTE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            def relay(process=process, item=item, log=log):
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith("trial "):
                        print(
                            "TRIAL_PROGRESS "
                            + json.dumps({"name": item["name"], "line": line.strip()}),
                            flush=True,
                        )

            reader = threading.Thread(target=relay, daemon=True)
            reader.start()
            try:
                code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = 124
            reader.join(timeout=10)
        new = set(team_results.glob("*")) - old
        summary_path = next(iter(new)) / "summary.json" if len(new) == 1 else None
        summary = (
            json.loads(summary_path.read_text())
            if summary_path and summary_path.exists()
            else {}
        )
        record = {
            **item,
            "exit_code": code,
            "summary": summary,
            "result_path": str(summary_path.parent) if summary_path else None,
        }
        completed.append(record)
        (folder / "experiments.json").write_text(json.dumps(completed, indent=2))
        print(json.dumps(record), flush=True)
        results.commit()
        if not summary.get("complete", False):
            print(
                "Stopping stage after an incomplete run; fix the failure before resuming.",
                flush=True,
            )
            break
    charged_bound = (time.monotonic() - started + 120) * RATE
    (folder / "cost-bound.json").write_text(
        json.dumps(
            {"conservative_cost_bound": charged_bound, "stage_budget": budget}, indent=2
        )
    )
    results.commit()
    cache.commit()
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        tar.add(folder, arcname=f"experiments/{session}")
    return charged_bound, archive.getvalue()


@app.local_entrypoint()
def main(
    stage: str = "screen",
    manifest: str = "",
    budget: float = 20,
    profile: bool = False,
    diagnose_muon: bool = False,
):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    journal = OUTPUT / "budget.json"
    ledger = (
        json.loads(journal.read_text())
        if journal.exists()
        else {"spent_bound": 1.0, "runs": []}
    )
    if budget <= 0 or budget > 50 - ledger["spent_bound"]:
        raise ValueError(
            f"Budget exceeds remaining allowance: {50 - ledger['spent_bound']:.2f}"
        )
    items = json.loads(Path(manifest).read_text()) if manifest else screen_manifest()
    # Reserve before launch; a lost connection cannot erase its budget allocation.
    ledger["spent_bound"] += budget
    ledger["runs"].append({"stage": stage, "reserved": budget})
    journal.write_text(json.dumps(ledger, indent=2))
    cost, archive = experiment.remote(items, budget, profile, diagnose_muon)
    ledger = json.loads(journal.read_text())
    ledger["spent_bound"] -= budget - cost
    reservation = next(
        r for r in ledger["runs"] if r["stage"] == stage and "cost_bound" not in r
    )
    reservation["cost_bound"] = cost
    journal.write_text(json.dumps(ledger, indent=2))
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(OUTPUT, filter="data")
    print(
        f"Conservative cumulative spend: ${ledger['spent_bound']:.2f}; artifacts: {OUTPUT}"
    )
