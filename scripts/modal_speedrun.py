"""Run the CIFAR-100 speedrun harness on Modal (NVIDIA A100 80GB).

This file lives in the TEAM repo (scripts/), outside cifar100-speedrun/, so the
eventual upstream PR contains only submissions/futurebiohackers/.

Run everything from the team environment (Python 3.11 + modal), repo root:

    uv run modal run scripts/modal_speedrun.py::env_check
    uv run modal run scripts/modal_speedrun.py::download_data   # once: fills Volume cifar100-data
    uv run modal run scripts/modal_speedrun.py::smoke           # CPU + synthetic images, no GPU
    uv run modal run scripts/modal_speedrun.py::main --tag template-n1 --n 1 \
        --submission-path submission_template --no-accuracy-target
    uv run modal run scripts/modal_speedrun.py::main --tag mytag --n 3 \
        --submission futurebiohackers --params '{"epochs": 10}'
    uv run modal run scripts/modal_speedrun.py::sweep --tag mytag --n 1 \
        --params-list '[{"epochs": 5}, {"epochs": 10}]' --no-accuracy-target

Every run copies the whole results/<team>/<run_id>/ folder (summary.json,
trials.jsonl, config.json, source/, error.txt) back to
artifacts/speedrun_runs/<timestamp>_<tag>/ and prints a verdict plus a LOG.md row.

benchmark.run exit codes: 0 = complete and qualifying (or diagnostic run),
1 = complete but below the 75% target OR incomplete, 2 = invalid configuration.
Exit 1 is not a crash: summary.json ("complete", "qualified", "run_error") tells
the cases apart, and error.txt holds the worker traceback when there is one.

Image: by default an explicit uv image mirroring the organizer Dockerfile
(nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04, uv 0.10.8, `uv sync --frozen` on the
organizer uv.lock, Python 3.12.10). The speedrun code is mounted at container
start, so editing the recipe never rebuilds the image. SPEEDRUN_IMAGE=dockerfile
tries modal.Image.from_dockerfile on the organizer Dockerfile instead
(experimental: that Dockerfile uses BuildKit cache mounts and COPY --from, and
bakes the code into the image).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import modal

TEAM = "futurebiohackers"
REPO_ROOT = Path(__file__).resolve().parent.parent
SPEEDRUN_DIR = REPO_ROOT / "cifar100-speedrun"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "speedrun_runs"

IMAGE_MODE = os.environ.get("SPEEDRUN_IMAGE", "uv")  # "uv" (default) or "dockerfile"
GPU = "A100-80GB"  # never plain "A100": that can be a 40GB card
UV_VERSION = "0.10.8"  # same as the organizer Dockerfile
RUN_TIMEOUT = 8 * 60 * 60  # 40 trials x 600 s worst case is ~6.7 h
OFFICIAL_GPU = "NVIDIA A100 80GB PCIe"

APP_DIR = "/app"  # speedrun code, mounted at container start
LOCK_DIR = "/opt/speedrun-lock"  # pyproject.toml + uv.lock + .python-version, baked in
VENV = "/app/.venv" if IMAGE_MODE == "dockerfile" else "/opt/venv"
PY = f"{VENV}/bin/python"
DATA_ROOT = "/data"  # benchmark.data writes /data/cifar-100-python/
RESULTS_ROOT = "/results"

# Mirrors cifar100-speedrun/.dockerignore; keeps the local .venv (GBs) off the mount.
CODE_IGNORE = [
    "**/.venv/**",
    "**/data/**",
    "**/results/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.pytest_cache/**",
    "**/.ruff_cache/**",
    "**/.git/**",
    "**/.local/**",
    "**/seeds.json",
    "**/.env",
    "**/.env.*",
]


def _uv_image() -> modal.Image:
    """Same layers as the organizer Dockerfile, minus COPY . . (code is mounted)."""
    torch_check = (
        "import torch, torchvision; print('torch', torch.__version__, "
        "'torchvision', torchvision.__version__, 'cuda', torch.version.cuda)"
    )
    return (
        modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.12")
        .apt_install("ca-certificates", "build-essential", "git")
        .pip_install(f"uv=={UV_VERSION}")
        .env(
            {
                "UV_PYTHON_INSTALL_DIR": "/opt/python",
                "UV_LINK_MODE": "copy",
                "UV_PROJECT_ENVIRONMENT": VENV,
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "4",
            }
        )
        .add_local_file(SPEEDRUN_DIR / "pyproject.toml", f"{LOCK_DIR}/pyproject.toml", copy=True)
        .add_local_file(SPEEDRUN_DIR / "uv.lock", f"{LOCK_DIR}/uv.lock", copy=True)
        .add_local_file(SPEEDRUN_DIR / ".python-version", f"{LOCK_DIR}/.python-version", copy=True)
        .run_commands(
            f"cd {LOCK_DIR} && uv python install 3.12.10"
            " && uv sync --frozen --no-dev --no-install-project",
            f"{PY} -c {shlex.quote(torch_check)}",
        )
        .workdir(APP_DIR)
    )


def _dockerfile_image() -> modal.Image:
    return modal.Image.from_dockerfile(
        SPEEDRUN_DIR / "Dockerfile", context_dir=SPEEDRUN_DIR, add_python="3.12"
    )


image = (_dockerfile_image() if IMAGE_MODE == "dockerfile" else _uv_image()).add_local_dir(
    SPEEDRUN_DIR, APP_DIR, ignore=CODE_IGNORE
)

data_volume = modal.Volume.from_name("cifar100-data", create_if_missing=True)
results_volume = modal.Volume.from_name("cifar100-results", create_if_missing=True)

app = modal.App("cifar100-speedrun", image=image)


# --------------------------------------------------------------------------- remote


def _environment() -> dict:
    """Log GPU, software versions and CPUs (runs inside the container)."""
    info: dict = {}
    print("=== environment ===", flush=True)
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,power.limit,driver_version,mig.mode.current",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        print((smi.stdout or smi.stderr).strip(), flush=True)
        rows = [r.strip() for r in smi.stdout.strip().splitlines()]
        if smi.returncode == 0 and len(rows) >= 2:
            fields = [f.strip() for f in rows[1].split(",")]
            info.update(
                gpu_name=fields[0],
                gpu_memory=fields[1],
                gpu_power_limit=fields[2],
                driver=fields[3],
                mig=fields[4],
            )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"nvidia-smi unavailable: {exc}", flush=True)
    code = (
        "import sys, torch, torchvision; "
        "print('python', sys.version.split()[0], 'torch', torch.__version__, "
        "'torchvision', torchvision.__version__, 'cuda_runtime', torch.version.cuda, "
        "'cudnn', torch.backends.cudnn.version(), 'cuda_available', torch.cuda.is_available()); "
        "print('cuda_devices', [torch.cuda.get_device_name(i) "
        "for i in range(torch.cuda.device_count())])"
    )
    versions = subprocess.run([PY, "-c", code], capture_output=True, text=True)
    print((versions.stdout + versions.stderr).strip(), flush=True)
    info["versions"] = versions.stdout.strip()
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    print(
        f"cpus: os.cpu_count()={os.cpu_count()} sched_getaffinity={affinity} "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}",
        flush=True,
    )
    info.update(cpu_count=os.cpu_count(), cpu_affinity=affinity)
    name = info.get("gpu_name", "")
    if name and name != OFFICIAL_GPU:
        print(
            f"WARNING: GPU is '{name}', not the judges' '{OFFICIAL_GPU}' (300 W). "
            "An SXM part (~400 W) makes our timings slightly optimistic.",
            flush=True,
        )
    return info


def _run(spec: dict, results_root: str) -> dict:
    """Run benchmark.run once and return exit code plus the whole result folder."""
    env = _environment()
    cmd = [
        PY,
        "-m",
        "benchmark.run",
        "--n",
        str(spec.get("n", 1)),
        "--device",
        spec.get("device", "cuda"),
        "--data-root",
        DATA_ROOT,
        "--results-root",
        results_root,
        "--params",
        spec.get("params") or "{}",
    ]
    if spec.get("submission_path"):
        cmd += ["--submission-path", spec["submission_path"]]
    else:
        cmd += ["--submission", spec.get("submission") or TEAM]
    if spec.get("no_accuracy_target"):
        cmd.append("--no-accuracy-target")
    if spec.get("synthetic"):
        cmd.append("--synthetic")
    if spec.get("seed") is not None:
        cmd += ["--seed", str(spec["seed"])]
    print("$", shlex.join(cmd), flush=True)
    started = time.time()
    proc = subprocess.Popen(
        cmd, cwd=APP_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    result_dir = None
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        if line.startswith("Results: "):
            result_dir = line[len("Results: ") :].strip()
    returncode = proc.wait()
    files: dict[str, bytes] = {}
    if result_dir and Path(result_dir).is_dir():
        for path in sorted(Path(result_dir).rglob("*")):
            if path.is_file():
                files[path.relative_to(result_dir).as_posix()] = path.read_bytes()
    return {
        "returncode": returncode,
        "result_dir": result_dir,
        "files": files,
        "environment": env,
        "command": shlex.join(cmd),
        "wall_seconds": round(time.time() - started, 1),
        "spec": spec,
    }


@app.function(gpu=GPU, cpu=4, memory=8192, timeout=600)
def env_check() -> dict:
    """GPU + versions on the A100, no training."""
    return _environment()


@app.function(cpu=4, memory=8192, timeout=1800, volumes={DATA_ROOT: data_volume})
def download_data() -> None:
    """Download CIFAR-100 into the cifar100-data Volume. Run once; runs never download."""
    subprocess.run([PY, "-m", "benchmark.data", "--root", DATA_ROOT], cwd=APP_DIR, check=True)
    data_volume.commit()
    print("cifar100-data volume contains:", sorted(p.name for p in Path(DATA_ROOT).iterdir()))


@app.function(cpu=4, memory=8192, timeout=1800)
def smoke(submission: str = TEAM, submission_path: str = "") -> int:
    """CPU smoke test on synthetic images inside the Modal image (no GPU, no dataset)."""
    spec = {
        "n": 2,
        "device": "cpu",
        "synthetic": True,
        "submission": submission,
        "submission_path": submission_path,
    }
    payload = _run(spec, results_root="/tmp/results")
    summary = json.loads(payload["files"].get("summary.json", b"{}"))
    print(
        f"smoke: complete={summary.get('complete')} qualified={summary.get('qualified')} "
        f"exit={payload['returncode']} (expected: complete=True qualified=None exit=0)",
        flush=True,
    )
    return payload["returncode"]


@app.function(
    gpu=GPU,
    cpu=4,
    memory=16384,
    timeout=RUN_TIMEOUT,
    volumes={DATA_ROOT: data_volume, RESULTS_ROOT: results_volume},
)
def run_benchmark(spec: dict) -> dict:
    """One benchmark.run on the A100 with real CIFAR-100; results persist in the Volume."""
    payload = _run(spec, results_root=RESULTS_ROOT)
    results_volume.commit()
    return payload


# --------------------------------------------------------------------------- local


def _save_run(payload: dict, tag: str, suffix: str = "") -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", tag).strip("-") or "run"
    out = ARTIFACTS_DIR / f"{stamp}_{safe_tag}{suffix}"
    out.mkdir(parents=True, exist_ok=False)
    for rel, data in payload["files"].items():
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    meta = {k: v for k, v in payload.items() if k != "files"}
    (out / "modal_run.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}"


def _sec(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _report(payload: dict, out: Path, tag: str) -> bool:
    """Print the verdict for one run and a ready-to-paste LOG.md row."""
    rc = payload["returncode"]
    env = payload.get("environment", {})
    print(f"\n=== {out.relative_to(REPO_ROOT)} ===")
    error = out / "error.txt"
    if error.exists():
        print("--- error.txt ---")
        print(error.read_text())
    summary_path = out / "summary.json"
    if not summary_path.exists():
        print(
            f"benchmark.run exit code {rc}: no summary.json was produced "
            "(2 = invalid configuration; see the container log above)."
        )
        return False
    s = json.loads(summary_path.read_text())
    if s["qualified"] is True:
        verdict = "COMPLETE and QUALIFIED (>= 75%)"
    elif s["complete"] and s["qualified"] is None:
        verdict = "COMPLETE, diagnostic run (no accuracy target)"
    elif s["complete"]:
        verdict = "COMPLETE but BELOW the 75% target"
    else:
        verdict = f"INCOMPLETE ({s.get('run_error') or 'failed trials'})"
    print(f"exit code {rc}: {verdict}")
    print(
        f"trials {s['successful_trials']}/{s['requested_trials']}  "
        f"mean acc {_pct(s['mean_accuracy'])}% (std {_pct(s['accuracy_std'])})  "
        f"mean prep+train {_sec(s['mean_training_time'])} s "
        f"(std {_sec(s['training_time_std'])})  "
        f"mean eval {_sec(s['mean_evaluation_time'], 3)} s"
    )
    print(f"GPU: {env.get('gpu_name', '?')}  power limit {env.get('gpu_power_limit', '?')}")
    spec = payload.get("spec", {})
    config = spec.get("submission_path") or spec.get("submission") or TEAM
    if spec.get("params") and spec["params"] != "{}":
        config += " " + spec["params"]
    if spec.get("no_accuracy_target"):
        config += " (no target)"
    print("LOG.md row:")
    print(
        f"| {datetime.now():%Y-%m-%d} | {tag} | {config} | {s['requested_trials']} | "
        f"{_pct(s['mean_accuracy'])} | {_pct(s['accuracy_std'])} | "
        f"{_sec(s['mean_training_time'])} | {env.get('gpu_name', '?')} "
        f"({env.get('gpu_power_limit', '?')}) | {verdict} |"
    )
    return bool(s["complete"])


@app.local_entrypoint()
def main(
    tag: str = "dev",
    n: int = 1,
    submission: str = TEAM,
    submission_path: str = "",
    params: str = "{}",
    no_accuracy_target: bool = False,
    seed: int = -1,
):
    """Run benchmark.run on one A100 and copy the result folder to artifacts/speedrun_runs/."""
    if not isinstance(json.loads(params), dict):
        raise SystemExit("--params must be a JSON object")
    spec = {
        "n": n,
        "submission": submission,
        "submission_path": submission_path,
        "params": params,
        "no_accuracy_target": no_accuracy_target,
        "seed": None if seed < 0 else seed,
    }
    payload = run_benchmark.remote(spec)
    out = _save_run(payload, tag)
    _report(payload, out, tag)
    if payload["returncode"] != 0:
        sys.exit(payload["returncode"])


@app.local_entrypoint()
def sweep(
    tag: str = "sweep",
    n: int = 1,
    submission: str = TEAM,
    submission_path: str = "",
    params_list: str = "[]",
    no_accuracy_target: bool = False,
):
    """Run one A100 container per --params-list entry (in parallel) and save each result."""
    configs = json.loads(params_list)
    valid = isinstance(configs, list) and configs and all(isinstance(c, dict) for c in configs)
    if not valid:
        raise SystemExit("--params-list must be a non-empty JSON array of objects")
    specs = [
        {
            "n": n,
            "submission": submission,
            "submission_path": submission_path,
            "params": json.dumps(c),
            "no_accuracy_target": no_accuracy_target,
            "seed": None,
        }
        for c in configs
    ]
    print(f"sweep '{tag}': {len(specs)} configs x {n} trials, one {GPU} container each")
    failures = 0
    for i, payload in enumerate(run_benchmark.map(specs, order_outputs=True)):
        out = _save_run(payload, tag, f"_{i:02d}")
        if not _report(payload, out, f"{tag}_{i:02d}"):
            failures += 1
    if failures:
        sys.exit(1)
