"""Run the CIFAR-100 speedrun harness on Modal (NVIDIA A100 80GB PCIe).

This file lives in the TEAM repo (scripts/), outside cifar100-speedrun/, so the
eventual upstream PR contains only submissions/futurebiohackers/.

Run everything from the team environment (Python 3.11 + modal), repo root
(Windows: set PYTHONUTF8=1 first):

    uv run modal run scripts/modal_speedrun.py::env_check
    uv run modal run scripts/modal_speedrun.py::download_data   # once: fills Volume cifar100-data
    uv run modal run scripts/modal_speedrun.py::smoke           # CPU + synthetic images, no GPU
    uv run modal run scripts/modal_speedrun.py::main --tag baseline-n2 --n 2
    uv run modal run scripts/modal_speedrun.py::main --tag template-n1 --n 1 \
        --submission-path submission_template --no-accuracy-target
    uv run modal run scripts/modal_speedrun.py::ab --tag compile --n 2 \
        --variants '[{"use_compile": true}, {"use_compile": true, "compile_mode": "max-autotune"}]'
    uv run modal run scripts/modal_speedrun.py::fetch --run-id 20261003T120042Z-f6dae59d
    uv run modal run scripts/modal_speedrun.py::sweep --tag mytag --n 1 \
        --params-list '[{"epochs": 5}, {"epochs": 10}]' --no-accuracy-target  # gated, see below

PCIe guard (on by default for main and ab, --no-require-pcie to disable): Modal's
A100-80GB pool mixes PCIe (300 W, the judges' card) and SXM4 (500 W) parts. The
container checks nvidia-smi BEFORE build; on a non-PCIe card it returns at once
(about 0.35 GPU-min with the start-up allowance) and the launcher retries, up to
SPEEDRUN_PCIE_ATTEMPTS calls in total (default 8). GPU functions are single-use
containers, so a retry is never served by the container that just failed. Every
attempt is logged with the GPU name, Modal task id, region and cloud, so a region
that hands out PCIe cards can later be pinned with SPEEDRUN_REGION.

A/B mode (::ab): the control (the submission's defaults, or --control-params) and
every --variants entry (a JSON list of parameter deltas merged over the control)
or --submission-paths entry run SEQUENTIALLY in ONE container on ONE card, each as
its own benchmark.run process (fresh build, fresh worker, cold torch.compile
cache, nothing shared). The launcher prints a comparison table and saves
artifacts/speedrun_runs/<ts>_<tag>/NN_<label>/ plus ab_summary.{md,json}.

Time caps: every GPU function has a hard Modal timeout (SPEEDRUN_TIMEOUT_MIN,
default 15 min) so a hung run cannot burn more than that; the harness's own
limits are --build-timeout (default here 300 s, the 5-minute compile rule) and
600 s per trial. Before launching, the launcher estimates the container time
from the specs and refuses launches that cannot fit. Inside the container each
variant gets a deadline: the harness is interrupted with SIGINT (it keeps the
finished trials) and variants that no longer fit are skipped, so the payload
always comes back before Modal kills the container. If Modal does kill it,
::fetch --run-id <id> copies a finished run out of the cifar100-results Volume.

GPU budget: each container's wall time plus a start-up allowance is appended to
artifacts/speedrun_runs/gpu_ledger.jsonl and the running total is printed before
and after every call ("GPU used: 12.3/90 min"). SPEEDRUN_GPU_BUDGET_MIN (90) is a
hard stop, SPEEDRUN_GPU_STOP_MIN (60) a checkpoint that needs
SPEEDRUN_GPU_CONTINUE=1 to pass. Failed and timed-out calls are logged too.

Results: every harness run copies results/<team>/<run_id>/ (summary.json,
trials.jsonl, config.json, source/, error.txt) back under artifacts/speedrun_runs/.
benchmark.run exit codes: 0 = complete and qualifying (or diagnostic run),
1 = complete but below the 75% target OR incomplete, 2 = invalid configuration,
130 = interrupted (partial results kept). Exit 1 is not a crash: summary.json
("complete", "qualified", "run_error") tells the cases apart, and error.txt holds
the worker traceback when there is one.

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
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import modal
import modal.exception
from modal.volume import FileEntryType

TEAM = "futurebiohackers"
REPO_ROOT = Path(__file__).resolve().parent.parent
SPEEDRUN_DIR = REPO_ROOT / "cifar100-speedrun"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "speedrun_runs"
LEDGER_PATH = ARTIFACTS_DIR / "gpu_ledger.jsonl"

IMAGE_MODE = os.environ.get("SPEEDRUN_IMAGE", "uv")  # "uv" (default) or "dockerfile"
GPU = "A100-80GB"  # never plain "A100": that can be a 40GB card
UV_VERSION = "0.10.8"  # same as the organizer Dockerfile
# Hard cap per GPU container: a hung run cannot burn more than this. The harness's own
# per-trial limit is 600 s. Raise SPEEDRUN_TIMEOUT_MIN only for an approved run.
RUN_TIMEOUT = int(os.environ.get("SPEEDRUN_TIMEOUT_MIN", "15")) * 60
DEADLINE_MARGIN_S = 60.0  # the container stops starting work this long before RUN_TIMEOUT
MIN_RUN_S = 90.0  # do not start a variant with less container time left than this
GPU_BUDGET_MIN = float(os.environ.get("SPEEDRUN_GPU_BUDGET_MIN", "90"))
GPU_STOP_MIN = float(os.environ.get("SPEEDRUN_GPU_STOP_MIN", "60"))
CONTAINER_START_S = 15.0  # billed start-up allowance per container, not measurable inside it
ENV_CHECK_S = 20.0  # nvidia-smi + torch import at the start of every container (estimate)
HARNESS_OVERHEAD_S = 15.0  # process start, dataset load, eval and copy per harness run (estimate)
OFFICIAL_GPU = "NVIDIA A100 80GB PCIe"
PCIE_ATTEMPTS = int(os.environ.get("SPEEDRUN_PCIE_ATTEMPTS", "8"))  # calls before giving up
DEFAULT_BUILD_TIMEOUT = 300.0  # the 5-minute rule for torch.compile builds (harness default 600)
# Optional placement pins for the GPU functions (Modal cloud=/region=), e.g. SPEEDRUN_CLOUD=azure
# SPEEDRUN_REGION=us-east: used to find and then pin the pool that hands out PCIe cards.
PLACEMENT = {
    key: value
    for key, value in (
        ("cloud", os.environ.get("SPEEDRUN_CLOUD")),
        ("region", os.environ.get("SPEEDRUN_REGION")),
    )
    if value
}
SCHEDULE_WAIT_S = float(
    os.environ.get("SPEEDRUN_SCHEDULE_WAIT_S", "240")
)  # pinned pools can lack capacity

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


def _safe(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", tag).strip("-") or "run"


# --------------------------------------------------------------------------- remote


def _environment() -> dict:
    """Log GPU, software versions, CPUs and container identity (runs inside the container)."""
    info: dict = {
        "task_id": os.environ.get("MODAL_TASK_ID"),
        "hostname": socket.gethostname(),
        "region": os.environ.get("MODAL_REGION"),
        "cloud": os.environ.get("MODAL_CLOUD_PROVIDER"),
    }
    print("=== environment ===", flush=True)
    print(
        f"container task_id={info['task_id']} host={info['hostname']} "
        f"region={info['region']} cloud={info['cloud']}",
        flush=True,
    )
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
            if len(fields) >= 5:
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
    try:
        versions = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=60)
        print((versions.stdout + versions.stderr).strip(), flush=True)
        info["versions"] = versions.stdout.strip()
    except subprocess.SubprocessError as exc:
        print(f"torch version check failed: {exc}", flush=True)
        info["versions"] = f"failed: {exc}"
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
            "An SXM part (500 W) makes our timings optimistic.",
            flush=True,
        )
    return info


def _run(spec: dict, results_root: str, deadline: float | None = None) -> dict:
    """One benchmark.run process (its own build and worker); returns the result folder.

    With a deadline (time.time() value), the harness gets SIGINT at that moment, which
    makes it stop the worker, keep the finished trials and exit 130; SIGKILL follows
    45 s later if needed.
    """
    label = spec.get("label", "run")
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
    if spec.get("build_timeout"):
        cmd += ["--build-timeout", str(spec["build_timeout"])]
    if spec.get("eval_timeout"):
        cmd += ["--eval-timeout", str(spec["eval_timeout"])]
    env = dict(os.environ)
    if spec.get("torch_logs"):
        env["TORCH_LOGS"] = spec["torch_logs"]  # e.g. "recompiles" to spot torch.compile recompiles
    # Cold compile caches for every run: a later compiled variant must not inherit the
    # kernels or autotune results of an earlier one (the judges' container is cold too).
    cache_dir = tempfile.mkdtemp(prefix=f"inductor-{_safe(label)}-")
    env["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    env["TRITON_CACHE_DIR"] = f"{cache_dir}/triton"
    print(f"\n##### {label} #####\n$ {shlex.join(cmd)}", flush=True)
    started = time.time()
    proc = subprocess.Popen(
        cmd, cwd=APP_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    flags = {"interrupted": False}
    timers: list[threading.Timer] = []
    if deadline is not None:
        remaining = max(1.0, deadline - time.time())

        def interrupt() -> None:
            flags["interrupted"] = True
            print(
                f"\n##### {label}: container deadline after {remaining:.0f} s, sending SIGINT "
                "to benchmark.run (finished trials are kept)",
                flush=True,
            )
            proc.send_signal(signal.SIGINT)

        timers = [threading.Timer(remaining, interrupt), threading.Timer(remaining + 45, proc.kill)]
        for timer in timers:
            timer.start()
    result_dir = None
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        if line.startswith("Results: "):
            result_dir = line[len("Results: ") :].strip()
    returncode = proc.wait()
    for timer in timers:
        timer.cancel()
    shutil.rmtree(cache_dir, ignore_errors=True)
    files: dict[str, bytes] = {}
    if result_dir and Path(result_dir).is_dir():
        for path in sorted(Path(result_dir).rglob("*")):
            if path.is_file():
                files[path.relative_to(result_dir).as_posix()] = path.read_bytes()
    return {
        "returncode": returncode,
        "result_dir": result_dir,
        "files": files,
        "command": shlex.join(cmd),
        "wall_seconds": round(time.time() - started, 1),
        "deadline_hit": flags["interrupted"],
        "spec": spec,
    }


@app.function(gpu=GPU, cpu=4, memory=8192, timeout=180, single_use_containers=True, **PLACEMENT)
def gpu_environment() -> dict:
    """GPU + versions on the A100, no training."""
    started = time.time()
    info = _environment()
    info["container_seconds"] = round(time.time() - started, 1)
    return info


@app.function(cpu=4, memory=8192, timeout=1800, volumes={DATA_ROOT: data_volume})
def download_data() -> None:
    """Download CIFAR-100 into the cifar100-data Volume. Run once; runs never download."""
    subprocess.run([PY, "-m", "benchmark.data", "--root", DATA_ROOT], cwd=APP_DIR, check=True)
    data_volume.commit()
    print("cifar100-data volume contains:", sorted(p.name for p in Path(DATA_ROOT).iterdir()))


@app.function(cpu=4, memory=8192, timeout=1800)
def smoke(submission: str = TEAM, submission_path: str = "") -> int:
    """CPU smoke test on synthetic images inside the Modal image (no GPU, no dataset)."""
    _environment()
    spec = {
        "label": "smoke",
        "n": 2,
        "device": "cpu",
        "synthetic": True,
        "submission": submission,
        "submission_path": submission_path,
    }
    run = _run(spec, results_root="/tmp/results")
    summary = json.loads(run["files"].get("summary.json", b"{}"))
    print(
        f"smoke: complete={summary.get('complete')} qualified={summary.get('qualified')} "
        f"exit={run['returncode']} (expected: complete=True qualified=None exit=0)",
        flush=True,
    )
    return run["returncode"]


@app.function(
    gpu=GPU,
    cpu=4,
    memory=16384,
    timeout=RUN_TIMEOUT,
    single_use_containers=True,
    volumes={DATA_ROOT: data_volume, RESULTS_ROOT: results_volume},
    **PLACEMENT,
)
def run_benchmarks(specs: list, require_pcie: bool = True, budget_s: float = 900.0) -> dict:
    """Run each spec sequentially in THIS container (one card), each as a fresh harness run.

    With require_pcie, a card other than the judges' A100 80GB PCIe aborts before any
    build; the launcher then retries on a fresh container. budget_s is the Modal timeout
    of this function (passed in because the container does not see the local env): work
    stops DEADLINE_MARGIN_S before it so the payload always gets back to the launcher.
    """
    started = time.time()
    deadline = started + budget_s - DEADLINE_MARGIN_S
    environment = _environment()
    payload: dict = {"environment": environment, "gpu_mismatch": False, "runs": []}
    if require_pcie and environment.get("gpu_name") != OFFICIAL_GPU:
        payload["gpu_mismatch"] = True
        print(
            f"PCIe guard: '{environment.get('gpu_name')}' is not '{OFFICIAL_GPU}'; "
            "aborting before build.",
            flush=True,
        )
    else:
        for spec in specs:
            remaining = deadline - time.time()
            if remaining < MIN_RUN_S:
                reason = f"skipped: only {remaining:.0f} s of container time left"
                print(f"\n##### {spec.get('label', 'run')}: {reason}", flush=True)
                payload["runs"].append(
                    {
                        "returncode": None,
                        "result_dir": None,
                        "files": {},
                        "command": "",
                        "wall_seconds": 0.0,
                        "deadline_hit": False,
                        "skipped": reason,
                        "spec": spec,
                    }
                )
                continue
            payload["runs"].append(_run(spec, RESULTS_ROOT, deadline))
            results_volume.commit()  # keep finished variants even if a later one hangs
    payload["container_seconds"] = round(time.time() - started, 1)
    return payload


# --------------------------------------------------------------------------- local: budget


def _ledger_total() -> float:
    if not LEDGER_PATH.exists():
        return 0.0
    total = 0.0
    for number, line in enumerate(LEDGER_PATH.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            total += float(json.loads(line).get("gpu_minutes", 0.0))
        except (ValueError, AttributeError):
            print(f"WARNING: ignoring unreadable ledger line {number}: {line[:80]}", flush=True)
    return total


def _ledger_add(label: str, gpu: str, container_seconds: float, note: str = "") -> float:
    """Append one container's GPU time to the ledger and print the running total."""
    minutes = (container_seconds + CONTAINER_START_S) / 60
    try:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        with LEDGER_PATH.open("a", encoding="utf-8") as ledger:
            ledger.write(
                json.dumps(
                    {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "label": label,
                        "gpu": gpu,
                        "container_seconds": container_seconds,
                        "gpu_minutes": round(minutes, 2),
                        "note": note,
                    }
                )
                + "\n"
            )
    except OSError as exc:
        print(f"WARNING: ledger write failed ({exc}); add {minutes:.1f} min by hand", flush=True)
    total = _ledger_total()
    print(
        f"GPU used: {total:.1f}/{GPU_BUDGET_MIN:.0f} min  (+{minutes:.1f} min: {label} on {gpu})",
        flush=True,
    )
    return total


def _estimate_seconds(spec: dict) -> float:
    """Rough container time of one harness run, deliberately on the high side.

    SPEEDRUN_TRIAL_S (default 60, per trial at 40 epochs) and SPEEDRUN_COMPILE_BUILD_S
    (default 150) let an approved launch use measured values instead of the defaults.
    """
    params = json.loads(spec.get("params") or "{}")
    per_trial = float(os.environ.get("SPEEDRUN_TRIAL_S", "60"))
    trial = max(8.0, per_trial * float(params.get("epochs", 40)) / 40)
    if params.get("use_compile"):
        mode = str(params.get("compile_mode", "default"))
        if mode.startswith("max-autotune"):
            build = float(spec.get("build_timeout") or 600)
        else:
            build = float(os.environ.get("SPEEDRUN_COMPILE_BUILD_S", "150"))
    else:
        build = 15.0
    return HARNESS_OVERHEAD_S + build + int(spec.get("n", 1)) * trial


RECOVERY_HINT = (
    "Finished runs are still in the cifar100-results Volume: list them with "
    "`uv run modal volume ls cifar100-results futurebiohackers` and copy one back with "
    "`uv run modal run scripts/modal_speedrun.py::fetch --run-id <run_id>` (no GPU)."
)


def _call_gpu(specs: list[dict], require_pcie: bool, label: str) -> dict:
    """run_benchmarks.remote with budget checks, the PCIe retry loop and ledger accounting."""
    total = _ledger_total()
    if total >= GPU_BUDGET_MIN:
        raise SystemExit(f"GPU budget exhausted: {total:.1f}/{GPU_BUDGET_MIN:.0f} min used.")
    if total >= GPU_STOP_MIN and not os.environ.get("SPEEDRUN_GPU_CONTINUE"):
        raise SystemExit(
            f"GPU checkpoint: {total:.1f} min used, at or past the {GPU_STOP_MIN:.0f} min stop "
            "line. Report first; set SPEEDRUN_GPU_CONTINUE=1 to continue."
        )
    estimate_s = ENV_CHECK_S + sum(_estimate_seconds(spec) for spec in specs)
    estimate_min = (estimate_s + CONTAINER_START_S) / 60
    print(
        f"estimate: ~{estimate_s:.0f} s in the container, ~{estimate_min:.1f} GPU-min "
        f"(ledger {total:.1f} -> ~{total + estimate_min:.1f}/{GPU_BUDGET_MIN:.0f} min); "
        f"container cap {RUN_TIMEOUT} s; placement {PLACEMENT or 'any'}"
        + ("; NEEDS APPROVAL: over 10 GPU-min" if estimate_min > 10 else ""),
        flush=True,
    )
    if estimate_s > RUN_TIMEOUT - DEADLINE_MARGIN_S - MIN_RUN_S:
        raise SystemExit(
            f"estimated {estimate_s:.0f} s does not fit the {RUN_TIMEOUT} s container cap: "
            "split the launch (fewer variants or trials, --build-timeout) or raise "
            "SPEEDRUN_TIMEOUT_MIN for this approved run."
        )
    landed: list[str] = []
    for attempt in range(1, PCIE_ATTEMPTS + 1):
        t0 = time.time()
        call = run_benchmarks.spawn(specs, require_pcie, float(RUN_TIMEOUT))
        try:
            payload = call.get(timeout=RUN_TIMEOUT + SCHEDULE_WAIT_S)
        except modal.exception.FunctionTimeoutError:
            _ledger_add(f"{label} (TIMEOUT)", "?", float(RUN_TIMEOUT), note="Modal timeout")
            raise SystemExit(
                f"Modal killed the container after {RUN_TIMEOUT} s. {RECOVERY_HINT}"
            ) from None
        except (TimeoutError, modal.exception.TimeoutError):
            call.cancel()
            _ledger_add(
                f"{label} (NO START)", "?", float(RUN_TIMEOUT), note="cancelled: never got a result"
            )
            raise SystemExit(
                f"no result after {RUN_TIMEOUT + SCHEDULE_WAIT_S:.0f} s (placement "
                f"{PLACEMENT or 'any'} may lack capacity); the call was cancelled. "
                "Check the Modal dashboard."
            ) from None
        except BaseException as exc:
            _ledger_add(
                f"{label} (FAILED {type(exc).__name__})",
                "?",
                min(time.time() - t0, float(RUN_TIMEOUT)),
                note="local wall time of the failed call",
            )
            raise
        env = payload["environment"]
        gpu = env.get("gpu_name", "?")
        where = f"{gpu} [task {env.get('task_id')}]"
        landed.append(where)
        if not payload["gpu_mismatch"]:
            _ledger_add(label, gpu, payload["container_seconds"])
            payload["pcie_attempts"] = landed
            return payload
        _ledger_add(
            f"{label} (pcie-guard attempt {attempt})",
            gpu,
            payload["container_seconds"],
            note="aborted before build",
        )
        print(
            f"PCIe guard: attempt {attempt}/{PCIE_ATTEMPTS} landed on {where}. "
            + ("Retrying on a fresh container." if attempt < PCIE_ATTEMPTS else "Giving up."),
            flush=True,
        )
    raise SystemExit(f"PCIe guard: no '{OFFICIAL_GPU}' in {PCIE_ATTEMPTS} attempts: {landed}")


# --------------------------------------------------------------------------- local: reports


def _save_run(run: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for rel, data in run["files"].items():
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    meta = {k: v for k, v in run.items() if k != "files"}
    (out / "modal_run.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _config_label(spec: dict) -> str:
    config = spec.get("submission_path") or spec.get("submission") or TEAM
    if spec.get("params") and spec["params"] != "{}":
        config += " " + spec["params"]
    if spec.get("no_accuracy_target"):
        config += " (no target)"
    return config


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _stats(run: dict, out: Path, environment: dict) -> dict:
    """Summarise one saved harness run (summary.json, trials.jsonl, config.json)."""
    spec = run["spec"]
    row: dict = {
        "label": spec.get("label", "run"),
        "params": spec.get("params") or "{}",
        "config": _config_label(spec),
        "returncode": run["returncode"],
        "folder": out.relative_to(REPO_ROOT).as_posix(),
        "gpu": environment.get("gpu_name", "?"),
        "power_limit": environment.get("gpu_power_limit", "?"),
        "container_wall": run["wall_seconds"],
        "deadline_hit": bool(run.get("deadline_hit")),
    }
    if run.get("skipped"):
        row.update(complete=False, qualified=False, verdict=run["skipped"].upper())
        return row
    s = _read_json(out / "summary.json")
    if not s:
        row.update(
            complete=False, qualified=False, verdict=f"NO summary.json (exit {row['returncode']})"
        )
        return row
    trials_path = out / "trials.jsonl"
    trials = (
        [
            json.loads(line)
            for line in trials_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if trials_path.exists()
        else []
    )
    ok = [t for t in trials if t.get("status") == "ok"]
    config = _read_json(out / "config.json")
    build_time = config.get("build_time")
    if s["qualified"] is True:
        verdict = "QUALIFIED (>= 75%)"
    elif s["complete"] and s["qualified"] is None:
        verdict = "complete, no target"
    elif s["complete"]:
        verdict = "BELOW 75%"
    else:
        verdict = f"INCOMPLETE ({s.get('run_error') or 'failed trials'})"
    if s.get("run_error") == "build_timeout" and build_time is None:
        build_time = config.get("build_timeout")
        verdict += f": build exceeded {_sec(build_time)} s"
    if row["deadline_hit"]:
        verdict += " (interrupted at the container deadline)"
    several = s["successful_trials"] > 1  # the std of one trial is not a measurement
    row.update(
        complete=s["complete"],
        qualified=s["qualified"],
        run_error=s.get("run_error"),
        n=s["requested_trials"],
        trials_ok=s["successful_trials"],
        mean_acc=s["mean_accuracy"],
        acc_std=s["accuracy_std"] if several else None,
        mean_time=s["mean_training_time"],
        time_std=s["training_time_std"] if several else None,
        mean_eval=s["mean_evaluation_time"],
        build_time=build_time,
        mean_prepare=statistics.mean(t["prepare_time"] for t in ok) if ok else None,
        mean_train=statistics.mean(t["train_time"] for t in ok) if ok else None,
        accs=[t["accuracy"] for t in ok],
        times=[t["total_timed_time"] for t in ok],
        evals=[t["evaluation_time"] for t in ok],
        verdict=verdict,
    )
    return row


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}"


def _sec(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _print_row(row: dict) -> None:
    print(f"\n=== {row['folder']} ===")
    error = REPO_ROOT / row["folder"] / "error.txt"
    if error.exists():
        print("--- error.txt ---")
        print(error.read_text(encoding="utf-8"))
    if "n" not in row:
        print(f"exit code {row['returncode']}: {row['verdict']} (see the container log above)")
        return
    print(f"exit code {row['returncode']}: {row['verdict']}")
    print(
        f"trials {row['trials_ok']}/{row['n']}  mean acc {_pct(row['mean_acc'])}% "
        f"(std {_pct(row['acc_std'])})  mean prep+train {_sec(row['mean_time'])} s "
        f"(std {_sec(row['time_std'])}; prepare {_sec(row['mean_prepare'], 3)} + "
        f"train {_sec(row['mean_train'])})  mean eval {_sec(row['mean_eval'], 3)} s  "
        f"build {_sec(row['build_time'])} s"
    )
    print(
        "per trial: "
        + ", ".join(
            f"{_pct(a)}% / {_sec(t)} s / eval {_sec(e, 3)} s"
            for a, t, e in zip(row["accs"], row["times"], row["evals"], strict=True)
        )
    )
    print(f"GPU: {row['gpu']}  power limit {row['power_limit']}")


def _log_row(row: dict, tag: str) -> str:
    if "n" not in row:
        return (
            f"| {datetime.now():%Y-%m-%d} | {tag} | {row['config']} | ? | n/a | n/a | n/a | "
            f"{row['gpu']} ({row['power_limit']}) | {row['verdict']} |"
        )
    return (
        f"| {datetime.now():%Y-%m-%d} | {tag} | {row['config']} | {row['n']} | "
        f"{_pct(row['mean_acc'])} | {_pct(row['acc_std'])} | {_sec(row['mean_time'])} | "
        f"{row['gpu']} ({row['power_limit']}) | {row['verdict']}; "
        f"build {_sec(row['build_time'])} s, "
        f"eval {_sec(row['mean_eval'], 3)} s |"
    )


def _table(rows: list[dict]) -> str:
    head = (
        "| variant | config | n | mean acc % | std | mean prep+train s | std | prepare s | "
        "train s | eval s | build s | verdict |\n| --- | --- | --- | --- | --- | --- | --- | --- "
        "| --- | --- | --- | --- |"
    )
    lines = [head]
    for r in rows:
        if "n" not in r:
            lines.append(f"| {r['label']} | `{r['config']}` | ? | | | | | | | | | {r['verdict']} |")
            continue
        lines.append(
            f"| {r['label']} | `{r['config']}` | {r['trials_ok']}/{r['n']} | "
            f"{_pct(r['mean_acc'])} | "
            f"{_pct(r['acc_std'])} | {_sec(r['mean_time'])} | {_sec(r['time_std'])} | "
            f"{_sec(r['mean_prepare'], 3)} | {_sec(r['mean_train'])} | {_sec(r['mean_eval'], 3)} | "
            f"{_sec(r['build_time'])} | {r['verdict']} |"
        )
    return "\n".join(lines)


def _spec(
    label: str,
    n: int,
    params: dict,
    submission: str,
    submission_path: str,
    no_accuracy_target: bool,
    seed: int | None,
    build_timeout: float,
    torch_logs: str,
) -> dict:
    return {
        "label": label,
        "n": n,
        "submission": submission,
        "submission_path": submission_path,
        "params": json.dumps(params, sort_keys=True),
        "no_accuracy_target": no_accuracy_target,
        "seed": seed,
        "build_timeout": build_timeout,
        "torch_logs": torch_logs,
    }


# --------------------------------------------------------------------------- entrypoints


@app.local_entrypoint()
def env_check():
    """GPU name/memory/power limit, versions and CPUs; logs the container to the ledger."""
    t0 = time.time()
    call = gpu_environment.spawn()
    try:
        info = call.get(timeout=180 + SCHEDULE_WAIT_S)
    except (TimeoutError, modal.exception.TimeoutError):
        call.cancel()
        _ledger_add("env_check (NO START)", "?", 0.0, note=f"placement {PLACEMENT or 'any'}")
        raise SystemExit(f"env_check: no container within {SCHEDULE_WAIT_S:.0f} s; cancelled")
    except BaseException as exc:
        _ledger_add(f"env_check (FAILED {type(exc).__name__})", "?", min(time.time() - t0, 180.0))
        raise
    print(
        f"placement {PLACEMENT or 'any'} -> region={info.get('region')} cloud={info.get('cloud')}"
    )
    _ledger_add("env_check", info.get("gpu_name", "?"), info.get("container_seconds", 0.0))
    print(json.dumps({k: v for k, v in info.items() if k != "versions"}, indent=2))


@app.local_entrypoint()
def main(
    tag: str = "dev",
    n: int = 1,
    submission: str = TEAM,
    submission_path: str = "",
    params: str = "{}",
    no_accuracy_target: bool = False,
    seed: int = -1,
    require_pcie: bool = True,
    build_timeout: float = DEFAULT_BUILD_TIMEOUT,
    torch_logs: str = "",
):
    """One benchmark.run on an A100 80GB PCIe; results in artifacts/speedrun_runs/<ts>_<tag>/."""
    parameters = json.loads(params)
    if not isinstance(parameters, dict):
        raise SystemExit("--params must be a JSON object")
    spec = _spec(
        tag,
        n,
        parameters,
        submission,
        submission_path,
        no_accuracy_target,
        None if seed < 0 else seed,
        build_timeout,
        torch_logs,
    )
    payload = _call_gpu([spec], require_pcie, tag)
    run = payload["runs"][0]
    out = ARTIFACTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{_safe(tag)}"
    _save_run(run, out)
    row = _stats(run, out, payload["environment"])
    _print_row(row)
    print(f"PCIe guard attempts: {payload['pcie_attempts']}")
    print("LOG.md row:")
    print(_log_row(row, tag))
    if run["returncode"] != 0:
        sys.exit(run["returncode"] or 1)


@app.local_entrypoint()
def ab(
    tag: str = "ab",
    n: int = 2,
    submission: str = TEAM,
    control_params: str = "{}",
    variants: str = "[]",
    labels: str = "",
    submission_paths: str = "",
    no_accuracy_target: bool = False,
    require_pcie: bool = True,
    build_timeout: float = DEFAULT_BUILD_TIMEOUT,
    torch_logs: str = "",
):
    """Control + variants sequentially in ONE container; prints a comparison table.

    --variants is a JSON list of parameter deltas merged over --control-params;
    --submission-paths is a JSON list of folders to compare against the control's folder;
    --labels gives comma-separated names for the variants (default v01, v02, ...).
    """
    control = json.loads(control_params)
    deltas = json.loads(variants)
    paths = json.loads(submission_paths) if submission_paths else []
    if not isinstance(control, dict):
        raise SystemExit("--control-params must be a JSON object")
    if not isinstance(deltas, list) or not all(isinstance(d, dict) for d in deltas):
        raise SystemExit("--variants must be a JSON array of objects")
    if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
        raise SystemExit("--submission-paths must be a JSON array of strings")
    if not deltas and not paths:
        raise SystemExit("Give at least one variant (--variants or --submission-paths)")
    names = [name.strip() for name in labels.split(",") if name.strip()] if labels else []

    def make(label: str, params: dict, path: str = "") -> dict:
        return _spec(
            label, n, params, submission, path, no_accuracy_target, None, build_timeout, torch_logs
        )

    specs = [make("control", control)]
    for i, delta in enumerate(deltas):
        specs.append(make(names[i] if i < len(names) else f"v{i + 1:02d}", {**control, **delta}))
    for j, path in enumerate(paths):
        k = len(deltas) + j
        specs.append(make(names[k] if k < len(names) else f"p{j + 1:02d}", control, path))
    seen = [s["label"] for s in specs]
    if len(set(seen)) != len(seen):
        raise SystemExit(f"variant labels must be unique: {seen}")
    print(f"A/B '{tag}': {len(specs)} runs x {n} trials, sequential in one {GPU} container:")
    for spec in specs:
        print(f"  {spec['label']:>12}: {_config_label(spec)}")
    payload = _call_gpu(specs, require_pcie, tag)
    out = ARTIFACTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{_safe(tag)}"
    rows = []
    for i, run in enumerate(payload["runs"]):
        sub = out / f"{i:02d}_{_safe(run['spec']['label'])}"
        _save_run(run, sub)
        rows.append(_stats(run, sub, payload["environment"]))
    for row in rows:
        _print_row(row)
    env = payload["environment"]
    table = _table(rows)
    header = (
        f"# A/B {tag}\n\n{datetime.now():%Y-%m-%d %H:%M}, {env.get('gpu_name', '?')} "
        f"(power limit {env.get('gpu_power_limit', '?')}, task {env.get('task_id')}), one "
        f"container, sequential, {n} trial(s) per variant, container wall "
        f"{payload['container_seconds']} s. PCIe guard attempts: {payload['pcie_attempts']}.\n\n"
    )
    (out / "ab_summary.md").write_text(header + table + "\n", encoding="utf-8")
    (out / "ab_summary.json").write_text(
        json.dumps(
            {
                "tag": tag,
                "environment": env,
                "pcie_attempts": payload["pcie_attempts"],
                "container_seconds": payload["container_seconds"],
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\n" + header + table)
    print("\nLOG.md rows:")
    for row in rows:
        print(_log_row(row, f"{tag}/{row['label']}"))
    print(f"\nsaved: {out.relative_to(REPO_ROOT).as_posix()}/ab_summary.md")
    if any(row["returncode"] != 0 for row in rows):
        sys.exit(1)


@app.local_entrypoint()
def fetch(run_id: str, team: str = TEAM, tag: str = ""):
    """Copy results/<team>/<run_id>/ out of the cifar100-results Volume (no GPU)."""
    prefix = f"{team}/{run_id}"
    files: dict[str, bytes] = {}
    for entry in results_volume.listdir(prefix, recursive=True):
        if entry.type != FileEntryType.FILE:
            continue
        rel = entry.path.split(prefix, 1)[-1].lstrip("/")
        files[rel] = b"".join(results_volume.read_file(entry.path))
    if not files:
        raise SystemExit(f"nothing under {prefix} in Volume cifar100-results")
    label = tag or f"fetched-{run_id}"
    run = {
        "returncode": None,
        "result_dir": f"{RESULTS_ROOT}/{prefix}",
        "files": files,
        "command": "",
        "wall_seconds": 0.0,
        "deadline_hit": False,
        "spec": {"label": label, "params": "{}", "submission": team},
    }
    out = ARTIFACTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{_safe(label)}"
    _save_run(run, out)
    row = _stats(run, out, {})
    _print_row(row)
    print("LOG.md row (fill in the GPU and params from the original launch):")
    print(_log_row(row, label))


@app.local_entrypoint()
def sweep(
    tag: str = "sweep",
    n: int = 1,
    submission: str = TEAM,
    submission_path: str = "",
    params_list: str = "[]",
    no_accuracy_target: bool = False,
    build_timeout: float = DEFAULT_BUILD_TIMEOUT,
):
    """One A100 container per --params-list entry, in parallel. Gated: SPEEDRUN_ALLOW_SWEEP=1."""
    if not os.environ.get("SPEEDRUN_ALLOW_SWEEP"):
        raise SystemExit(
            "sweeps are disabled under the current compute budget (parallel containers, no PCIe "
            "guard). Use ::ab, or set SPEEDRUN_ALLOW_SWEEP=1 for an approved sweep."
        )
    configs = json.loads(params_list)
    valid = isinstance(configs, list) and configs and all(isinstance(c, dict) for c in configs)
    if not valid:
        raise SystemExit("--params-list must be a non-empty JSON array of objects")
    specs = [
        _spec(
            f"{tag}_{i:02d}",
            n,
            c,
            submission,
            submission_path,
            no_accuracy_target,
            None,
            build_timeout,
            "",
        )
        for i, c in enumerate(configs)
    ]
    total = _ledger_total()
    estimate_min = sum((ENV_CHECK_S + _estimate_seconds(s) + CONTAINER_START_S) / 60 for s in specs)
    print(
        f"sweep '{tag}': {len(specs)} configs x {n} trials, one {GPU} container each; "
        f"estimate ~{estimate_min:.1f} GPU-min (ledger {total:.1f}/{GPU_BUDGET_MIN:.0f})"
    )
    if total + estimate_min > GPU_BUDGET_MIN:
        raise SystemExit("this sweep would exceed the GPU budget")
    failures = 0
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    for i, payload in enumerate(
        run_benchmarks.map(
            [[s] for s in specs],
            kwargs={"require_pcie": False, "budget_s": float(RUN_TIMEOUT)},
            order_outputs=True,
            return_exceptions=True,
        )
    ):
        if isinstance(payload, Exception):
            timed_out = isinstance(payload, modal.exception.FunctionTimeoutError)
            seconds = float(RUN_TIMEOUT) if timed_out else 0.0
            _ledger_add(f"{tag}_{i:02d} (FAILED {type(payload).__name__})", "?", seconds)
            print(f"{tag}_{i:02d} failed: {payload!r}")
            failures += 1
            continue
        gpu = payload["environment"].get("gpu_name", "?")
        _ledger_add(f"{tag}_{i:02d}", gpu, payload["container_seconds"])
        run = payload["runs"][0]
        out = ARTIFACTS_DIR / f"{stamp}_{_safe(tag)}_{i:02d}"
        _save_run(run, out)
        row = _stats(run, out, payload["environment"])
        _print_row(row)
        print(_log_row(row, f"{tag}_{i:02d}"))
        failures += run["returncode"] != 0
    if failures:
        sys.exit(1)
