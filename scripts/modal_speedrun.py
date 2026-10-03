"""Run the CIFAR-100 speedrun harness on Modal A100-80GB GPUs (judging: A100-SXM4-80GB).

This file lives in the TEAM repo (scripts/), outside cifar100-speedrun/, so the upstream
PR contains only submissions/futurebiohackers/. Run it from the team env at the repo root
(Windows: set PYTHONUTF8=1 first). `modal run` needs the entrypoint name after `::`.

    uv run modal run scripts/modal_speedrun.py::main --tag ref --n 40            # one cold run
    uv run modal run scripts/modal_speedrun.py::ab --n 8 --tag widths \
        --variants '[{"widths": [128, 320, 768]}]' --labels w320-768            # one container
    uv run modal run scripts/modal_speedrun.py::screen --jobs jobs/r1.json --parallel 10
    uv run modal run scripts/modal_speedrun.py::profile --params '{}'
    uv run modal run scripts/modal_speedrun.py::env_check

::main: harness flags pass straight through to `benchmark.run` (`--n`, `--params`,
`--no-accuracy-target`, `--seed`, `--build-timeout`, `--submission-path`, ...); default
`--n 1`. Launcher-only flags: --tag T, --require-gpu sxm|pcie|any, --require-power W|any,
--warm (persistent compile cache instead of a cold build), --count-nonfinite (adds
{"count_nonfinite": true} to --params so the recipe reports non-finite step losses).
CIFAR-100 is cached in the `cifar100-data` Volume (downloaded on first use); results land in
the `cifar100-results` Volume and are copied back to cifar100-speedrun/results/ (so
`just last` works) and to artifacts/speedrun_runs/<timestamp>_<tag>/.

GPU guard (default: require an A100-SXM4-80GB, any power limit): Modal's A100-80GB pool
mixes SXM4 parts at 400 W and 500 W with PCIe cards (300 W), and the power limit changes
timings a lot. Every container reads nvidia-smi BEFORE any build; on the wrong card it
returns at once (about 0.3 GPU-min) and the launcher retries on a fresh single-use
container, up to MODAL_GPU_ATTEMPTS calls (default 11 = one try + 10 retries), logging GPU,
power limit, task id, region and cloud for every attempt. There is NO fallback: a job that
never gets the required card fails. --require-power 400|500 pins the power limit too.

Screening (::screen): a jobs file describes a round: {"round": "r1", "n": 8, "seed": 0,
"control": {...control params...}, "jobs": [{"name": "lowres", "variants": [{"label": ...,
"params": {...deltas over the control...}, "hypothesis": "..."}, ...]}, ...]}. Every job runs
in ONE container: the control first, then its variants (at most 3), each as its own
`benchmark.run` process, paired on the same seeds. Up to --parallel containers run at once.
Screening uses a persistent WARM torch.compile cache (Volume `cifar100-inductor-cache`,
copied into the container at start and merged back at the end) plus the FX graph cache, so
only new graphs pay a compile; cold builds are measured for finalists with ::main. Each
variant is appended to artifacts/speedrun_runs/registry.jsonl (config, hypothesis, GPU and
power limit, per-trial accuracies and times, paired dacc +- SE, dtime, dtime_adj = dtime -
dacc_pp / k with k from artifacts/speedrun_runs/k.json, build time, non-finite count) and
to LOG.md; scripts/leaderboard.py is regenerated afterwards.

Time caps: every GPU function has a hard Modal timeout (MODAL_TIMEOUT_MIN, default 20 min).
Inside the container each run has a deadline: the harness gets SIGINT (it keeps finished
trials) and later runs are skipped, so the payload always comes back. Before launching, each
job's GPU time is estimated and jobs over MODAL_JOB_LIMIT_MIN (20) are refused unless
--allow-big is given (the rule: ask before any job above 20 GPU-minutes).

GPU budget: each container's wall time plus a start-up allowance is appended to
artifacts/speedrun_runs/gpu_ledger.jsonl and "GPU used: X/1263 min" is printed after every
call. MODAL_GPU_BUDGET_MIN (3000, the cap set on 3 Oct evening) is a hard stop.

Set TEAM for another submission folder and MODAL_GPU for another GPU type (the guard is then
off).
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import modal
import modal.exception

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEEDRUN = REPO_ROOT / "cifar100-speedrun"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "speedrun_runs"
LEDGER_PATH = ARTIFACTS_DIR / "gpu_ledger.jsonl"
REGISTRY_PATH = ARTIFACTS_DIR / "registry.jsonl"
K_PATH = ARTIFACTS_DIR / "k.json"
LOG_PATH = ARTIFACTS_DIR / "LOG.md"
LEADERBOARD = REPO_ROOT / "scripts" / "leaderboard.py"
PROFILER = REPO_ROOT / "scripts" / "profile_recipe.py"
REMOTE = "/root/speedrun"
REMOTE_PROFILER = "/root/profile_recipe.py"
DATA_ROOT = "/data"
RESULTS_ROOT = "/results"
CACHE_ROOT = "/cache"  # Volume with the warm torch.compile caches
CACHE_TAR = f"{CACHE_ROOT}/inductor.tar.gz"  # the whole cache as ONE file (see _warm_cache_start)
CACHE_TAR_MAX_BYTES = 2_500_000_000
LOCAL_CACHE = "/root/inductor-cache"  # per-container copy of the warm cache
TEAM = os.environ.get("TEAM", "futurebiohackers")
GPU = os.environ.get("MODAL_GPU", "A100-80GB") or None  # empty: CPU-only
# Modal's A100-80GB pool mixes SXM4 parts (400 W and 500 W power limits) with PCIe cards
# (300 W). Judging is on an A100-SXM4-80GB, so runs require that card by default; the power
# limit changes timings a lot and is recorded for every run (and can be required too).
GPU_NAMES = {"sxm": "NVIDIA A100-SXM4-80GB", "pcie": "NVIDIA A100 80GB PCIe"}
DEFAULT_REQUIRE_GPU = os.environ.get("MODAL_REQUIRE_GPU", "sxm")  # sxm | pcie | any
DEFAULT_REQUIRE_POWER = os.environ.get("MODAL_REQUIRE_POWER", "400")  # 400 | 500 | 300 | any
GPU_ATTEMPTS = int(os.environ.get("MODAL_GPU_ATTEMPTS", "11"))  # one try + 10 retries, no fallback

RUN_TIMEOUT = int(os.environ.get("MODAL_TIMEOUT_MIN", "20")) * 60  # hard cap per container
DEADLINE_MARGIN_S = 60.0  # the container stops starting work this long before RUN_TIMEOUT
MIN_RUN_S = 90.0  # do not start a run with less container time left than this
SCHEDULE_WAIT_S = 1800.0  # extra wait for Modal to find a container (parallel jobs may queue)
CONTAINER_START_S = 15.0  # billed start-up allowance per container, not measurable inside it
# Hard cap on the ledger total (raised to 2000 GPU-min on 3 Oct 2026, evening).
GPU_BUDGET_MIN = float(os.environ.get("MODAL_GPU_BUDGET_MIN", "3000"))
GPU_CHECKPOINTS = [
    float(x) for x in os.environ.get("MODAL_GPU_CHECKPOINTS", "").split(",") if x.strip()
]
JOB_LIMIT_MIN = float(os.environ.get("MODAL_JOB_LIMIT_MIN", "20"))  # ask before a bigger job
# Cost model for the pre-launch estimate (deliberately on the high side, SXM 400 W).
EST_TRIAL_S_PER_EPOCH = 1.0  # prepare + train seconds per epoch at the default width
EST_BUILD_COLD_S = 230.0  # cold max-autotune build
EST_BUILD_WARM_S = 70.0  # build with the warm cache (observed 16-18 s on a cache hit)
EST_BUILD_NEW_GRAPH_WARM_S = 110.0  # new graph, warm autotune/Triton caches (observed 38-55 s)
EST_BUILD_LOW_RES_S = 70.0  # extra default-mode graph for progressive resizing
EST_RUN_OVERHEAD_S = 25.0  # process start, dataset load, eval, result copy
EST_CONTAINER_S = 40.0  # nvidia-smi, cache copy, data check
# Parameters that change the compiled graph (so the warm cache cannot help on first sight).
GRAPH_KEYS = {
    "widths",
    "depths",
    "depth",
    "pool_first",
    "stem",
    "inner_kernels",
    "gelu_approximate",
    "activation",
    "scaling_factor",
    "bn_momentum",
    "bn_dtype",
    "global_pool",
    "optimizer",
    "compile_step",
    "compile_loss",
    "label_smoothing",
    "batch_size",
    "batch_schedule",
    "resolution_schedule",
    "train_resolution",
    "resolution_switch",
    "compile",
}  # low_res adds one extra default-mode graph on top (EST_BUILD_LOW_RES_S)

_LOCK = threading.Lock()  # ledger, registry and LOG.md appends from parallel jobs


def _power_watts(environment: dict) -> int | None:
    try:
        return int(round(float(str(environment.get("gpu_power_limit", "")).split()[0])))
    except (ValueError, IndexError):
        return None


def _gpu_mismatch(environment: dict, require: dict) -> str | None:
    """Why this container's card is not the required one (None when it is)."""
    want, power = require.get("gpu", "any"), require.get("power", "any")
    name = environment.get("gpu_name", "?")
    if want != "any" and name != GPU_NAMES[want]:
        return f"GPU '{name}' is not '{GPU_NAMES[want]}'"
    if str(power) != "any" and _power_watts(environment) != int(power):
        return f"power limit {environment.get('gpu_power_limit')} is not {power} W"
    return None


def _parse_require(gpu: str, power: str) -> dict:
    if gpu not in ("sxm", "pcie", "any"):
        raise SystemExit("--require-gpu must be sxm, pcie or any")
    if str(power) != "any" and not str(power).isdigit():
        raise SystemExit("--require-power must be a wattage (400, 500, 300) or any")
    if GPU != "A100-80GB" and gpu != "any":
        print(f"GPU guard off: MODAL_GPU={GPU!r} is not the A100-80GB pool.", flush=True)
        return {"gpu": "any", "power": "any"}
    return {"gpu": gpu, "power": str(power)}


def _where(environment: dict) -> str:
    return (
        f"{environment.get('gpu_name', '?')} @ {environment.get('gpu_power_limit', '?')} "
        f"[task {environment.get('task_id')} {environment.get('cloud')}/{environment.get('region')}]"
    )


image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(SPEEDRUN))
    .env({"PYTHONPATH": REMOTE, "PYTHONUNBUFFERED": "1"})
    .add_local_dir(
        SPEEDRUN, REMOTE, ignore=["**/.venv", "**/__pycache__", "data", "results", "**/.DS_Store"]
    )
    .add_local_file(PROFILER, REMOTE_PROFILER)
)
app = modal.App("cifar100-speedrun", image=image)
data = modal.Volume.from_name("cifar100-data", create_if_missing=True)
results = modal.Volume.from_name("cifar100-results", create_if_missing=True)
cache = modal.Volume.from_name("cifar100-inductor-cache", create_if_missing=True)


def _safe(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", tag).strip("-") or "run"


# --------------------------------------------------------------------------- remote


def _environment() -> dict:
    """GPU name/power, container identity (task id, region, cloud); runs inside the container."""
    info: dict = {
        "task_id": os.environ.get("MODAL_TASK_ID"),
        "hostname": socket.gethostname(),
        "region": os.environ.get("MODAL_REGION"),
        "cloud": os.environ.get("MODAL_CLOUD_PROVIDER"),
        "cpu_count": os.cpu_count(),
    }
    print(
        f"=== container task={info['task_id']} region={info['region']} cloud={info['cloud']} "
        f"cpus={info['cpu_count']}",
        flush=True,
    )
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,power.limit,driver_version,mig.mode.current",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        line = (smi.stdout or smi.stderr).strip()
        print(line, flush=True)
        fields = [f.strip() for f in line.split(",")]
        if smi.returncode == 0 and len(fields) >= 5:
            info.update(
                gpu_name=fields[0],
                gpu_memory=fields[1],
                gpu_power_limit=fields[2],
                driver=fields[3],
                mig=fields[4],
            )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"nvidia-smi unavailable: {exc}", flush=True)
    return info


def _ensure_data() -> None:
    if not Path(f"{DATA_ROOT}/cifar-100-python").exists():
        subprocess.run(
            [sys.executable, "-m", "benchmark.data", "--root", DATA_ROOT], cwd=REMOTE, check=True
        )
        data.commit()


def _warm_cache_start() -> None:
    """Unpack the shared compile cache (ONE tarball in the Volume) onto local disk.

    The cache holds tens of thousands of small Triton files; copying them one by one from a
    Modal Volume took longer than the training runs, so the Volume keeps a single tar.gz.
    """
    local = Path(LOCAL_CACHE)
    local.mkdir(parents=True, exist_ok=True)
    tar_path = Path(CACHE_TAR)
    if not tar_path.exists():
        print("warm cache: Volume has no tarball yet (cold start)", flush=True)
        return
    started = time.time()
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(local, filter="data")
        print(
            f"warm cache: {tar_path.stat().st_size / 1e6:.0f} MB tarball extracted in "
            f"{time.time() - started:.1f} s",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 - a cache problem must never block a run
        print(f"warm cache: extraction failed ({exc}); building cold", flush=True)


def _warm_cache_commit() -> None:
    """Pack the local cache into one tarball and replace the Volume's copy (last writer wins)."""
    started = time.time()
    tmp = "/root/inductor-cache.tar.gz"
    try:
        with tarfile.open(tmp, "w:gz", compresslevel=1) as tar:
            tar.add(LOCAL_CACHE, arcname=".")
        size = os.path.getsize(tmp)
        if size > CACHE_TAR_MAX_BYTES:
            print(
                f"warm cache: {size / 1e6:.0f} MB tarball over the cap; not committed", flush=True
            )
            return
        Path(CACHE_TAR).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp, CACHE_TAR)
        cache.commit()
        print(
            f"warm cache: {size / 1e6:.0f} MB tarball committed in {time.time() - started:.1f} s",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 - a cache problem must never lose results
        print(f"warm cache: commit failed ({exc}); results are unaffected", flush=True)


def _popen_with_deadline(cmd: list[str], env: dict, deadline: float | None, label: str) -> dict:
    """Run cmd, stream its output (prefixed with the label), SIGINT it at the deadline."""
    started = time.time()
    proc = subprocess.Popen(
        cmd, cwd=REMOTE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    flags = {"interrupted": False}
    timers: list[threading.Timer] = []
    if deadline is not None:
        remaining = max(1.0, deadline - time.time())

        def interrupt() -> None:
            flags["interrupted"] = True
            print(
                f"\n##### {label}: container deadline after {remaining:.0f} s, sending SIGINT "
                "(finished trials are kept)",
                flush=True,
            )
            proc.send_signal(signal.SIGINT)

        timers = [threading.Timer(remaining, interrupt), threading.Timer(remaining + 45, proc.kill)]
        for timer in timers:
            timer.start()
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[{label}] {line}", end="", flush=True)
        lines.append(line)
    returncode = proc.wait()
    for timer in timers:
        timer.cancel()
    return {
        "returncode": returncode,
        "lines": lines,
        "wall_seconds": round(time.time() - started, 1),
        "deadline_hit": flags["interrupted"],
    }


def _run(spec: dict, deadline: float | None, warm: bool) -> dict:
    """One benchmark.run process (own build and worker); cold or warm compile cache."""
    label = spec.get("label", "run")
    cmd = [
        sys.executable,
        "-m",
        "benchmark.run",
        "--data-root",
        DATA_ROOT,
        "--results-root",
        RESULTS_ROOT,
        *spec["args"],
    ]
    env = dict(os.environ)
    env.update(spec.get("env") or {})  # e.g. {"TORCH_LOGS": "recompiles"} for a diagnostic run
    if warm:
        env["TORCHINDUCTOR_CACHE_DIR"] = LOCAL_CACHE
        env["TRITON_CACHE_DIR"] = f"{LOCAL_CACHE}/triton"
        env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
        cache_dir = None
    else:
        # Cold compile caches: a compiled variant must not inherit the kernels or autotune
        # results of an earlier one (the judges' container is cold too).
        cache_dir = tempfile.mkdtemp(prefix=f"inductor-{_safe(label)}-")
        env["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
        env["TRITON_CACHE_DIR"] = f"{cache_dir}/triton"
    print(
        f"\n##### {label} ({'warm' if warm else 'cold'} cache) #####\n$ {shlex.join(cmd)}",
        flush=True,
    )
    run = _popen_with_deadline(cmd, env, deadline, label)
    if cache_dir:
        shutil.rmtree(cache_dir, ignore_errors=True)
    result_dir = None
    nonfinite = None
    for line in run.pop("lines"):
        if line.startswith("Results: "):
            result_dir = line[len("Results: ") :].strip()
        elif line.startswith("NONFINITE_LOSSES "):
            try:
                nonfinite = (nonfinite or 0) + int(line.split()[1])
            except (IndexError, ValueError):
                pass
    files: dict[str, bytes] = {}
    if result_dir and Path(result_dir).is_dir():
        for path in sorted(Path(result_dir).rglob("*")):
            if path.is_file() and "source" not in path.relative_to(result_dir).parts:
                files[path.relative_to(result_dir).as_posix()] = path.read_bytes()
    # The harness prints the Volume's realpath, not /results/...: keep <team>/<run_id> only.
    run.update(
        result_rel="/".join(Path(result_dir).parts[-2:]) if result_dir else None,
        files=files,
        command=shlex.join(cmd),
        spec=spec,
        nonfinite=nonfinite,
        build_mode="warm" if warm else "cold",
    )
    return run


def _profile(spec: dict, deadline: float | None) -> dict:
    cmd = [sys.executable, REMOTE_PROFILER, "--data-root", DATA_ROOT, *spec["args"]]
    print(f"\n##### {spec['label']} #####\n$ {shlex.join(cmd)}", flush=True)
    run = _popen_with_deadline(cmd, dict(os.environ), deadline, spec["label"])
    report = {}
    for line in run.pop("lines"):
        if line.startswith("PROFILE_JSON "):
            report = json.loads(line[len("PROFILE_JSON ") :])
    run.update(result_rel=None, files={}, command=shlex.join(cmd), spec=spec, profile=report)
    return run


@app.function(gpu=GPU, cpu=4, memory=8192, timeout=180, single_use_containers=True)
def gpu_environment() -> dict:
    started = time.time()
    info = _environment()
    info["container_seconds"] = round(time.time() - started, 1)
    return info


@app.function(
    gpu=GPU,
    cpu=4,
    memory=16384,
    timeout=RUN_TIMEOUT,
    single_use_containers=True,
    volumes={DATA_ROOT: data, RESULTS_ROOT: results, CACHE_ROOT: cache},
)
def run_benchmarks(specs: list, require: dict, budget_s: float, warm: bool = False) -> dict:
    """Run each spec sequentially in THIS container (one card), each a fresh harness run.

    `require` ({"gpu": sxm|pcie|any, "power": watts|any}) is checked against nvidia-smi
    before any build or download; on a mismatch the container returns at once and the
    launcher retries on a fresh single-use container. With `warm`, runs share the persistent
    compile cache; otherwise every run builds cold.
    """
    started = time.time()
    deadline = started + budget_s - DEADLINE_MARGIN_S
    environment = _environment()
    payload: dict = {"environment": environment, "gpu_mismatch": False, "runs": [], "warm": warm}
    mismatch = _gpu_mismatch(environment, require)
    if mismatch:
        payload["gpu_mismatch"] = True
        print(f"GPU guard: {mismatch}; aborting before build.", flush=True)
    else:
        _ensure_data()
        if warm:
            _warm_cache_start()
        for spec in specs:
            remaining = deadline - time.time()
            if remaining < MIN_RUN_S:
                reason = f"skipped: only {remaining:.0f} s of container time left"
                print(f"\n##### {spec.get('label', 'run')}: {reason}", flush=True)
                payload["runs"].append(
                    {
                        "returncode": None,
                        "result_rel": None,
                        "files": {},
                        "command": "",
                        "wall_seconds": 0.0,
                        "deadline_hit": False,
                        "skipped": reason,
                        "spec": spec,
                    }
                )
                continue
            if spec.get("kind") == "profile":
                payload["runs"].append(_profile(spec, deadline))
            else:
                payload["runs"].append(_run(spec, deadline, warm))
                results.commit()  # keep finished runs even if a later one hangs
        if warm:
            _warm_cache_commit()
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


def _ledger_add(
    label: str, gpu: str, container_seconds: float, note: str = "", power: str | None = None
) -> float:
    """Append one container's GPU time to the ledger and print the running total."""
    # Every container that ran is billed at least its start-up; only a cancelled call that
    # never got a container costs nothing.
    minutes = 0.0 if note.startswith("cancelled") else (container_seconds + CONTAINER_START_S) / 60
    with _LOCK:
        try:
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            with LEDGER_PATH.open("a", encoding="utf-8") as ledger:
                ledger.write(
                    json.dumps(
                        {
                            "time": datetime.now().isoformat(timespec="seconds"),
                            "label": label,
                            "gpu": gpu,
                            "power": power,
                            "container_seconds": container_seconds,
                            "gpu_minutes": round(minutes, 2),
                            "note": note,
                        }
                    )
                    + "\n"
                )
        except OSError as exc:
            print(
                f"WARNING: ledger write failed ({exc}); add {minutes:.1f} min by hand", flush=True
            )
        total = _ledger_total()
    print(
        f"GPU used: {total:.1f}/{GPU_BUDGET_MIN:.0f} min  (+{minutes:.1f} min: {label} on {gpu} "
        f"{power or ''})",
        flush=True,
    )
    return total


def _check_budget(extra_min: float = 0.0) -> None:
    total = _ledger_total()
    if total >= GPU_BUDGET_MIN:
        raise SystemExit(f"GPU budget exhausted: {total:.1f}/{GPU_BUDGET_MIN:.0f} min used.")
    if total + extra_min > GPU_BUDGET_MIN:
        raise SystemExit(
            f"GPU budget: {total:.1f} min used, this launch needs ~{extra_min:.1f} min, cap is "
            f"{GPU_BUDGET_MIN:.0f} min. Ask before raising MODAL_GPU_BUDGET_MIN."
        )
    passed = [c for c in GPU_CHECKPOINTS if total >= c]
    if passed and not os.environ.get("MODAL_GPU_CONTINUE"):
        raise SystemExit(
            f"GPU checkpoint: {total:.1f} min used, past the {passed[-1]:.0f} min line. Report "
            "first; set MODAL_GPU_CONTINUE=1 to continue."
        )
    print(f"GPU used before this call: {total:.1f}/{GPU_BUDGET_MIN:.0f} min", flush=True)


def _estimate_run_seconds(spec: dict, warm: bool, control_params: dict | None = None) -> float:
    """Rough GPU seconds for one harness run (high side)."""
    if spec.get("kind") == "profile":
        return 120.0
    params = json.loads(spec.get("params") or "{}")
    base = control_params or {}
    epochs = float(params.get("epochs", base.get("epochs", 9.0)))
    width_scale = 1.0
    if "widths" in params:
        widths = params["widths"]
        width_scale = max(0.5, sum(widths) / (128 + 384 + 576))
    trial = EST_TRIAL_S_PER_EPOCH * epochs * width_scale + 0.3
    if params.get("compile", "max-autotune") == "":
        build = 20.0
    elif not warm:
        build = EST_BUILD_COLD_S
    elif any(k in params and params[k] != base.get(k) for k in GRAPH_KEYS):
        build = EST_BUILD_NEW_GRAPH_WARM_S
    else:
        build = EST_BUILD_WARM_S
    if params.get("low_res") and params.get("low_res_epochs"):
        build += EST_BUILD_LOW_RES_S
    # The baseline recipe compiles one static graph per (batch, resolution) pair: every low
    # resolution of resolution_schedule (default [[28, 0.5]]) and every extra batch size of
    # batch_schedule adds graphs. Our older recipes used res_schedule / train_resolution.
    schedule = params.get("resolution_schedule", params.get("res_schedule"))
    if schedule is None and "train_resolution" in params:
        schedule = [[params["train_resolution"], 1]] if params["train_resolution"] < 32 else []
    low_res = len(schedule) if schedule is not None else 1  # the default has one 28 px stage
    batches = 1 + len(params.get("batch_schedule") or [])
    build += EST_BUILD_LOW_RES_S * (low_res * batches + (batches - 1))
    if params.get("hard_fraction", 1.0) < 1.0:
        build += 300.0  # PR #5's offline proxy prepass compiles and trains a second network
    return EST_RUN_OVERHEAD_S + build + int(spec.get("n", 1)) * trial


def _estimate_job_minutes(
    specs: list[dict], warm: bool, control_params: dict | None = None
) -> float:
    seconds = EST_CONTAINER_S + sum(_estimate_run_seconds(s, warm, control_params) for s in specs)
    return (seconds + CONTAINER_START_S) / 60


def _call_gpu(specs: list[dict], require: dict, label: str, warm: bool = False, say=print) -> dict:
    """run_benchmarks.remote with the GPU-guard retry loop and ledger accounting."""
    guarded = require.get("gpu", "any") != "any" or str(require.get("power", "any")) != "any"
    attempts = GPU_ATTEMPTS if guarded else 1
    say(
        f"container cap {RUN_TIMEOUT} s; GPU guard {require} ({attempts} attempts, no fallback); "
        f"{'warm' if warm else 'cold'} compile cache"
    )
    landed: list[str] = []
    for attempt in range(1, attempts + 1):
        t0 = time.time()
        call = run_benchmarks.spawn(specs, require, float(RUN_TIMEOUT), warm)
        try:
            payload = call.get(timeout=RUN_TIMEOUT + SCHEDULE_WAIT_S)
        except modal.exception.FunctionTimeoutError:
            _ledger_add(f"{label} (TIMEOUT)", "?", float(RUN_TIMEOUT), note="Modal timeout")
            raise RuntimeError(
                f"{label}: Modal killed the container after {RUN_TIMEOUT} s. Finished runs are "
                "still in the cifar100-results Volume (`uv run modal volume ls cifar100-results`)."
            ) from None
        except (TimeoutError, modal.exception.TimeoutError):
            call.cancel()
            _ledger_add(f"{label} (NO START)", "?", 0.0, note="cancelled: never got a container")
            raise RuntimeError(
                f"{label}: no result after {RUN_TIMEOUT + SCHEDULE_WAIT_S:.0f} s; cancelled."
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
        gpu, power = env.get("gpu_name", "?"), env.get("gpu_power_limit")
        where = _where(env)
        landed.append(where)
        if not payload["gpu_mismatch"]:
            _ledger_add(label, gpu, payload["container_seconds"], power=power)
            payload["gpu_attempts"] = landed
            return payload
        _ledger_add(
            f"{label} (gpu-guard miss {attempt})",
            gpu,
            payload["container_seconds"],
            note=f"aborted before build: {_gpu_mismatch(env, require)}",
            power=power,
        )
        say(f"GPU guard: attempt {attempt}/{attempts} landed on {where}.")
    raise RuntimeError(f"{label}: no card matching {require} in {attempts} attempts: {landed}")


# --------------------------------------------------------------------------- local: reports


def _save_run(run: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for rel, blob in run["files"].items():
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    meta = {k: v for k, v in run.items() if k != "files"}
    (out / "modal_run.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if run.get("result_rel"):  # mirror for `just last`
        mirror = SPEEDRUN / "results" / run["result_rel"]
        mirror.mkdir(parents=True, exist_ok=True)
        for rel, blob in run["files"].items():
            (mirror / rel).parent.mkdir(parents=True, exist_ok=True)
            (mirror / rel).write_bytes(blob)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _config_label(spec: dict) -> str:
    return spec.get("config") or " ".join(spec["args"])


def _stats(run: dict, out: Path, environment: dict) -> dict:
    """Summarise one saved harness run (summary.json, trials.jsonl, config.json)."""
    spec = run["spec"]
    row: dict = {
        "label": spec.get("label", "run"),
        "params": spec.get("params", "{}"),
        "delta": spec.get("delta", {}),
        "hypothesis": spec.get("hypothesis", ""),
        "config": _config_label(spec),
        "returncode": run["returncode"],
        "folder": out.relative_to(REPO_ROOT).as_posix(),
        "gpu": environment.get("gpu_name", "?"),
        "power_limit": environment.get("gpu_power_limit", "?"),
        "task_id": environment.get("task_id"),
        "region": environment.get("region"),
        "cloud": environment.get("cloud"),
        "container_wall": run["wall_seconds"],
        "deadline_hit": bool(run.get("deadline_hit")),
        "nonfinite": run.get("nonfinite"),
        "build_mode": run.get("build_mode", "cold"),
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
        [json.loads(x) for x in trials_path.read_text(encoding="utf-8").splitlines() if x.strip()]
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
    if build_time is not None and build_time > 300 and row["build_mode"] == "cold":
        verdict += f"; BUILD > 300 s ({build_time:.0f} s cold)"
    if row["nonfinite"]:
        verdict += f"; NON-FINITE LOSSES: {row['nonfinite']}"
    several = s["successful_trials"] > 1
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
        by_trial={t["trial"]: (t["accuracy"], t["total_timed_time"]) for t in ok},
        seeds=[t.get("seed") for t in ok],
        accs=[t["accuracy"] for t in ok],
        times=[t["total_timed_time"] for t in ok],
        evals=[t["evaluation_time"] for t in ok],
        verdict=verdict,
    )
    return row


def _paired(control: dict, variant: dict) -> dict:
    """Per-trial differences variant minus control (same trial index = same seed)."""
    a, b = control.get("by_trial") or {}, variant.get("by_trial") or {}
    shared = sorted(set(a) & set(b))
    if not shared:
        return {}
    dacc = [b[i][0] - a[i][0] for i in shared]
    dtime = [b[i][1] - a[i][1] for i in shared]
    n = len(shared)
    se = statistics.stdev(dacc) / math.sqrt(n) if n > 1 else None
    return {
        "pairs": n,
        "dacc": statistics.mean(dacc),
        "dacc_se": se,
        "dtime": statistics.mean(dtime),
        "dtime_se": statistics.stdev(dtime) / math.sqrt(n) if n > 1 else None,
    }


def _load_k() -> float | None:
    """Exchange rate k in accuracy percentage points per second (from the epoch calibration)."""
    k = _read_json(K_PATH).get("k")
    return float(k) if k else None


def _dtime_adj(paired: dict, k: float | None) -> float | None:
    """Accuracy-adjusted saving: dtime minus the time the accuracy change is worth."""
    if not paired or not k:
        return None
    return paired["dtime"] - (100 * paired["dacc"]) / k


def _pct(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{100 * value:.{digits}f}"


def _sec(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _signed(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:+.{digits}f}"


def _print_row(row: dict, say=print) -> None:
    say(f"=== {row['folder']} ===")
    error = REPO_ROOT / row["folder"] / "error.txt"
    if error.exists():
        say("--- error.txt ---\n" + error.read_text(encoding="utf-8"))
    if "n" not in row:
        say(f"exit code {row['returncode']}: {row['verdict']} (see the container log above)")
        return
    say(f"exit code {row['returncode']}: {row['verdict']}")
    say(
        f"trials {row['trials_ok']}/{row['n']}  mean acc {_pct(row['mean_acc'])}% "
        f"(std {_pct(row['acc_std'])})  mean prep+train {_sec(row['mean_time'])} s "
        f"(std {_sec(row['time_std'])}; prepare {_sec(row['mean_prepare'], 3)} + "
        f"train {_sec(row['mean_train'])})  mean eval {_sec(row['mean_eval'], 3)} s  "
        f"build {_sec(row['build_time'])} s ({row['build_mode']})  nonfinite {row['nonfinite']}"
    )
    say(
        "per trial: "
        + ", ".join(
            f"{_pct(a)}% / {_sec(t)} s" for a, t in zip(row["accs"], row["times"], strict=True)
        )
    )
    if row.get("paired"):
        p = row["paired"]
        say(
            f"paired vs control (n={p['pairs']}): dacc {_signed(100 * p['dacc'])} pp "
            f"+- {_pct(p['dacc_se'])}, dtime {_signed(p['dtime'])} s, "
            f"dtime_adj {_signed(row.get('dtime_adj'))} s"
        )
    say(f"GPU: {row['gpu']}  power limit {row['power_limit']}")


def _log_row(row: dict, tag: str) -> str:
    if "n" not in row:
        return (
            f"| {datetime.now():%Y-%m-%d} | {tag} | {row['config']} | ? | n/a | n/a | n/a | "
            f"{row['gpu']} ({row['power_limit']}) | {row['verdict']} |"
        )
    extra = ""
    if row.get("paired"):
        p = row["paired"]
        extra = (
            f" paired vs control (n={p['pairs']}): dacc {_signed(100 * p['dacc'])} pp +- "
            f"{_pct(p['dacc_se'])}, dtime {_signed(p['dtime'])} s, dtime_adj "
            f"{_signed(row.get('dtime_adj'))} s."
        )
    hypothesis = f" Hypothesis: {row['hypothesis']}" if row.get("hypothesis") else ""
    return (
        f"| {datetime.now():%Y-%m-%d} | {tag} | {row['config']} | {row['n']} | "
        f"{_pct(row['mean_acc'])} | {_pct(row['acc_std'])} | {_sec(row['mean_time'])} | "
        f"{row['gpu']} ({row['power_limit']}) | {row['verdict']}; build {_sec(row['build_time'])} "
        f"s {row['build_mode']}, eval {_sec(row['mean_eval'], 3)} s, nonfinite "
        f"{row['nonfinite']}.{extra}{hypothesis} |"
    )


def _table(rows: list[dict]) -> str:
    head = (
        "| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | "
        "dtime s | dtime_adj s | build s | nonfinite | GPU @ power | verdict |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    lines = [head]
    for r in rows:
        gpu = f"{r['gpu']} @ {r['power_limit']}"
        if "n" not in r:
            lines.append(
                f"| {r['label']} | `{r['params']}` | ? | | | | | | | | | {gpu} | {r['verdict']} |"
            )
            continue
        p = r.get("paired") or {}
        dacc = f"{_signed(100 * p['dacc'])} +- {_pct(p.get('dacc_se'))}" if p else "control"
        dtime = _signed(p["dtime"]) if p else "control"
        adj = _signed(r.get("dtime_adj")) if p else "control"
        lines.append(
            f"| {r['label']} | `{json.dumps(r.get('delta', {}), sort_keys=True)}` | "
            f"{r['trials_ok']}/{r['n']} | {_pct(r['mean_acc'])} | {_pct(r['acc_std'])} | {dacc} | "
            f"{_sec(r['mean_time'])} | {dtime} | {adj} | {_sec(r['build_time'])} "
            f"({r['build_mode']}) | {r['nonfinite']} | {gpu} | {r['verdict']} |"
        )
    return "\n".join(lines)


def _registry_add(row: dict, round_name: str, job: str, control: dict | None, k: float | None):
    """Append one variant (or control) to artifacts/speedrun_runs/registry.jsonl."""
    entry = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "round": round_name,
        "job": job,
        "tag": f"{round_name}/{job}",
        "label": row["label"],
        "is_control": row["label"] == "control",
        "hypothesis": row.get("hypothesis", ""),
        "params": row.get("delta", {}),
        "control_params": json.loads(control["params"]) if control else {},
        "full_params": json.loads(row.get("params") or "{}"),
        "gpu": row["gpu"],
        "power_limit": row["power_limit"],
        "task_id": row.get("task_id"),
        "region": row.get("region"),
        "cloud": row.get("cloud"),
        "n": row.get("n"),
        "trials_ok": row.get("trials_ok"),
        "seeds": row.get("seeds"),
        "accs": row.get("accs"),
        "times": row.get("times"),
        "mean_acc": row.get("mean_acc"),
        "acc_std": row.get("acc_std"),
        "mean_time": row.get("mean_time"),
        "time_std": row.get("time_std"),
        "mean_prepare": row.get("mean_prepare"),
        "build_time": row.get("build_time"),
        "build_mode": row.get("build_mode"),
        "nonfinite": row.get("nonfinite"),
        "paired": row.get("paired") or None,
        "k_used": k,
        "dtime_adj": row.get("dtime_adj"),
        "verdict": row.get("verdict"),
        "folder": row.get("folder"),
        "returncode": row.get("returncode"),
    }
    with _LOCK:
        with REGISTRY_PATH.open("a", encoding="utf-8") as registry:
            registry.write(json.dumps(entry) + "\n")


def _log_append(lines: list[str]) -> None:
    with _LOCK:
        with LOG_PATH.open("a", encoding="utf-8") as log:
            log.write("\n".join(lines) + "\n")


def _make_spec(
    label: str,
    n: int,
    params: dict,
    seed: int,
    build_timeout: float,
    no_accuracy_target: bool,
    count_nonfinite: bool,
    hypothesis: str = "",
    delta: dict | None = None,
    env: dict | None = None,
) -> dict:
    full = {**params, "count_nonfinite": True} if count_nonfinite else dict(params)
    encoded = json.dumps(full, sort_keys=True)
    args = ["--submission", TEAM, "--n", str(n), "--params", encoded, "--seed", str(seed)]
    args += ["--build-timeout", str(build_timeout)]
    if no_accuracy_target:
        args.append("--no-accuracy-target")
    return {
        "label": label,
        "n": n,
        "args": args,
        "params": encoded,
        "delta": delta if delta is not None else params,
        "hypothesis": hypothesis,
        "env": env or {},
        "config": f"{TEAM} {json.dumps(delta if delta is not None else params, sort_keys=True)}",
    }


def _job_specs(job: dict, cfg: dict) -> list[dict]:
    """The harness runs of one job: the control first, then the variants (+ optional profile)."""
    name = job["name"]
    n = int(job.get("n", cfg["n"]))
    control_params = {**cfg["control"], **job.get("control_params", {})}
    specs = []
    if job.get("control", True):
        specs.append(
            _make_spec(
                "control",
                n,
                control_params,
                cfg["seed"],
                cfg["build_timeout"],
                cfg["no_target"],
                cfg["count_nonfinite"],
                hypothesis="control",
                delta={},
            )
        )
    for i, variant in enumerate(job["variants"]):
        label = variant.get("label") or f"v{i + 1:02d}"
        specs.append(
            _make_spec(
                label,
                int(variant.get("n", n)),
                {**control_params, **variant["params"]},
                cfg["seed"],
                cfg["build_timeout"],
                cfg["no_target"],
                cfg["count_nonfinite"],
                hypothesis=variant.get("hypothesis", ""),
                delta=variant["params"],
                env=variant.get("env"),
            )
        )
    if job.get("profile"):
        specs.append(
            {
                "kind": "profile",
                "label": "profile-control",
                "args": ["--params", json.dumps(control_params, sort_keys=True)],
                "params": json.dumps(control_params, sort_keys=True),
            }
        )
    labels = [s["label"] for s in specs]
    if len(set(labels)) != len(labels):
        raise RuntimeError(f"{name}: variant labels must be unique: {labels}")
    # Later runs in a container are 0.05-0.1 s slower (the card heats up), so a job may put the
    # control in the middle ("control_index": 1) instead of first.
    position = int(job.get("control_index", 0))
    if job.get("control", True) and 0 < position < len(specs):
        control = specs.pop(0)
        specs.insert(position, control)
    return specs


def _run_job(job: dict, cfg: dict) -> dict:
    """One container (sequential launcher path): call the GPU, then finish the job."""
    specs = _job_specs(job, cfg)
    say = _sayer(job["name"])
    say(f"{len(specs)} runs: " + ", ".join(f"{s['label']}={_config_label(s)}" for s in specs))
    payload = _call_gpu(specs, cfg["require"], f"{cfg['round']}/{job['name']}", cfg["warm"], say)
    return _finish_job(job, cfg, payload)


def _sayer(name: str):
    def say(message: str) -> None:
        print(f"[{name}] {message}", flush=True)

    return say


def _finish_job(job: dict, cfg: dict, payload: dict) -> dict:
    """Save a finished container's runs; stats, paired deltas, registry and LOG rows."""
    name = job["name"]
    say = _sayer(name)
    n = int(job.get("n", cfg["n"]))
    out = cfg["out_root"] / _safe(name)
    rows: list[dict] = []
    profile_report = None
    for i, run in enumerate(payload["runs"]):
        sub = out / f"{i:02d}_{_safe(run['spec']['label'])}"
        if run["spec"].get("kind") == "profile":
            sub.mkdir(parents=True, exist_ok=True)
            profile_report = run.get("profile") or {}
            (sub / "profile.json").write_text(
                json.dumps(profile_report, indent=2) + "\n", encoding="utf-8"
            )
            continue
        _save_run(run, sub)
        rows.append(_stats(run, sub, payload["environment"]))
    control = next((row for row in rows if row["label"] == "control"), None)
    for row in rows:
        if control is not None and row is not control:
            row["paired"] = _paired(control, row)
            row["dtime_adj"] = _dtime_adj(row["paired"], cfg["k"])
        _print_row(row, say)
    env = payload["environment"]
    table = _table(rows)
    header = (
        f"# {cfg['round']}/{name}\n\n{datetime.now():%Y-%m-%d %H:%M}, {env.get('gpu_name', '?')} "
        f"@ {env.get('gpu_power_limit', '?')} (task {env.get('task_id')}, "
        f"{env.get('cloud')}/{env.get('region')}), one container, sequential, {n} trial(s) per "
        f"run, seeds from {cfg['seed']}, {'warm' if cfg['warm'] else 'cold'} compile cache, "
        f"k = {cfg['k']}, container wall {payload['container_seconds']} s. GPU guard attempts: "
        f"{payload['gpu_attempts']}.\n\n"
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "ab_summary.md").write_text(header + table + "\n", encoding="utf-8")
    (out / "ab_summary.json").write_text(
        json.dumps(
            {
                "round": cfg["round"],
                "job": name,
                "environment": env,
                "gpu_attempts": payload["gpu_attempts"],
                "container_seconds": payload["container_seconds"],
                "rows": rows,
                "profile": profile_report,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    log_rows = [_log_row(row, f"{cfg['round']}/{name}/{row['label']}") for row in rows]
    if cfg["registry"]:
        for row in rows:
            _registry_add(row, cfg["round"], name, control, cfg["k"])
        _log_append(log_rows)
    say("\n" + header + table)
    return {
        "name": name,
        "rows": rows,
        "environment": env,
        "gpu_attempts": payload["gpu_attempts"],
        "container_seconds": payload["container_seconds"],
        "log_rows": log_rows,
        "folder": out.relative_to(REPO_ROOT).as_posix(),
    }


def _run_jobs(jobs: list[dict], cfg: dict, parallel: int) -> dict:
    """Run jobs on up to `parallel` containers at once from ONE thread: spawn, poll, retry.

    The Modal client is not safe to drive from several threads inside `modal run`, so each
    job is a small state machine: spawned -> (guard miss -> respawn) -> finished/failed.
    Finished jobs are saved immediately, so a failure elsewhere never loses them.
    """
    label_of = {job["name"]: f"{cfg['round']}/{job['name']}" for job in jobs}
    guarded = cfg["require"].get("gpu", "any") != "any" or str(cfg["require"].get("power")) != "any"
    attempts = GPU_ATTEMPTS if guarded else 1
    pending = list(jobs)
    active: dict[str, dict] = {}
    outcomes: dict[str, dict | Exception] = {}
    wait_limit = RUN_TIMEOUT + SCHEDULE_WAIT_S

    def launch(job: dict, state: dict) -> None:
        state["attempt"] += 1
        state["t0"] = time.time()
        state["call"] = run_benchmarks.spawn(
            state["specs"], cfg["require"], float(RUN_TIMEOUT), cfg["warm"]
        )
        _sayer(job["name"])(
            f"spawned attempt {state['attempt']}/{attempts} ({len(state['specs'])} runs, "
            f"{'warm' if cfg['warm'] else 'cold'} cache)"
        )

    while pending or active:
        while pending and len(active) < max(1, parallel):
            job = pending.pop(0)
            try:
                state = {"job": job, "specs": _job_specs(job, cfg), "attempt": 0, "landed": []}
                launch(job, state)
                active[job["name"]] = state
            except Exception as exc:  # noqa: BLE001 - keep the other jobs going
                outcomes[job["name"]] = exc
                print(f"[{job['name']}] FAILED to launch: {exc}", flush=True)
        for name, state in list(active.items()):
            job, say = state["job"], _sayer(name)
            try:
                payload = state["call"].get(timeout=0)
            except (TimeoutError, modal.exception.TimeoutError):
                if time.time() - state["t0"] > wait_limit:
                    state["call"].cancel()
                    _ledger_add(f"{label_of[name]} (NO START)", "?", 0.0, note="cancelled")
                    outcomes[name] = RuntimeError(f"no result after {wait_limit:.0f} s")
                    say(f"FAILED: {outcomes[name]}")
                    del active[name]
                continue
            except modal.exception.FunctionTimeoutError:
                _ledger_add(f"{label_of[name]} (TIMEOUT)", "?", float(RUN_TIMEOUT), note="timeout")
                outcomes[name] = RuntimeError(f"Modal killed the container after {RUN_TIMEOUT} s")
                say(f"FAILED: {outcomes[name]}")
                del active[name]
                continue
            except Exception as exc:  # noqa: BLE001
                _ledger_add(
                    f"{label_of[name]} (FAILED {type(exc).__name__})",
                    "?",
                    min(time.time() - state["t0"], float(RUN_TIMEOUT)),
                    note="local wall time of the failed call",
                )
                outcomes[name] = exc
                say(f"FAILED: {exc!r}")
                del active[name]
                continue
            env = payload["environment"]
            gpu, power = env.get("gpu_name", "?"), env.get("gpu_power_limit")
            state["landed"].append(_where(env))
            if payload["gpu_mismatch"]:
                _ledger_add(
                    f"{label_of[name]} (gpu-guard miss {state['attempt']})",
                    gpu,
                    payload["container_seconds"],
                    note=f"aborted before build: {_gpu_mismatch(env, cfg['require'])}",
                    power=power,
                )
                say(f"GPU guard: attempt {state['attempt']}/{attempts} landed on {_where(env)}.")
                if state["attempt"] < attempts:
                    launch(job, state)
                else:
                    outcomes[name] = RuntimeError(
                        f"no card matching {cfg['require']} in {attempts} attempts"
                    )
                    say(f"FAILED: {outcomes[name]}")
                    del active[name]
                continue
            _ledger_add(label_of[name], gpu, payload["container_seconds"], power=power)
            payload["gpu_attempts"] = state["landed"]
            del active[name]
            try:
                outcomes[name] = _finish_job(job, cfg, payload)
            except Exception as exc:  # noqa: BLE001 - results are in the Volume regardless
                outcomes[name] = exc
                say(f"FAILED while saving results: {exc!r}")
        if active:
            time.sleep(5)
    return outcomes


def _regenerate_leaderboard() -> None:
    if not LEADERBOARD.exists():
        print("scripts/leaderboard.py not found; skipping the leaderboard", flush=True)
        return
    try:
        subprocess.run([sys.executable, str(LEADERBOARD)], cwd=REPO_ROOT, timeout=120, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"leaderboard failed: {exc}", flush=True)


def _recipe_supports(key: str) -> bool:
    """Whether the submission's DEFAULTS mention a parameter (the harness rejects unknown keys)."""
    source = SPEEDRUN / "submissions" / TEAM / "submission.py"
    try:
        return f'"{key}"' in source.read_text(encoding="utf-8")
    except OSError:
        return False


def _nonfinite_switch(requested: bool) -> bool:
    if requested and not _recipe_supports("count_nonfinite"):
        print("note: this recipe has no count_nonfinite switch; non-finite losses not counted")
        return False
    return requested


def _split_launcher_flags(argv: tuple[str, ...]) -> tuple[dict, list[str]]:
    """Pull the launcher's own flags out of the pass-through harness flags."""
    opts = {
        "tag": "dev",
        "require_gpu": DEFAULT_REQUIRE_GPU,
        "require_power": DEFAULT_REQUIRE_POWER,
        "warm": False,
        "count_nonfinite": False,
        "round": "single",
    }
    valued = {
        "--tag": "tag",
        "--require-gpu": "require_gpu",
        "--require-power": "require_power",
        "--round": "round",
    }
    rest: list[str] = []
    args = list(argv)
    while args:
        arg = args.pop(0)
        key, _, inline = arg.partition("=")
        if key in valued and (inline or args):
            opts[valued[key]] = inline if inline else args.pop(0)
        elif arg == "--warm":
            opts["warm"] = True
        elif arg == "--count-nonfinite":
            opts["count_nonfinite"] = True
        else:
            rest.append(arg)
    return opts, rest


def _stamp(tag: str) -> Path:
    return ARTIFACTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{_safe(tag)}"


# --------------------------------------------------------------------------- entrypoints


@app.local_entrypoint()
def env_check():
    """GPU name/memory/power limit and container placement; no training."""
    _check_budget()
    t0 = time.time()
    call = gpu_environment.spawn()
    try:
        info = call.get(timeout=180 + SCHEDULE_WAIT_S)
    except (TimeoutError, modal.exception.TimeoutError):
        call.cancel()
        _ledger_add("env_check (NO START)", "?", 0.0)
        raise SystemExit(f"env_check: no container within {SCHEDULE_WAIT_S:.0f} s") from None
    except BaseException as exc:
        _ledger_add(f"env_check (FAILED {type(exc).__name__})", "?", min(time.time() - t0, 180.0))
        raise
    _ledger_add(
        "env_check",
        info.get("gpu_name", "?"),
        info.get("container_seconds", 0.0),
        power=info.get("gpu_power_limit"),
    )
    print(json.dumps(info, indent=2))


@app.local_entrypoint()
def main(*argv: str):
    """One benchmark.run on the GPU; harness flags pass through (default --n 1, cold build)."""
    opts, harness = _split_launcher_flags(argv)
    args = ["--submission", TEAM, *harness]
    if "--n" not in harness:
        args += ["--n", "1"]  # the harness defaults to the full 40 official trials
    params: dict = {}
    for i, arg in enumerate(args):
        if arg == "--params" and i + 1 < len(args):
            params = json.loads(args[i + 1])
    if _nonfinite_switch(opts["count_nonfinite"]):
        params = {**params, "count_nonfinite": True}
        if "--params" in args:
            args[args.index("--params") + 1] = json.dumps(params, sort_keys=True)
        else:
            args += ["--params", json.dumps(params, sort_keys=True)]
    n = int(args[args.index("--n") + 1])
    spec = {
        "label": opts["tag"],
        "n": n,
        "args": args,
        "params": json.dumps(params, sort_keys=True),
        "delta": params,
        "config": f"{TEAM} {' '.join(harness)}".strip(),
    }
    require = _parse_require(opts["require_gpu"], opts["require_power"])
    estimate = _estimate_job_minutes([spec], opts["warm"])
    print(f"estimate: ~{estimate:.1f} GPU-min (job limit {JOB_LIMIT_MIN:.0f} min)", flush=True)
    if estimate > JOB_LIMIT_MIN and not os.environ.get("MODAL_ALLOW_BIG"):
        raise SystemExit("this job is over the per-job limit: ask first, then MODAL_ALLOW_BIG=1")
    _check_budget(estimate)
    payload = _call_gpu([spec], require, opts["tag"], opts["warm"])
    run = payload["runs"][0]
    out = _stamp(opts["tag"])
    _save_run(run, out)
    row = _stats(run, out, payload["environment"])
    _print_row(row)
    print(f"GPU guard attempts: {payload['gpu_attempts']}")
    print(f"Results copied to {SPEEDRUN / 'results' / (run.get('result_rel') or '')}")
    _registry_add(row, opts["round"], opts["tag"], None, _load_k())
    log_row = _log_row(row, opts["tag"])
    _log_append([log_row])
    print("LOG.md row (appended):")
    print(log_row)
    if run["returncode"] != 0:
        sys.exit(run["returncode"] or 1)


def _screen_config(
    round_name: str,
    n: int,
    seed: int,
    control: dict,
    warm: bool,
    require: dict,
    count_nonfinite: bool,
    no_target: bool,
    build_timeout: float,
    registry: bool,
) -> dict:
    out_root = _stamp(round_name)
    out_root.mkdir(parents=True, exist_ok=True)
    return {
        "round": round_name,
        "n": n,
        "seed": seed,
        "control": control,
        "warm": warm,
        "require": require,
        "count_nonfinite": _nonfinite_switch(count_nonfinite),
        "no_target": no_target,
        "build_timeout": build_timeout,
        "k": _load_k(),
        "out_root": out_root,
        "registry": registry,
    }


def _plan(jobs: list[dict], cfg: dict, allow_big: bool) -> float:
    """Print the per-job GPU estimate, enforce the per-job limit, return the total."""
    total = 0.0
    for job in jobs:
        runs = 1 + len(job["variants"]) + (1 if job.get("profile") else 0)
        if len(job["variants"]) > 3:
            raise SystemExit(f"job {job['name']}: at most 3 variants per container")
        n = int(job.get("n", cfg["n"]))
        specs = [
            _make_spec(
                "x",
                int(v.get("n", n)),
                {**cfg["control"], **v["params"]},
                0,
                600,
                False,
                False,
                delta=v["params"],
            )
            for v in job["variants"]
        ]
        if job.get("control", True):
            specs.insert(0, _make_spec("control", n, cfg["control"], 0, 600, False, False))
        estimate = _estimate_job_minutes(specs, cfg["warm"], cfg["control"])
        total += estimate
        flag = "  <-- OVER THE PER-JOB LIMIT" if estimate > JOB_LIMIT_MIN else ""
        print(
            f"  job {job['name']}: {runs} runs x n={n}, ~{estimate:.1f} GPU-min{flag}", flush=True
        )
        if estimate > JOB_LIMIT_MIN and not allow_big:
            raise SystemExit(
                f"job {job['name']} is estimated at {estimate:.1f} GPU-min > {JOB_LIMIT_MIN:.0f}: "
                "split it, or ask first and pass --allow-big"
            )
        if estimate * 60 > RUN_TIMEOUT - DEADLINE_MARGIN_S:
            print(
                f"  WARNING: job {job['name']} may not fit the {RUN_TIMEOUT} s container cap; "
                "late runs would be skipped",
                flush=True,
            )
    print(f"total estimate ~{total:.1f} GPU-min for {len(jobs)} job(s)", flush=True)
    return total


@app.local_entrypoint()
def screen(
    jobs: str,
    parallel: int = 10,
    only: str = "",
    warm: bool = True,
    require_gpu: str = DEFAULT_REQUIRE_GPU,
    require_power: str = DEFAULT_REQUIRE_POWER,
    count_nonfinite: bool = True,
    no_accuracy_target: bool = False,
    build_timeout: float = 600.0,
    allow_big: bool = False,
):
    """Run a round of screening jobs, up to --parallel containers at once (one job each).

    --jobs: path to the jobs JSON ({"round", "n", "seed", "control", "jobs": [...]});
    --only: comma-separated job names to run. Every variant lands in registry.jsonl and
    LOG.md; the leaderboard is regenerated at the end.
    """
    plan = json.loads(Path(jobs).read_text(encoding="utf-8"))
    selected = plan["jobs"]
    if only:
        wanted = {x.strip() for x in only.split(",") if x.strip()}
        selected = [j for j in selected if j["name"] in wanted]
        missing = wanted - {j["name"] for j in selected}
        if missing:
            raise SystemExit(f"unknown job names: {sorted(missing)}")
    names = [j["name"] for j in selected]
    if len(set(names)) != len(names):
        raise SystemExit(f"job names must be unique: {names}")
    cfg = _screen_config(
        plan.get("round", Path(jobs).stem),
        int(plan.get("n", 8)),
        int(plan.get("seed", 0)),
        plan.get("control", {}),
        warm,
        _parse_require(require_gpu, require_power),
        count_nonfinite,
        no_accuracy_target,
        build_timeout,
        registry=True,
    )
    print(
        f"round {cfg['round']}: {len(selected)} job(s), n={cfg['n']}, seeds from {cfg['seed']}, "
        f"control {json.dumps(cfg['control'])}, k={cfg['k']}, parallel {parallel}, "
        f"{'warm' if warm else 'cold'} cache, guard {cfg['require']}",
        flush=True,
    )
    total = _plan(selected, cfg, allow_big)
    _check_budget(total)
    outcomes = _run_jobs(selected, cfg, parallel)
    print("\n" + "=" * 100 + f"\nround {cfg['round']} summary\n" + "=" * 100, flush=True)
    for name in names:
        outcome = outcomes.get(name)
        if isinstance(outcome, Exception) or outcome is None:
            print(f"\n## {name}: FAILED {outcome}", flush=True)
            continue
        env = outcome["environment"]
        print(
            f"\n## {name}: {env.get('gpu_name')} @ {env.get('gpu_power_limit')}, container "
            f"{outcome['container_seconds']} s, attempts {len(outcome['gpu_attempts'])}, saved "
            f"{outcome['folder']}",
            flush=True,
        )
        print(_table(outcome["rows"]), flush=True)
    print(f"\nGPU used: {_ledger_total():.1f}/{GPU_BUDGET_MIN:.0f} min", flush=True)
    _regenerate_leaderboard()
    if any(isinstance(o, Exception) for o in outcomes.values()):
        sys.exit(1)


@app.local_entrypoint()
def ab(
    tag: str = "ab",
    n: int = 2,
    control_params: str = "{}",
    variants: str = "[]",
    labels: str = "",
    hypotheses: str = "",
    no_accuracy_target: bool = False,
    build_timeout: float = 600.0,
    profile: bool = False,
    warm: bool = False,
    count_nonfinite: bool = True,
    require_gpu: str = DEFAULT_REQUIRE_GPU,
    require_power: str = DEFAULT_REQUIRE_POWER,
    seed: int = 0,
    round_name: str = "ab",
    registry: bool = True,
    allow_big: bool = False,
):
    """Control + variants sequentially in ONE container; paired comparison table.

    --variants: JSON list of parameter deltas merged over --control-params; --labels and
    --hypotheses: comma- and semicolon-separated names/notes. --profile appends a profiler
    run of the control. An empty delta re-runs the control (timing-noise check).
    """
    control = json.loads(control_params)
    deltas = json.loads(variants)
    if not isinstance(control, dict):
        raise SystemExit("--control-params must be a JSON object")
    if not isinstance(deltas, list) or not all(isinstance(d, dict) for d in deltas):
        raise SystemExit("--variants must be a JSON array of objects")
    if not deltas and not profile:
        raise SystemExit("Give at least one variant (--variants) or --profile")
    names = [x.strip() for x in labels.split(",") if x.strip()] if labels else []
    notes = [x.strip() for x in hypotheses.split(";")] if hypotheses else []
    job = {
        "name": tag,
        "profile": profile,
        "variants": [
            {
                "label": names[i] if i < len(names) else f"v{i + 1:02d}",
                "params": delta,
                "hypothesis": notes[i] if i < len(notes) else "",
            }
            for i, delta in enumerate(deltas)
        ],
    }
    cfg = _screen_config(
        round_name,
        n,
        seed,
        control,
        warm,
        _parse_require(require_gpu, require_power),
        count_nonfinite,
        no_accuracy_target,
        build_timeout,
        registry,
    )
    total = _plan([job], cfg, allow_big)
    _check_budget(total)
    outcome = _run_job(job, cfg)
    print(f"\nsaved: {outcome['folder']}/ab_summary.md")
    print(f"GPU used: {_ledger_total():.1f}/{GPU_BUDGET_MIN:.0f} min")
    if registry:
        _regenerate_leaderboard()
    if any(row["returncode"] != 0 for row in outcome["rows"]):
        sys.exit(1)


@app.local_entrypoint()
def profile(
    tag: str = "profile",
    params: str = "{}",
    epochs: int = 2,
    require_gpu: str = DEFAULT_REQUIRE_GPU,
    require_power: str = DEFAULT_REQUIRE_POWER,
):
    """Where does prepare+train time go? Runs scripts/profile_recipe.py on the real data."""
    spec = {
        "kind": "profile",
        "label": tag,
        "args": ["--params", params, "--epochs", str(epochs)],
        "params": params,
    }
    _check_budget(3.0)
    payload = _call_gpu([spec], _parse_require(require_gpu, require_power), tag)
    out = _stamp(tag)
    out.mkdir(parents=True, exist_ok=True)
    report = payload["runs"][0].get("profile") or {}
    report["environment"] = payload["environment"]
    (out / "profile.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {(out / 'profile.json').relative_to(REPO_ROOT).as_posix()}")
    print(f"GPU guard attempts: {payload['gpu_attempts']}")
