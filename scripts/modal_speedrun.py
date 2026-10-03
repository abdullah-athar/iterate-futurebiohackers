"""Run the CIFAR-100 speedrun harness on a Modal A100-80GB GPU.

This file lives in the TEAM repo (scripts/), outside cifar100-speedrun/, so the upstream
PR contains only submissions/futurebiohackers/. Run it from the team env at the repo root
(Windows: set PYTHONUTF8=1 first). `modal run` needs the entrypoint name after `::`.

    uv run modal run scripts/modal_speedrun.py::main --n 3 --params '{"epochs": 10}'
    uv run modal run scripts/modal_speedrun.py::ab --n 8 --tag widths \
        --variants '[{"widths": [128, 320, 768]}]' --labels w320-768
    uv run modal run scripts/modal_speedrun.py::profile --params '{}'
    uv run modal run scripts/modal_speedrun.py::env_check

::main: harness flags pass straight through to `benchmark.run` (`--n`, `--params`,
`--no-accuracy-target`, `--seed`, `--build-timeout`, `--submission-path`, ...). CIFAR-100 is
cached in the `cifar100-data` Volume (downloaded on first use); results land in the
`cifar100-results` Volume and are copied back to cifar100-speedrun/results/ (so `just last`
works) and to artifacts/speedrun_runs/<timestamp>_<tag>/. Launcher-only flags: --tag T,
--no-require-pcie, --no-pcie-fallback.

PCIe guard (default on): Modal's A100-80GB pool mixes the judges' A100 80GB PCIe (300 W)
with SXM4 parts (400-500 W). The container reads nvidia-smi BEFORE any build; on a non-PCIe
card it returns at once (about 0.3 GPU-min) and the launcher retries on a fresh single-use
container, up to MODAL_PCIE_ATTEMPTS calls (default 4), logging GPU, task id, region and
cloud per attempt. After that it falls back to whatever card it gets and every result row
carries the GPU name (--no-pcie-fallback aborts instead).

A/B mode (::ab): the control (defaults, or --control-params) and every --variants entry
(parameter deltas merged over the control) run SEQUENTIALLY in ONE container on ONE card,
each as its own `benchmark.run` process (fresh build, cold torch.compile cache, nothing
shared). Harness seeds are deterministic by trial index, so trial i of the control and of
each variant share init, data order and augmentation: the table reports paired per-seed
dacc +- SE and dtime next to the usual mean / std / build time / GPU. --profile adds a run
of scripts/profile_recipe.py on the control at the end.

Time caps: every GPU function has a hard Modal timeout (MODAL_TIMEOUT_MIN, default 20 min).
Inside the container each run has a deadline: the harness gets SIGINT (it keeps finished
trials) and later runs are skipped, so the payload always comes back.

GPU budget: each container's wall time plus a start-up allowance is appended to
artifacts/speedrun_runs/gpu_ledger.jsonl and "GPU used: X/120 min" is printed after every
call. MODAL_GPU_BUDGET_MIN (120) is a hard stop; MODAL_GPU_CHECKPOINTS (60,100) are report
lines that need MODAL_GPU_CONTINUE=1 to pass.

Set TEAM for another submission folder and MODAL_GPU for another GPU type (the PCIe guard
is then off); official judging uses an A100 80GB PCIe.
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
PROFILER = REPO_ROOT / "scripts" / "profile_recipe.py"
REMOTE = "/root/speedrun"
REMOTE_PROFILER = "/root/profile_recipe.py"
DATA_ROOT = "/data"
RESULTS_ROOT = "/results"
TEAM = os.environ.get("TEAM", "futurebiohackers")
GPU = os.environ.get("MODAL_GPU", "A100-80GB") or None  # empty: CPU-only
OFFICIAL_GPU = "NVIDIA A100 80GB PCIe"
GUARD_POSSIBLE = GPU == "A100-80GB"  # another GPU type can never be the judges' card

RUN_TIMEOUT = int(os.environ.get("MODAL_TIMEOUT_MIN", "20")) * 60  # hard cap per container
DEADLINE_MARGIN_S = 60.0  # the container stops starting work this long before RUN_TIMEOUT
MIN_RUN_S = 90.0  # do not start a run with less container time left than this
SCHEDULE_WAIT_S = 240.0  # extra wait for Modal to find a container
CONTAINER_START_S = 15.0  # billed start-up allowance per container, not measurable inside it
PCIE_ATTEMPTS = int(os.environ.get("MODAL_PCIE_ATTEMPTS", "4"))
GPU_BUDGET_MIN = float(os.environ.get("MODAL_GPU_BUDGET_MIN", "120"))
GPU_CHECKPOINTS = [
    float(x) for x in os.environ.get("MODAL_GPU_CHECKPOINTS", "60,100").split(",") if x.strip()
]

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
    name = info.get("gpu_name", "")
    if name and name != OFFICIAL_GPU:
        print(f"WARNING: GPU is '{name}', not the judges' '{OFFICIAL_GPU}' (300 W).", flush=True)
    return info


def _ensure_data() -> None:
    if not Path(f"{DATA_ROOT}/cifar-100-python").exists():
        subprocess.run(
            [sys.executable, "-m", "benchmark.data", "--root", DATA_ROOT], cwd=REMOTE, check=True
        )
        data.commit()


def _popen_with_deadline(cmd: list[str], env: dict, deadline: float | None, label: str) -> dict:
    """Run cmd, stream its output, SIGINT it at the deadline (SIGKILL 45 s later)."""
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
        print(line, end="", flush=True)
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


def _run(spec: dict, deadline: float | None) -> dict:
    """One benchmark.run process (own build, own worker, cold compile cache)."""
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
    # Cold compile caches for every run: a compiled variant must not inherit the kernels or
    # autotune results of an earlier one (the judges' container is cold too).
    cache_dir = tempfile.mkdtemp(prefix=f"inductor-{_safe(label)}-")
    env["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    env["TRITON_CACHE_DIR"] = f"{cache_dir}/triton"
    print(f"\n##### {label} #####\n$ {shlex.join(cmd)}", flush=True)
    run = _popen_with_deadline(cmd, env, deadline, label)
    shutil.rmtree(cache_dir, ignore_errors=True)
    result_dir = None
    for line in run.pop("lines"):
        if line.startswith("Results: "):
            result_dir = line[len("Results: ") :].strip()
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
    volumes={DATA_ROOT: data, RESULTS_ROOT: results},
)
def run_benchmarks(specs: list, require_pcie: bool, budget_s: float) -> dict:
    """Run each spec sequentially in THIS container (one card), each a fresh harness run.

    With require_pcie, a card other than the judges' A100 80GB PCIe aborts before any
    build or download; the launcher retries on a fresh container.
    """
    started = time.time()
    deadline = started + budget_s - DEADLINE_MARGIN_S
    environment = _environment()
    payload: dict = {"environment": environment, "gpu_mismatch": False, "runs": []}
    if require_pcie and environment.get("gpu_name") != OFFICIAL_GPU:
        payload["gpu_mismatch"] = True
        print(f"PCIe guard: not '{OFFICIAL_GPU}'; aborting before build.", flush=True)
    else:
        _ensure_data()
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
                payload["runs"].append(_run(spec, deadline))
                results.commit()  # keep finished runs even if a later one hangs
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


def _check_budget() -> None:
    total = _ledger_total()
    if total >= GPU_BUDGET_MIN:
        raise SystemExit(f"GPU budget exhausted: {total:.1f}/{GPU_BUDGET_MIN:.0f} min used.")
    passed = [c for c in GPU_CHECKPOINTS if total >= c]
    if passed and not os.environ.get("MODAL_GPU_CONTINUE"):
        raise SystemExit(
            f"GPU checkpoint: {total:.1f} min used, past the {passed[-1]:.0f} min line. Report "
            "first; set MODAL_GPU_CONTINUE=1 to continue."
        )
    print(f"GPU used before this call: {total:.1f}/{GPU_BUDGET_MIN:.0f} min", flush=True)


def _call_gpu(specs: list[dict], require_pcie: bool, pcie_fallback: bool, label: str) -> dict:
    """run_benchmarks.remote with the budget check, the PCIe retry loop and the ledger."""
    _check_budget()
    if require_pcie and not GUARD_POSSIBLE:
        print(f"PCIe guard off: MODAL_GPU={GPU!r} can never be the judges' card.", flush=True)
        require_pcie = False
    print(f"container cap {RUN_TIMEOUT} s (MODAL_TIMEOUT_MIN); PCIe guard {require_pcie}")
    landed: list[str] = []
    attempts = PCIE_ATTEMPTS if require_pcie else 1
    for attempt in range(1, attempts + 2):
        guard = require_pcie and attempt <= attempts
        if require_pcie and not guard:
            if not pcie_fallback:
                raise SystemExit(
                    f"PCIe guard: no '{OFFICIAL_GPU}' in {attempts} attempts: {landed}"
                )
            print(
                f"PCIe guard: no '{OFFICIAL_GPU}' in {attempts} attempts; FALLING BACK to any "
                "A100-80GB. Every result row is labelled with the GPU used.",
                flush=True,
            )
        t0 = time.time()
        call = run_benchmarks.spawn(specs, guard, float(RUN_TIMEOUT))
        try:
            payload = call.get(timeout=RUN_TIMEOUT + SCHEDULE_WAIT_S)
        except modal.exception.FunctionTimeoutError:
            _ledger_add(f"{label} (TIMEOUT)", "?", float(RUN_TIMEOUT), note="Modal timeout")
            raise SystemExit(
                f"Modal killed the container after {RUN_TIMEOUT} s. Finished runs are still in "
                "the cifar100-results Volume (`uv run modal volume ls cifar100-results`)."
            ) from None
        except (TimeoutError, modal.exception.TimeoutError):
            call.cancel()
            _ledger_add(f"{label} (NO START)", "?", float(RUN_TIMEOUT), note="cancelled")
            raise SystemExit(
                f"no result after {RUN_TIMEOUT + SCHEDULE_WAIT_S:.0f} s; the call was cancelled."
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
        where = f"{gpu} [task {env.get('task_id')} {env.get('cloud')}/{env.get('region')}]"
        landed.append(where)
        if not payload["gpu_mismatch"]:
            _ledger_add(label, gpu, payload["container_seconds"])
            payload["pcie_attempts"] = landed
            payload["on_pcie"] = gpu == OFFICIAL_GPU
            return payload
        _ledger_add(
            f"{label} (pcie-guard attempt {attempt})",
            gpu,
            payload["container_seconds"],
            note="aborted before build",
        )
        print(f"PCIe guard: attempt {attempt}/{attempts} landed on {where}.", flush=True)
    raise AssertionError("unreachable")


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
        print(f"Results copied to {mirror}")


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
    if build_time is not None and build_time > 300:
        verdict += f"; BUILD > 300 s ({build_time:.0f} s cold)"
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


def _pct(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{100 * value:.{digits}f}"


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
    extra = ""
    if row.get("paired"):
        p = row["paired"]
        extra = (
            f" paired vs control (n={p['pairs']}): dacc {_pct(p['dacc'], 2):+} pp +- "
            f"{_pct(p['dacc_se'], 2)}, dtime {p['dtime']:+.2f} s."
        )
    return (
        f"| {datetime.now():%Y-%m-%d} | {tag} | {row['config']} | {row['n']} | "
        f"{_pct(row['mean_acc'])} | {_pct(row['acc_std'])} | {_sec(row['mean_time'])} | "
        f"{row['gpu']} ({row['power_limit']}) | {row['verdict']}; build {_sec(row['build_time'])} "
        f"s cold, eval {_sec(row['mean_eval'], 3)} s.{extra} |"
    )


def _table(rows: list[dict]) -> str:
    head = (
        "| variant | params | n | mean acc % | std | paired dacc pp +- SE | mean prep+train s | "
        "dtime s | prepare s | build s cold | GPU | verdict |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    lines = [head]
    for r in rows:
        if "n" not in r:
            lines.append(
                f"| {r['label']} | `{r['params']}` | ? | | | | | | | | {r['gpu']} | {r['verdict']} |"
            )
            continue
        p = r.get("paired") or {}
        dacc = f"{_pct(p['dacc']):+} +- {_pct(p.get('dacc_se'))}" if p else "control"
        dtime = f"{p['dtime']:+.2f}" if p else "control"
        lines.append(
            f"| {r['label']} | `{r['params']}` | {r['trials_ok']}/{r['n']} | "
            f"{_pct(r['mean_acc'])} | {_pct(r['acc_std'])} | {dacc} | {_sec(r['mean_time'])} | "
            f"{dtime} | {_sec(r['mean_prepare'], 3)} | {_sec(r['build_time'])} | {r['gpu']} | "
            f"{r['verdict']} |"
        )
    return "\n".join(lines)


def _split_launcher_flags(argv: tuple[str, ...]) -> tuple[dict, list[str]]:
    """Pull --tag/--no-require-pcie/--no-pcie-fallback out of the pass-through harness flags."""
    opts = {"tag": "dev", "require_pcie": True, "pcie_fallback": True}
    rest: list[str] = []
    args = list(argv)
    while args:
        arg = args.pop(0)
        if arg == "--tag" and args:
            opts["tag"] = args.pop(0)
        elif arg.startswith("--tag="):
            opts["tag"] = arg.split("=", 1)[1]
        elif arg == "--no-require-pcie":
            opts["require_pcie"] = False
        elif arg == "--no-pcie-fallback":
            opts["pcie_fallback"] = False
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
    _ledger_add("env_check", info.get("gpu_name", "?"), info.get("container_seconds", 0.0))
    print(json.dumps(info, indent=2))


@app.local_entrypoint()
def main(*argv: str):
    """One benchmark.run on the GPU; harness flags pass through (default --n 1)."""
    opts, harness = _split_launcher_flags(argv)
    args = ["--submission", TEAM, *harness]
    if "--n" not in harness:
        args += ["--n", "1"]  # the harness defaults to the full 40 official trials
    spec = {"label": opts["tag"], "args": args, "config": f"{TEAM} {' '.join(harness)}".strip()}
    for i, arg in enumerate(harness):
        if arg == "--params" and i + 1 < len(harness):
            spec["params"] = harness[i + 1]
    payload = _call_gpu([spec], opts["require_pcie"], opts["pcie_fallback"], opts["tag"])
    run = payload["runs"][0]
    out = _stamp(opts["tag"])
    _save_run(run, out)
    row = _stats(run, out, payload["environment"])
    _print_row(row)
    print(f"PCIe guard attempts: {payload['pcie_attempts']}")
    print("LOG.md row:")
    print(_log_row(row, opts["tag"]))
    if run["returncode"] != 0:
        sys.exit(run["returncode"] or 1)


@app.local_entrypoint()
def ab(
    tag: str = "ab",
    n: int = 2,
    control_params: str = "{}",
    variants: str = "[]",
    labels: str = "",
    no_accuracy_target: bool = False,
    build_timeout: float = 600.0,
    profile: bool = False,
    require_pcie: bool = True,
    pcie_fallback: bool = True,
    seed: int = 0,
):
    """Control + variants sequentially in ONE container; paired comparison table.

    --variants: JSON list of parameter deltas merged over --control-params; --labels:
    comma-separated names (default v01, v02, ...). --profile appends a profiler run of the
    control. Variants with an empty delta re-run the control (timing-noise check).
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

    def make(label: str, params: dict) -> dict:
        encoded = json.dumps(params, sort_keys=True)
        args = ["--submission", TEAM, "--n", str(n), "--params", encoded, "--seed", str(seed)]
        args += ["--build-timeout", str(build_timeout)]
        if no_accuracy_target:
            args.append("--no-accuracy-target")
        return {"label": label, "args": args, "params": encoded, "config": f"{TEAM} {encoded}"}

    specs = [make("control", control)]
    for i, delta in enumerate(deltas):
        specs.append(make(names[i] if i < len(names) else f"v{i + 1:02d}", {**control, **delta}))
    if profile:
        specs.append(
            {
                "kind": "profile",
                "label": "profile-control",
                "args": ["--params", json.dumps(control, sort_keys=True)],
                "params": json.dumps(control, sort_keys=True),
            }
        )
    seen = [s["label"] for s in specs]
    if len(set(seen)) != len(seen):
        raise SystemExit(f"variant labels must be unique: {seen}")
    print(f"A/B '{tag}': {len(specs)} runs x {n} trials, sequential in one {GPU} container:")
    for spec in specs:
        print(f"  {spec['label']:>16}: {_config_label(spec)}")
    payload = _call_gpu(specs, require_pcie, pcie_fallback, tag)
    out = _stamp(tag)
    rows = []
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
    for row in rows[1:]:
        row["paired"] = _paired(rows[0], row)
    for row in rows:
        _print_row(row)
    env = payload["environment"]
    table = _table(rows)
    header = (
        f"# A/B {tag}\n\n{datetime.now():%Y-%m-%d %H:%M}, {env.get('gpu_name', '?')} "
        f"(power limit {env.get('gpu_power_limit', '?')}, task {env.get('task_id')}, "
        f"{env.get('cloud')}/{env.get('region')}), one container, sequential, {n} trial(s) per "
        f"variant, seeds from {seed}, container wall {payload['container_seconds']} s. PCIe "
        f"guard attempts: {payload['pcie_attempts']}.\n\n"
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
                "profile": profile_report,
            },
            indent=2,
            default=str,
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
def profile(
    tag: str = "profile",
    params: str = "{}",
    epochs: int = 2,
    require_pcie: bool = True,
    pcie_fallback: bool = True,
):
    """Where does prepare+train time go? Runs scripts/profile_recipe.py on the real data."""
    spec = {
        "kind": "profile",
        "label": tag,
        "args": ["--params", params, "--epochs", str(epochs)],
        "params": params,
    }
    payload = _call_gpu([spec], require_pcie, pcie_fallback, tag)
    out = _stamp(tag)
    out.mkdir(parents=True, exist_ok=True)
    report = payload["runs"][0].get("profile") or {}
    report["environment"] = payload["environment"]
    (out / "profile.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {(out / 'profile.json').relative_to(REPO_ROOT).as_posix()}")
    print(f"PCIe guard attempts: {payload['pcie_attempts']}")
