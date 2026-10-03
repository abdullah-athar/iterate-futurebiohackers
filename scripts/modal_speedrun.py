"""Run the CIFAR-100 speedrun harness on a Modal A100-80GB GPU.

Arguments pass straight through to `benchmark.run`. CIFAR-100 is cached in the
`cifar100-data` volume; results land in `cifar100-results` and are copied back
to cifar100-speedrun/results/ so `just last` works. Set MODAL_GPU to try another
GPU type; official judging uses an A100 80GB PCIe.

    uv run modal run scripts/modal_speedrun.py --n 3 --params '{"epochs": 10}'
"""

import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import modal

SPEEDRUN = Path(__file__).resolve().parents[1] / "cifar100-speedrun"
REMOTE = "/root/speedrun"
TEAM = os.environ.get("TEAM", "futurebiohackers")
GPU = os.environ.get("MODAL_GPU", "A100-80GB") or None  # empty: CPU-only

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(SPEEDRUN))
    .env({"PYTHONPATH": REMOTE})
    .add_local_dir(
        SPEEDRUN, REMOTE, ignore=["**/.venv", "**/__pycache__", "data", "results", "**/.DS_Store"]
    )
)
app = modal.App("cifar100-speedrun", image=image)
data = modal.Volume.from_name("cifar100-data", create_if_missing=True)
results = modal.Volume.from_name("cifar100-results", create_if_missing=True)


@app.function(
    gpu=GPU,
    cpu=4,
    timeout=4 * 60 * 60,
    volumes={"/data": data, "/results": results},
)
def run(args: list[str]) -> tuple[int, bytes]:
    if not Path("/data/cifar-100-python").exists():
        subprocess.run([sys.executable, "-m", "benchmark.data", "--root", "/data"], cwd=REMOTE, check=True)
        data.commit()
    before = set(Path("/results").glob("*/*"))
    cmd = [sys.executable, "-m", "benchmark.run", "--data-root", "/data", "--results-root", "/results"]
    code = subprocess.run([*cmd, *args], cwd=REMOTE).returncode
    results.commit()
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for path in set(Path("/results").glob("*/*")) - before:
            tar.add(path, arcname=str(path.relative_to("/results")))
    return code, archive.getvalue()


@app.local_entrypoint()
def main(*argv: str):
    args = ["--submission", TEAM, *argv]
    if "--n" not in argv:
        args += ["--n", "1"]  # the harness defaults to the full 40 official trials
    code, archive = run.remote(args)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(SPEEDRUN / "results", filter="data")
        for name in sorted({Path(m.name).parts[:2] for m in tar.getmembers()}):
            print(f"Results copied to {SPEEDRUN / 'results' / Path(*name)}")
    if code:
        sys.exit(code)
