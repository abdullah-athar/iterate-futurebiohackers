"""Refresh the local Markdown/HTML progress log from streamed experiment results."""

import ast
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/runtime-optimization"
VISUAL = ROOT / ".lavish"


def describe_change(record):
    p = record.get("params", {})
    if record["name"].startswith("selected-default") and record.get("result_path"):
        source = (
            OUT
            / Path(record["result_path"]).relative_to("/results")
            / "source/submission.py"
        )
        if source.exists():
            for node in ast.parse(source.read_text()).body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "DEFAULTS" for t in node.targets
                ):
                    values = {
                        k.value: ast.literal_eval(v)
                        for k, v in zip(node.value.keys, node.value.values)
                        if isinstance(k, ast.Constant)
                        and k.value in ("widths", "epochs")
                    }
                    p = {**values, **p}
    if record.get("control") or "Initial reproduction" in record["name"]:
        return "Frozen PR #3 control"
    changes = []
    if p.get("hard_fraction", 1) < 1:
        changes.append(f"Proxy keeps hardest {p['hard_fraction']:.0%}")
    if any(p.get("pool_first", [])):
        groups = ",".join(
            str(i + 1) for i, enabled in enumerate(p["pool_first"]) if enabled
        )
        changes.append(f"Pool before conv in groups {groups}")
    if p.get("train_resolution", 32) < 32:
        changes.append(
            f"First {p.get('resolution_switch', 0.5):.0%} at {p['train_resolution']}px; finish 32px"
        )
    if "widths" in p:
        changes.append("Widths " + "/".join(map(str, p["widths"])))
    if "depths" in p:
        changes.append("Depths " + "/".join(map(str, p["depths"])))
    if p.get("fused_bn_gelu"):
        changes.append("Triton BatchNorm + GELU forward/backward fusion")
    if p.get("fused_sgd"):
        changes.append("Fused SGD")
    if p.get("compile_loss"):
        changes.append("Compiled loss")
    if p.get("gelu_approximate") == "tanh":
        changes.append("Approximate GELU")
    if "batch_size" in p:
        changes.append(f"Batch size {p['batch_size']}")
    if "lr" in p:
        changes.append(f"Learning rate {p['lr']}")
    if "label_smoothing" in p:
        changes.append(f"Label smoothing {p['label_smoothing']}")
    if not changes:
        changes.append(
            {
                "indexed": "Indexed channels-last crop",
                "triton": "Triton crop + flip",
            }.get(p.get("crop_mode"), "Global max-pool; baseline training settings")
        )
    if "epochs" in p:
        changes.append(f"{p['epochs']} epochs")
    return "; ".join(changes)


def refresh():
    records = {}
    baseline = (
        ROOT
        / "cifar100-speedrun/results/futurebiohackers/20261003T134539Z-13d43b3a/summary.json"
    )
    if baseline.exists():
        summary = json.loads(baseline.read_text())
        records[summary["run_id"]] = {
            "name": "Initial reproduction (different allocation)",
            "summary": summary,
        }
    for log in sorted(OUT.glob("*.log")):
        paired_control = None
        for line in log.read_text(errors="replace").splitlines():
            if not line.startswith('{"name":'):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = record.get("summary", {}).get("run_id", log.name + record["name"])
            record["stage"] = log.stem
            if record.get("control") and record.get("summary", {}).get("complete"):
                paired_control = record["summary"]
            if paired_control:
                record["control_seconds"] = paired_control["mean_training_time"]
                record["control_run_id"] = paired_control["run_id"]
            records[key] = record
    rows = []
    for key, record in sorted(records.items()):
        s = record.get("summary", {})
        acc, seconds = s.get("mean_accuracy"), s.get("mean_training_time")
        complete = s.get("complete", False)
        qualified = complete and acc is not None and acc > 0.75
        rows.append(
            {
                "name": record["name"],
                "main_change": describe_change(record),
                "stage": record.get("stage", "initial"),
                "error": s.get("run_error"),
                "trials": s.get("number_of_trials", 0),
                "accuracy": round(acc * 100, 3) if acc is not None else None,
                "seconds": round(seconds, 3) if seconds is not None else None,
                "status": (
                    "40-seed validated"
                    if qualified and s.get("number_of_trials") == 40
                    else "Qualifying development run"
                    if qualified
                    else "Below target"
                    if complete
                    else "Failed/incomplete"
                ),
                "params": record.get("params", {}),
                "run_id": key,
                "qualifies": qualified,
                "is_control": bool(record.get("control")),
                "control_seconds": record.get("control_seconds"),
                "control_run_id": record.get("control_run_id"),
            }
        )
    for row in rows:
        control = row["control_seconds"]
        if row["name"] == "control":
            row["effect"] = "Reference"
        elif control and row["seconds"] is not None:
            delta = 100 * (1 - row["seconds"] / control)
            row["effect"] = (
                f"{abs(delta):.1f}% {'faster' if delta >= 0 else 'slower'} (screen)"
            )
        else:
            row["effect"] = (
                "No paired control"
                if row["seconds"] is not None
                else row.get("error") or "Incomplete"
            )
    rows.sort(key=lambda r: r["seconds"] if r["seconds"] is not None else float("inf"))
    status_file = OUT / "status.json"
    status = json.loads(status_file.read_text()) if status_file.exists() else {}
    billing = OUT / "billing-current.json"
    start = OUT / "billing-start.json"
    spend = (
        max(
            0,
            float(json.loads(billing.read_text())["metered_cost"])
            - float(json.loads(start.read_text())["metered_cost"]),
        )
        if billing.exists() and start.exists()
        else None
    )
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    payload = {
        "updated": stamp,
        "status": status,
        "metered_spend_delta": spend,
        "rows": rows,
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2))
    md = [
        "# CIFAR-100 runtime optimization — live log",
        "",
        f"Updated: {stamp}",
        "",
        "Target: <3.00 s prepare + train; >75% mean accuracy over 40 fresh trials. Budget: $50.",
        "Hardware: Modal A100 SXM. Compare recipes against the control in the same allocation.",
        "",
        f"Current: {status.get('current', 'Starting experiments')}",
        "",
        f"Latest metered workspace cost increase: ${spend:.2f} (billing can lag; includes other workspace activity)."
        if spend is not None
        else "Billing: pending.",
        "",
        "| Experiment | Main change tested | Trials | Accuracy | Prepare + train | Status |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['name']} | {r['main_change']} | {r['trials']} | {r['accuracy'] if r['accuracy'] is not None else '—'}% | {r['seconds'] if r['seconds'] is not None else '—'} s | {r['status']} |"
        )
    md.extend(["", "## Notes", ""] + [f"- {note}" for note in status.get("notes", [])])
    md.extend(
        ["", "## Exact settings", "", "```json", json.dumps(rows, indent=2), "```", ""]
    )
    (OUT / "progress.md").write_text("\n".join(md))
    VISUAL.mkdir(exist_ok=True)
    (VISUAL / "metrics.json").write_text(json.dumps(payload))
    (VISUAL / "metrics.js").write_text(
        "window.speedrunMetricsLoaded(" + json.dumps(payload) + ");"
    )
    if not (VISUAL / "runtime-progress.html").exists():
        (VISUAL / "runtime-progress.html").write_text("""<!doctype html>
<html lang="en" data-theme="luxury"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>CIFAR-100 · live experiments</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/daisyui@5.5.19/daisyui.css"><link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/daisyui@5.5.19/themes.css"><script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4.2.4/dist/index.global.js"></script>
<style>*{box-sizing:border-box}main,section,div{min-width:0}td,th,p,li{overflow-wrap:anywhere}table{table-layout:fixed;width:100%}details pre{white-space:pre-wrap;overflow-wrap:anywhere}</style></head>
<body><main class="max-w-6xl mx-auto p-5 md:p-10"><p class="text-primary text-sm tracking-widest">FUTUREBIOHACKERS / CIFAR-100</p><h1 class="text-3xl md:text-5xl font-bold mt-3 mb-4">Every second counts.</h1><p class="text-base-content/70">Target: under 3 seconds, above 75% mean accuracy, 40 fresh trials. Modal A100 SXM development measurements.</p>
<section class="grid grid-cols-1 md:grid-cols-3 gap-4 my-7"><div class="card bg-base-200 p-5"><p class="text-sm text-base-content/60">Fastest qualifying development run</p><p id="best" class="text-3xl font-bold text-primary mt-2">Loading</p></div><div class="card bg-base-200 p-5"><p class="text-sm text-base-content/60">Validation</p><p id="validation" class="text-xl font-semibold mt-2">Pending</p></div><div class="card bg-base-200 p-5"><p class="text-sm text-base-content/60">Metered workspace cost increase / $50 cap</p><p id="spend" class="text-3xl font-bold mt-2">Pending</p><p class="text-xs text-base-content/60 mt-2">Billing can lag and include other workspace activity.</p></div></section>
<section class="card bg-base-200 p-5 mb-6"><h2 class="font-bold">Running now</h2><p id="current" class="mt-2">Connecting to log…</p><p id="updated" class="text-xs text-base-content/50 mt-3"></p></section>
<section class="card bg-base-200 p-5 mb-6"><h2 class="font-bold">Data leakage & judging audit</h2><p class="text-success mt-2">No test-set leakage found in current submission source.</p><ul class="list-disc pl-5 text-sm space-y-2 mt-3"><li>Training and proxy selection use only harness-supplied training tensors. Normalization and whitening use training images, inside the timer.</li><li>Build sees random synthetic tensors only. Parameters, BatchNorm statistics, gradients, optimizer, EMA and proxy masks reset for each trial.</li><li>Evaluation uses one image view and frozen training statistics; no fitting, test-batch statistics, test lookup, or state updates. CPU reset/evaluation checks and completed GPU harness runs pass.</li><li>The rules explicitly allow comparing reported test accuracy during development. It never controls stopping or selection within a judged trial.</li><li>Official acceptance remains pending: 40 organizer seeds, A100 80GB PCIe, pinned container, CPU quota/network isolation, source review and timing limits. SXM results are development measurements.</li></ul><a class="link text-sm mt-3" href="https://github.com/AIDDA-Institute/CIFAR-100-speedrun/blob/main/RULES.md" target="_blank" rel="noopener">Read the challenge rules</a></section>
<div class="flex flex-wrap gap-3 items-center justify-between mb-3"><h2 class="text-xl font-bold">Experiment ledger</h2><label class="text-sm flex gap-2 items-center"><input type="checkbox" id="qualifying" class="checkbox checkbox-sm">Qualifying only</label></div>
<div class="rounded-box bg-base-200 p-2 overflow-x-auto"><table style="min-width:1000px" class="table table-sm table-zebra"><thead><tr><th style="width:17%">Experiment</th><th style="width:27%">Main change tested</th><th style="width:5%">Trials</th><th style="width:9%">Accuracy</th><th style="width:9%">Time ↑</th><th style="width:14%">Effect vs control</th><th style="width:19%">Evidence</th></tr></thead><tbody id="rows"></tbody></table></div>
<section class="card bg-base-200 p-5 mt-6"><h2 class="font-bold">Suggest the next experiment</h2><p class="text-sm text-base-content/60 mt-2">Annotate a row, or queue a specific idea below. The $50 experiment cap still applies.</p><form class="mt-3" data-lavish-question="next-experiment" onsubmit="event.preventDefault();const idea=document.getElementById('idea').value.trim();if(!idea)return;if(window.lavish){window.lavish.queuePrompt('For the CIFAR-100 runtime experiments, try: '+idea,{queueKey:'next-experiment',tag:'experiment',text:idea,element:event.currentTarget});document.getElementById('queued').textContent='Queued. Click Send to Agent in Lavish to send it.'}else{document.getElementById('queued').textContent='Send this idea in chat or annotate the matching row.'}"><label class="block text-sm" for="idea">Optimization idea</label><textarea id="idea" class="textarea w-full mt-2" placeholder="For example: try more epochs on the narrow 128/320/576 model"></textarea><button class="btn btn-primary btn-sm mt-3" type="submit">Queue experiment</button><p id="queued" class="text-sm mt-2" aria-live="polite"></p></form></section>
<section class="mt-7"><h2 class="text-xl font-bold mb-3">Findings & interruptions</h2><ul id="notes" class="list-disc pl-5 space-y-2"></ul></section>
<details class="mt-7"><summary class="cursor-pointer">Exact settings and result IDs</summary><pre id="raw" class="bg-base-200 rounded-box p-4 mt-3 text-xs"></pre></details>
<p class="text-xs text-base-content/50 mt-8">Compare each recipe with its same-allocation control. Single-seed screens are candidates; successful 40-seed runs establish validation. Auto-refreshes every 5 seconds.</p>
</main><script>
let data; const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function render(){
 if(!data)return;
 const valid=data.rows.filter(r=>r.qualifies&&r.seconds!==null).sort((a,b)=>a.seconds-b.seconds);
 document.getElementById('best').textContent=valid.length?valid[0].seconds.toFixed(3)+' s · '+valid[0].accuracy+'% · n='+valid[0].trials:'None yet';
 document.getElementById('validation').textContent=data.rows.some(r=>!r.is_control&&r.status==='40-seed validated')?'40-seed recipe passed':'No new 40-seed validation';
 document.getElementById('spend').textContent=data.metered_spend_delta===null?'Pending':'$'+data.metered_spend_delta.toFixed(2);
 document.getElementById('current').textContent=data.status.current||'Starting';
 document.getElementById('updated').textContent='Updated '+data.updated;
 const filtered=data.rows.filter(r=>!document.getElementById('qualifying').checked||r.qualifies).sort((a,b)=>(a.seconds??Infinity)-(b.seconds??Infinity));
 document.getElementById('rows').innerHTML=filtered.map(r=>'<tr id="run-'+esc(r.run_id)+'"><td>'+esc(r.name)+'<span class="block text-xs text-base-content/50">'+esc(r.stage||'')+'</span></td><td>'+esc(r.main_change||'Control')+'</td><td>'+r.trials+'</td><td>'+esc(r.accuracy===null?'—':r.accuracy+'%')+'</td><td>'+esc(r.seconds===null?'—':r.seconds+' s')+'</td><td>'+esc(r.effect||'Pending')+'</td><td class="'+(r.qualifies?'text-success':'text-warning')+'">'+esc(r.status)+'</td></tr>').join('');
 document.getElementById('notes').innerHTML=(data.status.notes||[]).map(n=>'<li>'+esc(n)+'</li>').join('');
 document.getElementById('raw').textContent=JSON.stringify(data.rows,null,2)
}
window.speedrunMetricsLoaded=function(payload){data=payload;render()};
function update(){const script=document.createElement('script');script.src='metrics.js?t='+Date.now();script.onload=()=>script.remove();script.onerror=()=>{document.getElementById('current').textContent='Waiting for the live metrics script; retrying…';script.remove()};document.head.appendChild(script)}document.getElementById('qualifying').onchange=render;update();setInterval(update,5000);
</script></body></html>""")


if __name__ == "__main__":
    last_billing = 0
    while True:
        if time.monotonic() - last_billing > 60:
            try:
                r = subprocess.run(
                    ["uv", "run", "modal", "billing", "summary", "--json"],
                    cwd=ROOT,
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=20,
                )
                if r.returncode == 0:
                    json.loads(r.stdout)
                    (OUT / "billing-current.json").write_text(r.stdout)
            except (subprocess.TimeoutExpired, json.JSONDecodeError):
                pass
            last_billing = time.monotonic()
        refresh()
        time.sleep(5)
