# Speedrun plan and state

Written 4 Oct 2026, 02:20 London, before a context compaction. Source of truth for the pending
plan; numbers in `LOG.md` (narrative + one row per GPU run), `registry.jsonl`, `leaderboard.md`.
The machine clock is London time (Git Bash `date` prints "GMT" but shows local time).

## Deadlines and budget
- Freeze: 4 Oct 12:00 London. Upstream deadline 4 Oct 14:45 (fork AIDDA-Institute/CIFAR-100-speedrun,
  add only `submissions/futurebiohackers/`; the upstream PR is opened only on the user's explicit go).
- GPU ledger 1660 / 3000 min at 02:18 (`gpu_ledger.jsonl`; hard stop via `MODAL_GPU_BUDGET_MIN=3000`).
  Job gate 20 GPU-min per job (ask before `--allow-big`). Up to 10 containers in parallel; stagger
  `modal run` app creates 30-60 s apart (three at once hit "App create rate limit exceeded").
  Progress report to the user every 150 ledger-minutes (last full report at ~1653) without stopping.

## Records, tracks, floors (SXM 400 W, official 75% target, n=40 on seeds 0-39 AND 40-79, zero
## non-finite losses, all trials complete, real 4-CPU cold build < 400 s)
- SAFE track: mean >= 75.15% on both seed sets. Record = PR #23 (candidate B + g3 pair 512 + 10 ep):
  75.19% / 75.25%, 4.56 / 4.49 s standalone (4.46 s in the Round 31/34 containers), 4-CPU build 268 s.
- AGGRESSIVE track: mean >= 75.10% on both. Record = PR #24 (PR #23 + g2 pair 192 + maxmean_sum,
  10 ep): 75.13% / 75.20%, 4.34 / 4.33 s paired (-0.12 / -0.14 s vs #23), 4-CPU build 280 s.
- Cushions at n=16: ~75.20 SAFE, ~75.15 AGGRESSIVE. Exchange rate k ~= 1.05 pp/s (linear, 8.5-10.5 ep).
- RECORD RULE: a recipe that sets a new record on either track -> stop, tell the user at once with
  the PR draft ready (title "Speed up CIFAR-100 with <change> (<track>, >= floor%): -X s vs PR #N at
  equal accuracy, A% / B% in T s on A100 SXM 400 W"; body: independence line, track + margins on
  both seed sets, paired n=40 numbers, failure-risk estimate vs 75%, 4-CPU build, honest notes, test
  plan), ask "Open this PR?" and wait. One recipe setting both -> one PR stating both. Between
  records keep working. Never open/push/update a PR without a yes in chat; never merge; branch
  changes ONLY `cifar100-speedrun/submissions/futurebiohackers/` (verify `git diff --stat`); target
  main directly with the line "Independent of the other PRs in GitHub, but its diff also contains
  the changes of #22 and #23 (not merged yet) ..."; no personal names (refer to PR numbers); run the
  3-lens adversarial review workflow first and relay findings.
- Open PRs, all targeting main, unmerged (main at c4438c4): #21 safety 9.5 ep (75.28 / 75.32, 4.96 s),
  #22 candidate B (24/28 px, 9.5 ep), #23 SAFE record, #24 AGGRESSIVE record. Drafts (gitignored):
  `artifacts/pr_drafts/`.
- Working recipe (`crossary`) defaults = PR #23 recipe (control `{}` is the record); every extra
  switch default-off; `scripts/check_variants.py` compares against `origin/speedrun-g3-512-10ep`
  (211/211 at c35bb25). Judging note for every result: "A100 SXM 400 W"; official judging is on an
  A100 80GB PCIe (times ~8% higher).

## Round 34 RESULTS (02:21 London) and NEXT STEPS (first thing after compaction)
Both finalists passed (details in LOG.md "Round 34"): SAFE candidate `{"global_pool": "fullpool_avgsum"}`
75.24% / 75.31% at -0.13 / -0.13 s vs PR #23 (+0.03 / +0.09 pp), 4-CPU build 263 s; AGGRESSIVE
candidate `{"g2_pair": "inner192", "global_pool": "fullpool_avgsum", "epochs": 10.0}` 75.16% / 75.19% at
-0.26 / -0.27 s (-0.05 / -0.02 pp), 4-CPU build 289 s (also meets the SAFE floor by only +0.01 / +0.04).
0 non-finite in all 320 trials. Both are records (SAFE and AGGRESSIVE). The record rule applies: the
user was told at 02:25 London and asked "Open these PRs?"; NOTHING may be pushed or opened without a
yes in chat.
Next steps, in order:
1. Prepare PR branch A (SAFE): worktree off `origin/speedrun-g3-512-10ep` (PR #23), e.g.
   `C:/dev/iterate-fb-kpool`, branch `speedrun-kernel-pool-10ep`: in the Net forward replace the plain
   max pooling by `x4 = x.flatten(2).unsqueeze(-1)` (view [N, C, HW, 1]), `kernel = (HW, 1)`,
   `(F.max_pool2d(x4, kernel) + F.avg_pool2d(x4, kernel)).flatten(1)`; keep the structure of PR #23's
   file (a `global_pool` option "fullpool_avgsum" as the default, validation list extended); README:
   model paragraph (max + mean head input), implementation bullet (pooling kernels instead of
   Inductor's max(dim) reduction: -0.13 s), results table with the paired rows, build 263 s,
   experiments table rows (fullpool -0.14 s / 0 pp; fullpool_sum -0.07 s; maxmean_sum via the
   compiled reduction 0 s). Title above; body per the PR #24 template (independence line, track and
   margins +0.09 / +0.16, paired numbers, failure risk well under 0.1% (mean 75.27% over 80 trials),
   build, test plan).
2. Prepare PR branch B (AGGRESSIVE): worktree off PR #24's branch `speedrun-g2-192-maxmean-10ep`
   (e.g. `C:/dev/iterate-fb-kstack`, branch `speedrun-g2-192-kernel-pool-10ep`): change the pooling
   implementation to the kernel version (default "fullpool_avgsum"), README numbers (75.16 / 75.19,
   4.23 / 4.15 s, -0.26 / -0.27 s, build 289 s; margins +0.06 / +0.09 aggressive, +0.01 / +0.04 safe
   stated honestly as too thin), independence line mentioning #22, #23 and #24.
3. For each branch: CPU smoke, ruff, crossary `scripts/check_reset.py` against the folder, `git diff
   --stat` (folder only), adversarial review workflow (wiring / rules / docs + refutation), relay
   findings, then open ONLY on the user's yes (API as for PR #24), update LOG/PLAN/memory.
4. Working recipe (crossary) defaults stay = PR #23 until the user says otherwise (check_variants
   reference `origin/speedrun-g3-512-10ep`). If the user wants the baseline moved to the new records,
   change DEFAULTS, REFERENCE_REF and the launcher's MEASURED_4CPU_BUILDS/graph defaults together.
5. Then: Muon CUDA-graph track if approved; otherwise remaining ideas need the user's go.

## Round 34 setup (for reference)
Jobs `jobs/r34-paired40-s0.json` and `-s40.json`, paired n=40 vs the record in one container per
seed set (logs in the session scratchpad: r34-s0.log, r34-s40.log, cold4cpu-g2-avgsum-b.log; the
launcher appends rows to LOG.md/registry.jsonl and prints "paired vs control (n=40)" lines).
1. SAFE finalist `{"global_pool": "fullpool_avgsum"}` = the record with max + mean pooling through
   ATen kernels (`F.max_pool2d` + `F.avg_pool2d` over the flattened map viewed [N, C, HW, 1]).
   Round 32 n=16: +0.11 +- 0.09 pp, -0.12 s. Real 4-CPU cold build 263 s (done, passes).
   Decision: both seed sets >= 75.15 and faster -> SAFE record -> PR proposal, title "Speed up
   CIFAR-100 with max+mean pooling through the pooling kernels (safe, >= 75.15%): -0.12 s vs PR #23
   at equal accuracy, A% / B% in T s on A100 SXM 400 W"; note it ties PR #24's time with more
   accuracy. If only >= 75.10 both: it does not beat PR #24 on time, no PR.
2. AGGRESSIVE finalist `{"g2_pair": "inner192", "global_pool": "fullpool_avgsum", "epochs": 10.0}`.
   Round 32 n=16: -0.06 +- 0.11 pp, -0.24 s. 4-CPU build relaunched (first launch aborted after 11
   GPU-guard misses); must be < 400 s. Decision: both >= 75.10 and faster than PR #24 -> AGGRESSIVE
   record -> PR "Speed up CIFAR-100 with a 192-channel residual pair in group 2 and kernel-based
   max+mean pooling (aggressive, >= 75.10%): -0.24 s vs PR #23 ...". If also >= 75.15 both -> one PR
   for both tracks.
3. PR mechanics for either: worktree off `origin/speedrun-g3-512-10ep` (for the stack one, cut from
   PR #24's branch `speedrun-g2-192-maxmean-10ep` and change only the pooling implementation), README
   update (model paragraph, results table with the paired rows, "why", build time, experiments
   table), CPU smoke + ruff + the crossary copy of `scripts/check_reset.py` against the worktree folder,
   adversarial review workflow (wiring / rules / docs lenses + refutation), then STOP and ask.
4. Other Round 32 reads (n=16 vs record): fullpool (max only) -0.03 +- 0.08 / -0.14 s; fullpool_sum
   (max_pool2d + compiled mean) +0.01 / -0.07 s; g2 192 + fullpool at 10 ep -0.14 +- 0.08 / -0.29 s
   (would miss the aggressive floor), at 9.75 ep -0.26 / -0.36 s; g2 192 + fullpool_sum 10 ep
   -0.07 / -0.23 s, 9.75 ep -0.18 / -0.31 s.

## Muon CUDA-graph track (NOT started; needs the user's go)
- Devin's airbench Muon (`optimizer: "muon_airbench"`, `muon_lr` 0.24, `muon_momentum` 0.6,
  `muon_ns_steps` 3, `muon_groups` [0, 1, 2]) gives +0.91 +- 0.11 pp at 10 ep but +1.07 to +1.57 s per
  trial of host-side overhead (per-parameter Python loop, ~300 kernel launches per step, container-CPU
  dependent). Round 33 (user's request): break-even at every epoch count 7-10 (dtime_adj at k 1.05:
  7.0 ep -0.04 s, 7.5 +0.12, 8.0 -0.03, 8.5 +0.17, 10 +0.70); ns_steps 2 destroys the gain (+0.08 pp);
  Muon on group 3 only (`muon_groups: [2]`) keeps +0.57 +- 0.12 pp at +0.57 s (adj +0.03).
- Plan if approved: capture the Muon update as one CUDA graph (manual `torch.cuda.CUDAGraph` with
  static grad copies via `torch._foreach_copy_` and the lr as a device scalar, or
  `torch.compile(mode="reduce-overhead")` on a functional update with params/buffers marked static);
  persist the graph and momentum buffers across trials (zero them in `prepare`; confirm the net is
  built once and re-initialised in place); warm/capture in `build` on synthetic steps. GPU time
  estimate 0.4-0.9 ms/step -> 0.2-0.45 s per trial -> net ~-0.4 s at equal accuracy if the +0.9 pp
  holds near 9 ep. Needs a real 4-CPU build (new graph), check_variants/check_reset coverage, screen as
  a default-off `muon_impl: "graph"` switch, group 3 only as a cheaper variant.

## Verdicts so far (numbers in LOG.md)
- On the record recipe everything optimizer/augmentation is flat (Rounds 16, 19, 26); resolution and
  epochs trade at ~1.05 pp/s. Negative or on-the-line: stems, depth 2, bottleneck pair (-1.4 pp),
  g3 pairs 448/384, SE (break-even), multi-scale head (-0.3), Devin's wider group 1, BN recal, no-aug
  finish, LS schedule, progressive depth, inductor flags (coordinate descent -0.09 s but 4-CPU build
  523 s: fails the clause), the max+mean pooling's accuracy gain (+0.30 at n=8 not confirmed at n=16/40),
  Muon with the loop implementation.
- Free gains found: g2 pair 192 (-0.13 s, ~-0.05 pp); ATen pooling kernels instead of Inductor's
  max(dim) reduction (-0.13 s, 0 pp).
- Need the user's go: architecture options 2-5 (width rebalance, g3 depth 4, narrower whitening,
  fused BN+GELU kernel), late BatchNorm freeze, grouped convs, the Muon graph track.

## Operational notes
- Worktrees: `C:/dev/iterate-fb-g2stack` (PR #24 branch), `iterate-fb-g3` (PR #23), `iterate-fb-24px`
  (PR #22), `iterate-fb-safety` (PR #21); obsolete: fb-infra, fb-recipe, fb-stack, fb-w64.
- Launch: `PYTHONUTF8=1 ~/.local/bin/uv.exe run modal run scripts/modal_speedrun.py::screen --jobs
  artifacts/speedrun_runs/jobs/X.json --parallel N --warm`; 4-CPU build: `::main --cpus 4 --n 2 --seed 0
  --params '{...}' --tag cold4cpu-X`; 11 GPU-guard attempts, then RuntimeError -> relaunch.
- Checks: ruff `~/.local/bin/uvx.exe ruff@0.6.9 --line-length 100`; WSL for anything with torch
  (`scripts/wsl_speedrun.sh python -m benchmark.run --submission futurebiohackers --device cpu
  --synthetic --n 2`; worktrees: `cd <wt>/cifar100-speedrun && UV_PROJECT_ENVIRONMENT=$HOME/.venvs/
  cifar100-speedrun uv run --frozen ...`); `scripts/check_reset.py` lives only on crossary, run that
  copy against a worktree folder; check_variants ~10 min.
- GitHub without gh: token from `git credential fill`, PR via the REST API; commits and PR bodies end
  with the Claude attribution lines; the user's PR title/body settings for #24 are the template.
- Before the final upstream submission: one n=8 PCIe run of the chosen recipe (PR #23 recipe: 4.88 s);
  two finalists within 0.05 s -> time both on PCIe. The final-submission recommendation goes into the
  summary with records, PR drafts, what worked and what did not.
