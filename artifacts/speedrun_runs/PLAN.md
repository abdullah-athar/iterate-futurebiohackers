# Speedrun plan and state

Written 4 Oct 2026, 02:20 London, before a context compaction; updated 02:55 London (PRs #25 / #26 open). Source of truth for the pending
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
- SAFE track: mean >= 75.15% on both seed sets. Record = PR #25 (PR #23 recipe + max + mean pooling
  written as F.max_pool2d / F.avg_pool2d, lowered by torch.compile to pointwise kernels): 75.24% / 75.31%,
  4.33 / 4.23 s paired (-0.13 s vs PR #23 on both seed sets), 4-CPU build 263 s.
- AGGRESSIVE track: mean >= 75.10% on both. Record = PR #26 (PR #24 recipe = g2 pair 192 + that pooling):
  75.16% / 75.19%, 4.23 / 4.15 s paired (-0.26 / -0.27 s vs PR #23, -0.13 s vs PR #24), 4-CPU build 289 s;
  SAFE floor met by only +0.01 / +0.04 (not sold as safe). Previous: PR #23 (SAFE), PR #24 (AGGRESSIVE).
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
  #22 candidate B (24/28 px, 9.5 ep), #23 (ex SAFE record), #24 (ex AGGRESSIVE record), #25 SAFE record,
  #26 AGGRESSIVE record. Drafts (gitignored):
  `artifacts/pr_drafts/`.
- Working recipe (`crossary`) defaults = PR #23 recipe (control `{}` is the record); every extra
  switch default-off; `scripts/check_variants.py` compares against `origin/speedrun-g3-512-10ep`
  (211/211 at c35bb25). Judging note for every result: "A100 SXM 400 W"; official judging is on an
  A100 80GB PCIe (times ~8% higher).

## Exploration round on the AGGRESSIVE record (user's directions a-f, started 02:35 London)
Control for every screen: `{"g2_pair": "inner192", "global_pool": "fullpool_avgsum"}` (= PR #26, 10 ep),
paired n=8, dtime_adj at k 1.05. The working recipe's DEFAULTS are still PR #23's (check_variants
reference origin/speedrun-g3-512-10ep); moving them to PR #26's recipe (and the reference to
origin/speedrun-g2-192-kernel-pool-10ep, MEASURED_4CPU_BUILDS + 263/289) is pending until the Muon
diff is merged, to avoid conflicts.
- (a) kernel audit: profile of the record (artifacts/speedrun_runs/*_profile-aggr-record): GPU busy
  99.1%, convs ~75%, Inductor BN+GELU pointwise ~20%, fused SGD 0.09 ms/step, per-step batch gather
  0.08 ms/step. Swap under test: `batch_order: "contiguous"` (Round 39, with and without the longer
  24 px phase). No other compiled-vs-ATen swap candidate found (BN/GELU already fused; the pooling
  speedup is pointwise-vs-reduction lowering inside Inductor).
- (b) group 1 pair 48 / 40 (Round 37): no gain (closed).
- (c) widths 896 / 288 (Round 35): all slower (closed).
- (d) resolution / epochs (Round 36): epochs on the line; 24 px phase to 35% +0.15 +- 0.09 pp / -0.12 s at
  10 ep (adj -0.27) but +0.03 / -0.01 s at 10.25 ep -> candidate for n=16 (stacked in Round 39); 28 px to
  60% neutral; 20 px start -0.05 / -0.12 s with two extra graphs (adj -0.08), 20 instead of 24 neutral.
- (e) coordinate descent on the 32 px graphs only (Round 38): +0.00 s (no gain); cdt-all reference and
  the cdt32 4-CPU build pending.
- (f) Muon acceleration: an agent is implementing `muon_impl: "batched"` (batched Newton-Schulz per
  weight shape + torch.compile) in an isolated worktree with a CPU equivalence check; its diff lands in
  the scratchpad (muon_batched.diff) -> apply, smoke, check_variants, screen lr 0.24 / ns 3 / 7.5-8.5 ep
  vs the record (and muon_groups [2]); 4-CPU build if it wins.
- Next finalist candidates: 24 px phase to 35% (+ contiguous batches) -> n=16 paired, then n=40 both
  seed sets, 4-CPU build, record rule.

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
