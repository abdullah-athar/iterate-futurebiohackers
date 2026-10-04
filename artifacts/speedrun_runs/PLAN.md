# Speedrun plan and state (handover)

Written 4 Oct 2026, 03:00 London, for a fresh session. Source of truth for the pending plan; numbers
in `LOG.md` (narrative + one row per GPU run), `registry.jsonl`, `leaderboard.md`. The machine clock is
London time (Git Bash `date` prints "GMT" but shows local time); read `date` for every LOG timestamp.

## Deadlines and budget
- Freeze: 4 Oct 12:00 London. Upstream deadline 4 Oct 14:45 (fork AIDDA-Institute/CIFAR-100-speedrun,
  add only `submissions/futurebiohackers/`; the upstream PR is opened only on the user's explicit go;
  before it: one n=8 PCIe run of the chosen recipe, and if two finalists are within 0.05 s, time both
  on PCIe).
- GPU ledger: 1831 / 3000 min (`gpu_ledger.jsonl`; hard stop via `MODAL_GPU_BUDGET_MIN=3000`).
  Job gate 20 GPU-min per job (ask before `--allow-big`). Up to 10 containers in parallel; stagger
  `modal run` app creates 30-60 s apart (three at once hit "App create rate limit exceeded").
  Progress report to the user every 150 ledger-minutes without stopping.
- Competitor: another team reports 3.9 s at 75.15% (n=40, SXM 400 W). Our AGGRESSIVE record is 4.19 s
  (paired-run time; PR #26), so we are about 0.3 s behind.

## Records, tracks, floors
All records: SXM 400 W, official 75% target, n=40 on seeds 0-39 AND 40-79, zero non-finite losses,
all trials complete, real 4-CPU cold build < 400 s. Times below are paired runs (both recipes back to
back in one container per seed set); standalone cold-build times run 0.05-0.1 s higher on slow cards.
- SAFE track (mean >= 75.15% on both seed sets): **PR #25** = PR #23 recipe + max + mean global pooling
  written as `F.max_pool2d` / `F.avg_pool2d` over the flattened map ([N, C, HW, 1], kernel (HW, 1)),
  which torch.compile lowers to pointwise kernels instead of its reduction kernels for
  `max(dim).values` / `mean` (identical results, -0.13 s per trial): 75.24% / 75.31% (sd 0.25 / 0.26),
  4.33 / 4.23 s, vs PR #23 in the same containers 75.20% / 4.47 s and 75.22% / 4.36 s (+0.03 +- 0.06
  and +0.09 +- 0.06 pp); 4-CPU cold build 263 s. Margins +0.09 / +0.16 pp.
- AGGRESSIVE track (mean >= 75.10% on both): **PR #26** = PR #24 recipe (group 2 residual pair through
  192 channels) + the same pooling: 75.16% / 75.19% (sd 0.23 / 0.28), 4.23 / 4.15 s, vs PR #23
  75.20% / 4.50 s and 75.22% / 4.42 s (-0.05 +- 0.06 and -0.02 +- 0.06 pp; -0.26 / -0.27 s; -0.13 s vs
  PR #24); 4-CPU cold build 289 s. Margins +0.06 / +0.09 pp over 75.10; the SAFE floor is met by only
  +0.01 / +0.04 and is NOT claimed.
- Earlier records: PR #23 (SAFE+AGGRESSIVE before #25/#26: 75.19 / 75.25, 4.56 / 4.49 s standalone,
  build 268 s, PCIe n=8 4.88 s), PR #24 (AGGRESSIVE: 75.13 / 75.20, 4.34 / 4.33 s, build 280 s).
- Cushions at n=16: ~75.20 SAFE, ~75.15 AGGRESSIVE. Exchange rate k ~= 1.05 pp/s (linear, 8.5-10.5 ep).

## Open PRs (all target main directly, all unmerged, main at c4438c4; never merge)
- #21 safety 9.5 ep (75.28 / 75.32, 4.96 / 4.94 s, build 209 s)
- #22 candidate B (24/28 px schedule, 9.5 ep: 75.18 / 75.25, 4.51 / 4.55 s, build 371 s)
- #23 g3 pair 512 + 10 ep (ex record)
- #24 + g2 pair 192 + max+mean pooling via compiled reductions (ex AGGRESSIVE record)
- #25 SAFE record (kernel pooling on #23), branch speedrun-kernel-pool-10ep, worktree C:/dev/iterate-fb-kpool
- #26 AGGRESSIVE record (kernel pooling on #24), branch speedrun-g2-192-kernel-pool-10ep, worktree
  C:/dev/iterate-fb-kstack
PR bodies: `artifacts/pr_drafts/` (gitignored). Each PR body starts with "Independent of the other PRs in
GitHub, but its diff also contains the changes of #22/#23(/#24) (not merged yet) ..."; the diff vs main
of each touches only `cifar100-speedrun/submissions/futurebiohackers/`. All six were reviewed by the
3-lens adversarial workflow (wiring / rules / docs + refutation) before opening.

## Rules (unchanged)
- RECORD RULE: a recipe that sets a new record on either track -> stop, tell the user at once with the
  PR draft ready (title "Speed up CIFAR-100 with <change> (<track>, >= floor%): -X s vs PR #N at equal
  accuracy, A% / B% in T s on A100 SXM 400 W"; body: independence line, track + margins on both seed
  sets, paired n=40 numbers, failure-risk estimate vs 75%, 4-CPU build, honest notes, test plan), ask
  "Open this PR?" and wait. One recipe setting both -> one PR stating both. Between records keep
  working. Never open/push/update a PR without a yes in chat; never merge; branch changes ONLY the
  recipe folder (verify `git diff --stat`); no personal names (PR numbers instead); run the review
  workflow first and relay any blocker before opening.
- Finalist protocol: n=8 paired screen -> n=16 paired -> n=40 paired on both seed sets (one container
  per seed set) -> real 4-CPU cold build (`::main --cpus 4`, must be < 400 s; measure again whenever
  the graphs change) -> record rule. Report non-finite losses for every run; drop a variant with any.
- Competition rules: no TTA, no test-set use, full reset in prepare, no real data / seed in build,
  pinned torch 2.4.0, no extra packages, MIT attribution; never edit `cifar100-speedrun/benchmark/`,
  `tests/`, pyproject, uv.lock. Judging note on every result: "A100 SXM 400 W"; official judging is on an
  A100 80GB PCIe (times ~8% higher).
- Git: work on `crossary` only (never main), `git diff --stat` before each push, no force-push, never
  commit datasets / results / source copies / credentials; commits prefixed `speedrun:`.

## Exploration round on the AGGRESSIVE record (user's directions a-f), control = PR #26 recipe
All screens paired n=8 vs `{"g2_pair": "inner192", "global_pool": "fullpool_avgsum"}` (10 ep), dtime_adj at
k 1.05; details and every number in LOG.md (Rounds 35-40).
- (a) kernel audit: compiled profile of the record (artifacts/speedrun_runs/*_profile-aggr-record):
  GPU busy 99.1%, 178 kernels/step, convs ~75% (cuDNN/xmma), Inductor BN+GELU pointwise ~20% (already
  fused), fused SGD 0.09 ms/step, per-step batch gather (index kernel) 0.08 ms/step, no host syncs.
  Swap tested: `batch_order: "contiguous"` (shuffle folded into the per-epoch crop gather): -0.03 s,
  within noise. No other compiled-vs-ATen candidate: the pooling gain was pointwise-vs-reduction
  lowering inside Inductor. Closed.
- (b) group 1 pair 48 / 40 (`g1_pair`, Round 37): -0.04 to -0.17 s for -0.14 to -0.34 pp. Closed.
- (c) widths [64, 256, 896] / [64, 288, 768] (Round 35): all slower (+0.16 to +0.70 s). Closed.
- (d) resolution / epochs (Round 36): epochs on the line (9.75: -0.06 s / +0.02 pp; 10.25: +0.12 s /
  +0.12 pp); 28 px phase to 60% neutral; 20 px start -0.12 s / -0.05 pp but two extra graphs; 20 px
  instead of 24 neutral. LEAD: 24 px phase to 35% (`resolution_schedule [[24, 0.35], [28, 0.5]]`):
  +0.15 +- 0.09 pp / -0.12 s (Round 36) and, with contiguous batches, -0.10 +- 0.14 / -0.14 s (Round 39).
  Round 40 (n=16 paired, seeds 0-15, 03:00 London): 75.14% / 4.04 s vs PR #26 recipe 75.16% / 4.17 s =
  -0.01 +- 0.06 pp, -0.13 s (adj -0.12) -> CONFIRMED at n=16; with contiguous batches -0.06 +- 0.05 /
  -0.15 s (not worth the extra switch). 0 non-finite. NOT yet run at n=40 (wrap-up: no new launches).
- (e) coordinate descent (`inductor_tuning_resolutions`, Round 38): 32 px graphs only +0.00 s with a
  492 s 4-CPU build (fails the clause); all graphs -0.01 s. Closed.
- (f) Muon acceleration: MUON_PLACEHOLDER

## Pending items, in order
1. FIRST: the 24 px phase to 35% passed n=16 (-0.13 s at equal accuracy). Run paired n=40 on seeds 0-39 and 40-79 vs the PR #26 recipe (one
   container per seed set, jobs like `jobs/r34-paired40-s*.json` with control = PR #26 params and
   variant `{"g2_pair": "inner192", "global_pool": "fullpool_avgsum", "resolution_schedule": [[24, 0.35],
   [28, 0.5]]}`; ~9 GPU-min each), plus the real 4-CPU build of that config (same 6 graphs; measure
   anyway), then the record rule. Expected: AGGRESSIVE record at ~4.10 / 4.02 s with ~75.15 / 75.18%
   (floor +0.05 / +0.08). For the SAFE track the same schedule on the pooling-only recipe (PR #25) is
   unscreened: screen it n=8 -> n=16 first.
2. Muon: see (f). Screen plan once the batched/compiled step is verified: `{"optimizer": "muon_airbench",
   "muon_impl": "batched", "muon_lr": 0.24, "muon_momentum": 0.6, "muon_ns_steps": 3}` at 7.5 / 8.0 / 8.5
   ep on the PR #26 recipe, and `muon_groups: [2]` at 8.5 / 9.0 ep; a 4-CPU build if it wins (new graph).
   Evidence so far: +0.91 +- 0.11 pp at 10 ep with +1.07-1.57 s of host overhead (loop implementation);
   break-even at every epoch count 7-10; ns_steps 2 kills the gain; group 3 only keeps +0.57 pp at
   +0.57 s.
3. Move the working recipe's DEFAULTS to the PR #26 recipe (`g2_pair: "inner192"`, `global_pool:
   "fullpool_avgsum"`), set `scripts/check_variants.py` REFERENCE_REF to
   `origin/speedrun-g2-192-kernel-pool-10ep`, add 263 / 289 s to the launcher's MEASURED_4CPU_BUILDS
   (6 graphs); then control `{}` is the record again. Deferred so far to avoid conflicts with the Muon
   diff.
4. Ideas needing the user's go: architecture options (g3 depth 4, narrower whitening, fused BN+GELU
   kernel), late BatchNorm freeze, grouped convs. Everything optimizer/augmentation is flat.
5. Final submission: the user's call between #25 (safe) and #26 (aggressive); PCIe n=8 timing first.

## Working recipe and tooling state
- `cifar100-speedrun/submissions/futurebiohackers/submission.py` on crossary: DEFAULTS = PR #23 recipe;
  default-off switches: count_nonfinite, aug_off_last, skip_residual_*, label_smoothing_end, g3_pair,
  g2_pair, g1_pair, se_groups/se_ratio, multiscale_head, global_pool (maxmean_cat / maxmean_sum /
  fullpool / fullpool_sum / fullpool_avgsum), optimizer muon_airbench (+ muon_ns_steps, muon_groups),
  inductor_tuning_resolutions, batch_order. `scripts/check_variants.py`: 227/227 (default path
  bit-identical to origin/speedrun-g3-512-10ep, every switch exercised).
- Worktrees: fb-kpool (#25), fb-kstack (#26), fb-g2stack (#24), fb-g3 (#23), fb-24px (#22), fb-safety
  (#21); obsolete: fb-infra, fb-recipe, fb-stack, fb-w64; the Muon agent's worktree under
  `.claude/worktrees/`.
- Launch: `PYTHONUTF8=1 ~/.local/bin/uv.exe run modal run scripts/modal_speedrun.py::screen --jobs
  artifacts/speedrun_runs/jobs/X.json --parallel N --warm`; 4-CPU build: `::main --cpus 4 --n 2 --seed 0
  --params '{...}' --tag cold4cpu-X`; profile: `::profile --compiled --params '{...}' --tag T`; 11
  GPU-guard attempts then RuntimeError -> relaunch.
- Checks: ruff `~/.local/bin/uvx.exe ruff@0.6.9 --line-length 100`; WSL for torch
  (`scripts/wsl_speedrun.sh python -m benchmark.run --submission futurebiohackers --device cpu
  --synthetic --n 2`; worktrees: `cd <wt>/cifar100-speedrun && UV_PROJECT_ENVIRONMENT=$HOME/.venvs/
  cifar100-speedrun uv run --frozen ...`); `scripts/check_reset.py` lives only on crossary (run that copy
  against a worktree folder); check_variants ~10 min.
- GitHub without gh: token from `git credential fill`, PR via the REST API; commits and PR bodies end
  with the Claude attribution lines.
