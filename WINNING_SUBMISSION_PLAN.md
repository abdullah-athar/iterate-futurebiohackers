# Plan for a competitive FutureBioHackers submission

**Decision:** make the autoresearch entry convincing through trustworthy evaluation, a useful discovered algorithm, and measured research efficiency. Pursue CIFAR-100 alongside it by combining the strongest existing recipe and implementation work. Reserve Sunday afternoon for verification and submission.

This is a plan, not an implementation or a claim that we have won. Reviewed on **3 October 2026**, against latest fetched `main` at [`8d69bd8`](https://github.com/abdullah-athar/iterate-futurebiohackers/tree/8d69bd8e58512e379a7bbfa0b3d85f4ca7a5e2d6) and all six open PRs: #5, #10, #12, #14, #16 and #17. #15 merged during this review and is included in this main snapshot. GPU figures below are author-reported development results; this review did not rerun GPU training. Two evaluator weaknesses were reproduced locally through the existing CLI, including on the updated main.

## 1. What matters most

| Priority | Action | Why it earns its time | Completion gate |
| --- | --- | --- | --- |
| P0 | Fix the autoresearch evaluator boundary before new discovery runs | Candidates can currently read planted answers and change the data they are scored against | Answers/seeds absent from candidate input; independent scoring of original inputs; regression probes fail safely |
| P0 | Version the harder benchmark now merged through #15 and seal final evaluation | The old objective has plateaued; current five-instance results cannot establish generalization | Frozen benchmark manifest, equal-budget classical baselines, untouched final suites |
| P0 | Run the full framework against a simple search control | Judges score the research system, including efficiency, not just its best solver | Repeated runs, complete cost accounting, held-out quality and progress curves |
| P0 | Use #16 as the CIFAR candidate to beat; test #14's systems changes and #17's smaller-batch idea | #16 reports the best paired results; the other branches contain plausible incremental gains | Paired screening followed by a frozen 40-trial run; official-hardware verification where available |
| P0 | Assign a submission owner and prepare the video early | A competitive result must reach the correct destination on time | Upstream CIFAR PR and separate autoresearch submission package ready before deadline |
| P1 | Improve edit selection and budget allocation; audit novelty rejection | These target the binding 1-second solver budget and wasted research work | Improvements survive equal-budget comparison and fresh instances |
| P2 | Extra domains, asynchronous orchestration, new optimizers, elaborate kernels | Useful later, but weak returns until the evidence above exists | Attempt only after P0 deliverables are secure |

## 2. Requirements from the guides

Sources read: the supplied **CIFAR-100 Speedrun.pdf**, pp. 1–4, and **Build Your Own Algorithm Autoresearch Framework.pdf**, pp. 1–5, from the team's Downloads folder. They are not copied into this PR. The repository's [competition rules](cifar100-speedrun/RULES.md) give the detailed CIFAR contract; check the [live upstream rules](https://github.com/AIDDA-Institute/CIFAR-100-speedrun/blob/main/RULES.md) again before freezing.

| Track | What wins / what must be delivered |
| --- | --- |
| CIFAR-100 | Lowest mean **preparation + training** time, conditional on mean top-1 accuracy ≥75% and **all 40 trials succeeding**. No per-trial accuracy floor. One A100 **80GB PCIe**, four CPU threads/quota, pinned PyTorch 2.4.0 environment, network disabled. Full inference ≤5 seconds per trial; import/build ≤600 seconds. Submit an upstream PR adding only `submissions/futurebiohackers/`, using working defaults, before **4 October, 14:45**. |
| Autoresearch | A reusable system that autonomously proposes, tests, learns and improves algorithms. Judged on **novelty, performance, interpretability/ease of use, and research efficiency**. Email `admin@algorithmdiscovery.org`: team name, repo URL, short description, and a Google Drive link to a **maximum four-minute video**. Include installation/run instructions, design explanation and benchmark examples. |

Use **14:45 BST on Sunday 4 October** as the operational deadline from the team's existing event documentation. The autoresearch PDF does not independently state a deadline/time zone; the submission owner should verify its cutoff with organizers. Its four-minute video requirement is separate from the two-minute general event demo described in our README. Prepare a ≤4-minute track video and a two-minute cut. The CIFAR guide explicitly encourages doing that challenge alongside another track.

## 3. Current state and PR decisions

### CIFAR: integrate deliberately

| Revision reviewed | Reported result | Recommendation |
| --- | --- | --- |
| `main` `8d69bd8` | 75.29%, 8.11 s, 40 trials, A100 SXM; root README still describes these defaults | Retain as historical baseline; update its headline only after selecting and validating new defaults |
| [#5](https://github.com/abdullah-athar/iterate-futurebiohackers/pull/5), `55d7931` | 75.2495%, 6.859 s, 40 trials; smaller early convolution blocks | Foundational work already carried by later branches; preserve attribution and useful checks, avoid merging overlapping implementations sequentially |
| [#10](https://github.com/abdullah-athar/iterate-futurebiohackers/pull/10), `bcf5a0e` | 75.12575%, 6.3086 s, 40 trials; GELU alternative 75.33425%, 6.4633 s | Preserve pooling correctness tests, profiler evidence and negative results. SiLU's gain does not transfer reliably to #14 |
| [#12](https://github.com/abdullah-athar/iterate-futurebiohackers/pull/12), `2ed7335` | 75.34% ±0.28 pp SD, 5.76 s, seeds 0–39, SXM 400 W; cold build 187 s | Strong accuracy anchor: progressive resizing plus an accuracy-recovery stack |
| [#14](https://github.com/abdullah-athar/iterate-futurebiohackers/pull/14), `5f06aea` | Current fused-SGD defaults: 75.130%, 6.048 s vs paired control 75.268%, 6.319 s, 40 seeds. Earlier unfused version: 75.094%, 5.627 s vs 75.169%, 5.816 s, another 40 seeds | Strong implementation source: fused SGD, indexed crop, static compiled forward/loss, global max reduction, fp16 BatchNorm. Distinguish the two configurations; margin remains thin |
| [#16](https://github.com/abdullah-athar/iterate-futurebiohackers/pull/16), `aa46083` | Two paired 40-trial HF SXM runs: **75.24% / 5.109 s** and **75.32% / 5.126 s**; #12 controls 75.27% / 5.690 s and 75.31% / 5.712 s | **Primary integration base**: already combines #12's resizing/accuracy stack with a shallower first group, tuned schedule, full-map pool and compiled loss. Verify completion and qualification from raw records |
| [#17](https://github.com/abdullah-athar/iterate-futurebiohackers/pull/17), `026b396` | 75.225%, 5.575874 s, 40/40 trials; paired #10 control 75.1825%, 6.285433 s | Preserve batch-512, schedule/whitening retuning and fused-SGD evidence; screen transfer to #16 without assuming its SiLU recipe transfers |

The strongest evidence is #16's approximately **10.2% paired speedup over #12** in two runs. #17 improves its own #10 control by 11.3%; #14 improves its own control by 4.3% with current defaults. These percentages have different denominators and cannot rank the branches directly. None is an official PCIe score. The organizer's 59.30 s/75.36% baseline is a **two-trial PCIe pilot**, not a comparable 40-trial baseline.

Build one integration branch from main using #16's submission as the initial candidate, preserving exact #12/#14/#16/#17 snapshots as controls. Port changes explicitly rather than accepting a large conflict resolution as a new recipe. Keep third-party licenses. After the integrated replacement is reviewed, mark overlapping PRs superseded with links; no merges or closures are part of this planning PR.

### Autoresearch: substantial system, incomplete evidence

Main already has a working proposal swarm, prompt-mode UCB allocation, normalized-code novelty checking, a screen/validate/confirm cascade, per-instance specialists, a ledger, Modal evaluation and a dashboard. Do not spend the remaining hackathon rebuilding these.

Merged [#15](https://github.com/abdullah-athar/iterate-futurebiohackers/pull/15), branch reviewed at `882cc9a`, moves `median_string` to a larger `mid` suite and adds classical baseline reporting. Its local totals are set median **24,202**, template **23,085**, seed **22,816**, naive indel search **20,386**, and planted reference **18,787**. These are preliminary validation results, not matched remote-CPU or final held-out results. The naive search is already about **10.7% better than the seed**; beating only the seed will be an insufficient headline.

Build on #15 with the evaluator and benchmark changes below. Keep `median_string_long` as a transfer/stress regime. The old score **511** is empirically saturated under the searches reported in the PR; absence of a better local move or restart is **not an optimality proof**. A planted string is a feasible reference, not necessarily the optimal median. Remove the dashboard's current “planted optimum” label.

## 4. Autoresearch: first make the results trustworthy

### A. Close the observed evaluator failures

At the reviewed main, [`ProblemInstance`](https://github.com/abdullah-athar/iterate-futurebiohackers/blob/8d69bd8e58512e379a7bbfa0b3d85f4ca7a5e2d6/autoresearch-optimizer/median_string/instance.py) contains `planted_consensus`, `known_best_score` and metadata; the generator puts the seed in metadata. [`Evaluator.evaluate_solver`](https://github.com/abdullah-athar/iterate-futurebiohackers/blob/8d69bd8e58512e379a7bbfa0b3d85f4ca7a5e2d6/autoresearch-optimizer/median_string/evaluator.py) passes this object to candidate code and subsequently scores against it.

Two temporary candidates passed the normal import guard and `just autoresearch … try --split screen`:

| Probe | Observed result on main | Required regression check |
| --- | --- | --- |
| Return `instance.planted_consensus` | Accepted; total 51 versus baseline 58 without solving the problem | Candidate input exposes no planted answer, generator seed or reference score |
| Replace `instance.strings[:]` with its first string, then return that string | Accepted; both score and baseline became **0** | Candidate mutations cannot affect authoritative inputs, constraints, baseline or score |

This establishes evaluator weaknesses, **not evidence that previous agents exploited them**. Audit archived winners for answer access, input mutation and hard-coded cases, then re-evaluate them under the corrected protocol before presenting results.

Implement a minimal boundary:

1. The trusted evaluator retains original immutable strings, constraints and references. Send the candidate only strings, alphabet, metric, any public length constraint and CPU budget. Preserve derived properties needed by existing solvers, such as `num_strings`.
2. Execute the candidate in a subprocess receiving that sanitized input, with a narrow response containing only the candidate string or failure. Compute validity, edit distances and baselines in the trusted parent from its original input. Never accept candidate-generated score JSON as authoritative.
3. Strip secret environment variables from the candidate subprocess; keep hidden-suite generation in the trusted evaluator. Bound runtime and response size. Keep the import guard as a useful check, without presenting it or a subprocess as a complete security sandbox.
4. Add focused regression checks for planted-answer/seed access, mutation, forged output, invalid characters, crashes and budget overruns. Re-run existing solver and loop tests. Tag results with evaluator/source hashes so old and corrected runs cannot silently mix.

### B. Freeze a benchmark that measures useful progress

Keep the 1000 ms CPU budget and document the existing 25% enforcement grace. Use the same environment, distance kernel and limits for every baseline and candidate. Re-evaluate borderline candidates rather than treating a laptop timing as equivalent to a Modal CPU timing.

- Version #15's changed objective, confirm and holdout distributions in a manifest; reject resuming old runs under the new distribution. Include lengths, counts, alphabets, corruption model, CPU allocation and seed-set identifiers.
- Use cheap screening primarily for correctness. Add a representative mid-size screen before rejecting candidates for quality: performance on tiny strings can be a poor predictor of a solver designed for 300–1500 characters.
- Keep confirmation as **search-time validation**: repeated pass/fail feedback makes it adaptive, even if its seeds are hidden. Reserve new final suites for a single frozen comparison. Do not choose a winner after inspecting final-suite results.
- Change #15's initialization to compute public baselines first; postpone holdout baselines and dashboard exposure until final evaluation. Current `write_baselines` evaluates holdout at initialization.
- Freeze, for example, 10 fresh generated suites × 5 instances for final quality, plus an explicitly separate long-string transfer suite. Repeat timing-sensitive finalists on the same CPU class; report invalid/timeout rates and uncertainty across suites. Do not inflate sample size by treating related strings within an instance as independent trials.
- Keep fixed-length Hamming majority consensus as a sanity control. Measure progress primarily on Levenshtein cases with indels. Any biological claims should say **synthetic sequence-consensus benchmark** until demonstrated on appropriate real data.

### C. Compete with credible classical algorithms

Run set median, frequency consensus, the template, the starting solver and #15's budget-aware naive indel search under the corrected evaluator. Add one strong, interpretable comparison: alignment-guided refinement that ranks likely edits and exactly scores the most promising moves. The literature on [estimating the effect of edits in median-string refinement](https://arxiv.org/abs/1912.02217) gives a relevant classical baseline; label an adaptation as such rather than claiming an exact reproduction.

Prioritize these candidate hypotheses, in order:

1. **Alignment-guided edits:** use existing `levenshtein_editops` to propose substitutions, insertions and deletions; spend exact distance calls on promising sites instead of exhausting every edit.
2. **Better starts and restarts:** compare medoid and alignment-consensus starts; reserve a measured fraction of the budget for local improvement and return the best feasible answer before timeout.
3. **Measured instance adaptation:** choose phases using observable length, alphabet and disagreement. Charge feature extraction and all solver branches to the same budget. Avoid instance-name lookup or generator metadata.

Use the existing RapidFuzz distance implementation; pure-Python dynamic programming is unlikely to be a useful replacement at these lengths. Keep the original summed-distance objective and add per-regime/normalized gains for interpretation, so long DNA instances do not hide protein regressions. Plot quality versus CPU budget at 100, 300 and 1000 ms for the final solver and strongest baseline.

## 5. Show that the framework improves research efficiency

The defensible pitch is: **“Our system uses per-instance failures to preserve and combine complementary algorithms, then checks whether the gains survive fresh inputs under a fixed compute budget.”** Treat this as the contribution to demonstrate, not a claim that any one component is unprecedented.

[FunSearch](https://www.nature.com/articles/s41586-023-06924-6) already uses LLM-generated programs and evaluation-driven search. [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) extends that approach to broader algorithmic optimization. [ShinkaEvolve](https://arxiv.org/abs/2509.19349) explicitly combines parent sampling, novelty rejection and bandit-based **model** selection, and reports a circle-packing result with 150 samples. Our bandit selects **prompt modes**, and our archive retains instance winners; their value needs an experiment. None of those papers supplies a directly comparable median-string score for our task.

### Minimum evidence package

| Comparison | Hold constant | What it answers |
| --- | --- | --- |
| Simple incumbent-only loop versus full framework | Model/version, starting solver, benchmark, proposal concurrency, token/cost ceilings and evaluation budget | Does the complete framework improve discovery beyond ordinary generate/evaluate/keep? |
| Full framework versus no per-instance archive/merge | Same conditions; retain other full-system mechanisms | Do complementary specialists contribute to the result? |
| Uniform prompt modes versus UCB allocation, if time remains | Same eligible modes and total budget | Does adaptive allocation help on this task? |

Use **at least three independent search runs per primary arm**, initially 4–8 agents per generation, with a 20-minute cap per run and the same explicit token/dollar and evaluator-call ceilings. These controls require small configuration additions; they are not all existing CLI switches. Expand concurrency only after measuring useful unique candidates per token and per wall-clock minute. A 32-agent swarm beating one agent on elapsed time alone does not establish research efficiency.

Record every `try` call, candidate, rejected proposal, retry, failure, confirmation and baseline evaluation. Currently `cmd_try` explicitly avoids the ledger, so proposal counts omit agents' self-tests. Include their CPU time and calls in totals. Preserve model usage including cached tokens, and label missing usage on killed sessions as unknown. A duplicate rejected after generation saves a subsequent evaluation; it does **not** recover tokens already spent generating it. Correct that dashboard claim.

Audit the current 0.95 normalized-code similarity threshold: a small numerical or control-flow change can matter despite high similarity. Hard-reject exact normalized duplicates; treat near-duplicates as a cheap-screen decision or sample a small audit fraction. Show whether the gate saves evaluation cost without suppressing useful tuning.

Report final held-out gain versus **both seed and strongest classical baseline**, all run outcomes, best score versus tokens/evaluation CPU/wall time, and timeout/duplicate rates. Include one readable discovery trace: hypothesis → changed algorithm → per-instance evidence → confirmation → final evaluation. Prefer a repeatable positive result over a selected best run; if the framework loses, explain the result and retain the simpler configuration.

Demonstrate reuse within sequence optimization through mid and long regimes. The guide permits domain specialization. Move hard-coded solver-contract text in `swarm.py` behind the existing `Problem` adapter before claiming broad plug-and-play reuse. A CIFAR adapter or unrelated second domain is optional after the main evidence package; current CIFAR PR results were not shown to come from this framework.

## 6. CIFAR: the highest-value remaining experiments

### Sources worth borrowing from

| Primary source checked | Reported result | Implication here |
| --- | --- | --- |
| [Airbench](https://github.com/KellerJordan/cifar10-airbench) | CIFAR-10 94.01% in 2.59 s on a 400 W A100 | Existing lineage; its README now points to a successor, so do not call 2.59 s the current record |
| [Hiverge](https://github.com/hiverge/cifar10-speedrun) | CIFAR-10 94.02% in 1.98 s | Useful implementation reference; our #14 experiments already found a substantial accuracy loss from its Muon adaptation |
| [Fulcrum/Fable-5 report](https://github.com/fulcrumresearch/cifar-10-speedrun) | CIFAR-10 94% in 1.828 s versus 1.978 s, 200 trials per recipe; 24→28→32 curriculum | Supports testing progressive resolution. That evaluation uses six-view test-time augmentation; it is not a legal drop-in or a CIFAR-100 target |

These are source-reported results, not independently reproduced records. This search found no verified public winner under our exact CIFAR-100/75%/40-trial/PCIe contract; the organizer repo's PR listing was empty when checked. No “world SOTA” claim is warranted. Fable's reported whole-run CUDA graph benefit was only about 5 ms on its workload; prioritize reducing useful training work before investing in complex graph capture.

### Bounded experiment matrix

1. **Comparable controls:** run exact #16, #17 and current #14 defaults on the same allocated GPU and seed list; retain #12 as a fallback. Record hardware variant, power limit, software, source hash, build/prepare/train/inference times. Use an ordinary supported allocation; do not select unusually fast hosts, add cooldowns, or alter GPU clocks/power.
2. **#16 plus systems changes:** test fused SGD and indexed crop first, then #14's static compiled forward/loss and fp16 BatchNorm separately. #16 already has fast full-map pooling; benchmark before replacing it with #14's reduction. Preserve #16's inherited #12 augmentation and per-trial generator reset: `jitter` and #14's `color_jitter` are not interchangeable implementations. Fused SGD now has 40-trial support in #14 and #17, but still needs validation on this combination.
3. **Smaller batches:** screen batch 512 against 1024 on #16. #17's benefit includes retuned learning rate, weight decay, epoch count and whitening-bias duration; copy the hypothesis, not its scalar defaults blindly. Account for the change in optimizer-step count and schedule when comparing.
4. **Short resolution search:** compare existing 24→32 and 28→32 schedules, then one conservative 24→28→32 schedule with enough full-resolution training to recover accuracy. #14 already supports `resolution_schedule`, whereas #16 uses a two-stage interface; port only what the experiment requires and warm all paths on synthetic inputs. Cap initial combinations at 6–8, rather than a broad Cartesian sweep.
5. **Optional architecture screen:** revisit [64,256,768] or first-group depth only if the above leaves time. Do not assume gains from different PR architectures add together.

Screen on 4–8 paired seeds, promote at most two combinations to 16, then freeze defaults and evaluate the selected recipe on **40 fresh distinct seeds**, without dropping failures. Compare accuracy changes as well as time changes. Use screening results only for selection; retain all trial records. Re-test the exact exported submission folder in the pinned harness. If a recipe fails the final gate, report that failure and use a predeclared validated fallback; do not keep redrawing seeds until it passes.

**#16 validation gap:** its checked-in HF A/B driver uses `--no-accuracy-target`, ignores the subprocess exit status, and prints means without a completion/trial-count assertion. Its reported means exceed 75%, but that alone does not establish a completed qualifying run. Check raw summaries and all 40 ordered trial records, then run frozen defaults with the target enforced. The reported seed starts generate sequential blocks, unlike #14's random seed lists; use a fresh random 40-seed development file for final verification. Do not describe a diagnostic `qualified: null` as official acceptance.

Prefer approximately **75.25–75.35% mean** as a development margin when it costs only a few percent of time. This is a team selection policy, not an organizer rule. With SD 0.25 percentage points and n=40, the standard error of a mean is about 0.04 pp; #14's reported 0.094–0.130 pp margins warrant care after selection. Neither mean-minus-one-SD nor a confidence interval guarantees the next set of seeds will qualify. Do not pool #14's fused and unfused runs as if they were identical frozen defaults.

Use an organizer-compatible A100 80GB PCIe run before final selection if available. Otherwise publish SXM measurements with that limitation and leave the official score pending; do not convert times using a guessed power ratio. Keep #12's exact recipe as a fallback candidate, and evaluate its 9.0-epoch option if extra margin is needed.

### Final recipe gates

- Fresh trial resets cover weights, BatchNorm buffers, optimizer/momentum, EMA, gradients and custom RNGs after synthetic warmup; no learned state carries over.
- No real-data preprocessing in untimed build; no extra views, fitting or data-dependent state updates during inference. These restrictions rule out importing an external speedrun recipe unchanged.
- Cold build under 600 s, every trial complete, mean accuracy ≥75%, every full inference pass ≤5 s, including the final batch and state checks. Check compile behavior on both training branches and all evaluation shapes.
- Preserve the pinned environment and organizer code; changes inside `cifar100-speedrun/` stay within our submission folder. Use `just` from the root. Development orchestration stays outside the exported competition folder.

Do not reopen Muon, SiLU, aggressive whitening-stem changes, 1×1 late convolutions, EMA removal or BatchNorm recalibration without a specific reason to overturn the negative results in #14. Avoid spending hours chasing a three-second headline unsupported by current measurements.

## 7. Execution and submission schedule

Suggested roles should be claimed by teammates; these are proposed responsibilities, not existing commitments. Protect a final verification window even if experiments slip.

| Window, London time | Role | Deliverable / stop condition |
| --- | --- | --- |
| Saturday evening, first 60–90 minutes | Evaluation owner | Correct candidate/scorer boundary, focused tests and benchmark version on top of merged #15 |
| Saturday evening in parallel | CIFAR owner | Paired #16/#17/#14 controls, #16 qualification check and first combined recipe; request access to official-class hardware |
| Overnight, bounded runs | Framework/evaluation owner | Strong classical baselines and repeated simple/full-framework comparisons; hard cost caps and saved artifacts |
| Sunday 09:00–11:00 | Research owner | Inspect all runs, reproduce the best algorithm, finish the archive ablation; no new domain if evidence is incomplete |
| Sunday 11:00–12:00 | CIFAR owner | Freeze recipe/defaults and complete final 40 trials or select validated fallback |
| Sunday 12:00–13:00 | Whole team | Second-person reproduction; clean export; final held-out comparison for frozen autoresearch candidates |
| Sunday 13:00–14:00 | Submission/demo owner | Record/upload video, prepare email and upstream PR, verify links and licenses |
| Sunday 14:00–14:30 | Submission owner | Submit both track packages and verify receipt/PR visibility; preserve a 15-minute buffer before 14:45 |

If time collapses, ship the corrected evaluator, one harder benchmark, strong baselines, a simple/full comparison and a clear demo. Cut extra ablations, asynchronous scheduling and additional domains first. Do not sacrifice submission preparation for another hyperparameter sweep.

### Demo and handoff

A four-minute video outline: **0:00–0:35** problem and research bottleneck; **0:35–1:10** framework design and relation to prior work; **1:10–2:15** one actual discovery trace; **2:15–3:15** matched quality/cost comparisons and fresh-input results; **3:15–4:00** reproduction, limitations and next steps. Use the existing dashboard with an offline replay so the demonstration does not depend on an LLM responding live. Include CIFAR only as a separately evidenced result unless an actual integrated discovery run exists.

Store raw runs in `artifacts/` as the repo expects. Before temporary compute disappears, export a curated evidence bundle with frozen candidate sources, manifests, ledgers, all trial outcomes and reproduction commands; link it from the eventual submission README. Do not expose hidden seeds until the search is frozen, and exclude credentials, datasets and trained weights.

**Ready to submit when:** a teammate can install with the committed `uv.lock`, run the documented `just autoresearch-*` commands and reproduce the frozen solver comparison; claims match the evidence bundle; the upstream CIFAR PR contains only the team folder; and the autoresearch email contains all four required fields with a playable ≤4-minute Drive video. The official competition PR, email and implementation changes are follow-up work, not actions performed by this Markdown-only PR.
