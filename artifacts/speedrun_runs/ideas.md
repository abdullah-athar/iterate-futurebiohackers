# Ideas beyond the planned search space (3 Oct 2026)

Ranked by expected gain in `dtime_adj` (time saved, with accuracy changes converted to time) per
GPU-minute of screening. Baseline: 8.0 s on the A100 80GB PCIe (300 W), 75.3%, 8.5 epochs x 48
steps at batch 1024. Everything here was checked read-only against `submission.py`, the eager
profile (`20261003-160840_phase0-noise/02_profile-control/profile.json`), torch 2.4.0 in WSL
(CPU), and the public airbench / Hiverge / Fable write-ups (web tools were available; sources at
the end). Nothing was run on a GPU.

## Facts that frame every trade

- **Compute per step** (torch FlopCounterMode on the net, CPU): forward 0.948 GFLOP/image =
  0.97 TFLOP per 1024-batch; forward+backward about 2.9 TFLOP/step. At 19.4 ms/step on PCIe
  that is 150 TFLOP/s = 48% of the A100's 312 TFLOP/s fp16 dense peak (55% on the SXM at
  16.9 ms). FLOP cuts do translate into wall time here (unlike airbench94's tiny net, where the
  paper notes FLOP cuts did not), but about a quarter of the step is memory-bound glue.
- **Group 1 is the inefficient part**: it has 19.6% of the forward FLOPs but 35.8% of the eager
  forward time (1.8x the time per FLOP of group 2, 2.6x of group 3). Cause: its conv1 writes a
  1024x128x31x31 fp16 tensor (252 MB) that the 2x2 max-pool reads again, and the pool backward
  writes another 252 MB; its 128-wide GEMMs (N=128) also underfill tensor cores. Group 2 has
  48.4% of FLOPs / 40.5% of time; group 3 has 31.9% / 22.6%. The single most expensive convs
  are the group-2 and group-3 `conv1`s (102 and 100 GMAC/step each, 21% of forward FLOPs each)
  because they run at full group-input resolution before the pool.
- **The max-pool is not an eager fallback under compile**: torch 2.4 inductor has Triton
  lowerings for `max_pool2d_with_indices` and its backward (2x2 and 3x3 windows); the eager
  4.5% + 2.6% shares are an upper bound for the compiled run. The remaining cost is pure
  traffic (conv output -> pool -> indices -> backward).
- **Prepare**: `reset` 63 ms is mostly `nn.init.dirac_`, which is a Python loop issuing one
  tiny kernel per channel: 2,712 launches per reset for widths 128/384/576 (measured count).
  h2d+normalize 36 ms is dominated by the 154 MB pageable host-to-device copy; the rest of
  prepare is 7 ms. Prepare is 0.07-0.15 s, under 2% of the score.
- **Power cap drift**: on the PCIe card trials 0..7 went 7.88 -> 8.03 s while power sat at
  296-306 W and SM clocks fell 1395 -> 1290 MHz. The 40-trial official mean will be 1.5-2%
  above an n=2 cold estimate, and in a sequential `::ab` the later variant runs hotter unless
  a long cold compile separates them. Fewer bytes per step also means less throttling.
- **Exchange rate is unknown and decides every architecture trade.** Nothing in LOG.md
  measures d(acc)/d(epoch) for this recipe. From airbench-style curves a plausible local
  value is 0.4-0.6 pp per extra epoch (0.95 s on PCIe), i.e. **1 pp ~ 1.6-2.4 s**. Any idea
  that costs 0.3 pp must save 0.5-0.7 s just to break even. Recommendation: spend one
  paired n=8 run on `{"epochs": 9.5}` (plus 7.5 if affordable) before any gamble below.

## Ranked list

| # | Idea | Expected time (PCIe) | Expected acc | Confidence | Screen cost | Verdict |
| --- | --- | ---: | ---: | --- | --- | --- |
| 1 | Eye-based dirac reset (replaces 2,712 launches with 2 ops per conv) | -0.04..-0.05 s | 0 (bit-identical) | high | 0 GPU-min (CPU check), confirm in next control's prepare time | do now |
| 2 | `torch.optim.SGD(..., fused=True)` | -0.03..-0.05 s | ~0 | high | 0 extra (fold into a time-only n=2 with #3) | do now |
| 3 | Move the loss into the compiled graph (fullgraph, return loss) | -0.05..-0.10 s | ~0 | medium | 3 GPU-min with #2 | do now |
| 4 | Compiled-mode profile: kernel sum vs wall (gap size) | 0 (information) | 0 | high | 1-2 GPU-min | before #10 |
| 5 | `torch.compile(..., dynamic=False)` for any multi-shape variant (progressive resizing, batch ramp) | 0..-0.2 s on those variants | 0 | medium | free (fold into the next low-res run) | do with the retry |
| 6 | Pool-first in group 3 only (`pool -> conv1` at 3x3) | -1.0..-1.2 s | -0.3..-1.0 pp | low | n=4 paired, 6-8 GPU-min | best big lever; needs #0 exchange rate |
| 7 | 1x1 convs for `conv2`/`conv3` of group 3 (3x3 maps) | -0.6..-0.7 s | -0.2..-0.6 pp | low | n=4 paired, 6-8 GPU-min | second big lever |
| 8 | Stride-2 `conv1` (padding 0) in group 1, no pool there | -0.5..-0.6 s | -0.3..-1.0 pp | low | n=4 paired, 6-8 GPU-min | gamble |
| 9 | Head learning-rate multiplier (0.5x / 2x) | 0 | +-0.2 pp | low | n=4, 5 GPU-min | cheap hyperparameter |
| 10 | Whole-step CUDA graph: compiled optimizer step + tensor LR | -0.05..-0.15 s | 0 | low | 6-10 GPU-min of debugging | only if #4 shows >= 0.3 ms/step of gaps |
| 11 | Batch ramp 512 -> 1024 (first 2 epochs small) | +0.1 s raw, pays only if it buys ~0.5 epoch | +0.1..+0.4 pp | low | 2 graphs, 6 GPU-min | low |
| 12 | Whitening stem 24 -> 12 channels (drop the +-copies) | -0.05..-0.10 s | -0.1..-0.4 pp | low | 3 GPU-min | skip unless #8 fails |

Items 1-3 together are a near-certain -0.12..-0.20 s (1.5-2.5%). The goal of < 6.0 s (-25%)
is not reachable from systems work alone; it needs #6/#7 (or the planned resolution and
width cuts) to land with small accuracy costs, plus the planned accuracy gainers (jitter,
Muon) converted into fewer epochs.

## Details

**1. Eye-based dirac reset.** `Conv.reset_parameters` calls `nn.init.dirac_(w[:w.size(1)])`,
whose torch 2.4 source is a per-channel Python loop (`tensor[d, d, 1, 1] = 1`): 24+128+128+
128+384+384+384+576+576 = 2,712 kernel launches at ~15-20 us each = most of the 63 ms reset,
every trial. Replace with `k = w.size(1); w[:k].zero_(); w[:k, :, 1, 1].copy_(torch.eye(k,
dtype=w.dtype, device=w.device))` (verified `torch.equal` to `dirac_` on CPU; no RNG use, so
the control path stays bit-identical and `check_variants.py` passes unchanged). Rules: an
identity kernel is an initialization pattern, not a learned constant. The 29 ms of
`kaiming_uniform_` seen on CPU is CPU arithmetic; on the GPU it is one kernel per conv.

**2. Fused SGD.** torch 2.4 `SGD` has `fused=True` with Nesterov and coupled L2 weight decay
(same semantics as the current foreach path, 0.25 ms/step). It groups by (device, dtype), so
the fp16 conv/head weights and fp32 BN biases become ~4 launches instead of ~12. Saving ~0.1
ms/step x 408 = 0.04 s. Not bit-identical (different summation order), accuracy-neutral.
airbench94_muon and Hiverge both use it.

**3. Loss inside the compiled graph.** Give `Net.forward` an optional `labels` argument and
return `F.cross_entropy(logits.float(), labels, label_smoothing, reduction="sum")` when given;
compile with `fullgraph=True` (Hiverge does exactly this). Removes ~10 eager kernels per step
(log_softmax, gather, smoothing mean, their backwards, the fp32 logits round trip) and starts
the backward inside the compiled region. Pitfalls: the eval `Classifier` must keep calling
`net(x)` without labels (a second graph, already warmed by the eval warmup); keep the
`whiten_bias_grad` bool as a Python bool so the two training graphs stay static.

**4. Compiled-mode profile.** `profile_recipe.py` forces eager, and its "busy fraction 2.238"
is an artifact (record_function ranges were summed together with kernels). One run of the
compiled recipe under `torch.profiler` for 12 steps, reporting sum(kernel device time)/wall,
tells whether any CPU-launch gap exists at all (my estimate: CPU work ~3 ms vs 17-19 ms of GPU
per step, so gaps should be < 0.3 ms). It decides #10 and bounds every "overhead" idea.

**5. `dynamic=False`.** `torch.compile(net, mode=...)` is called without `dynamic`; torch
2.4's automatic dynamic shapes mark dims symbolic on the second distinct input shape. In the
Phase 1a low-res run the 28x28 graph was warmed first, then 32x32, so the 32-res epochs may
have run a dynamic-shape graph (generic Triton index math, weaker autotuning), understating
the saving. Pass `dynamic=False` (one static graph per resolution/batch size; build cost
unchanged). Zero risk, possibly a few % on 4.5 epochs of any resizing variant.

**6. Pool-first in group 3.** Today each group runs `conv1` at the input resolution, then
pools. Group 3's `conv1` (384 -> 576 at 7x7, 100 GMAC/step, 21% of forward FLOPs) becomes
18 GMAC at 3x3 if the 2x2 pool moves in front of it; the 58 MB conv output and its pool
traffic disappear too. Expected -14..-17% of step time (-1.0..-1.2 s). Accuracy risk: the
first conv of the group sees 4x fewer positions; airbench chose conv->pool over pool->conv on
CIFAR-10, but there the net is launch-bound and the trade was free. Implementation: a
per-group `pool_first` list in DEFAULTS (`[False, False, True]` to test), one branch in
`ConvGroup.forward`; shapes downstream are unchanged (7 -> 3 -> 3 -> final MaxPool(3)).
Pairs cleanly with the control (same RNG consumption). Break-even at the assumed exchange
rate: Delta acc >= -0.5 pp. Also screen `[False, True, False]` (group 2's conv1 is the other
100 GMAC conv) only if group 3 works.

**7. 1x1 convs in group 3.** `conv2`/`conv3` of group 3 are 3x3 convs on 3x3 maps with "same"
padding: 27.5 GMAC each (11% of forward FLOPs together), mostly multiplying zero padding.
1x1 kernels cut them to 3 GMAC each (-10% FLOPs, about -0.6 s); the final MaxPool(3) and
conv1 still mix space. A `kernel_sizes` per-group switch; dirac init still works (center tap).
Break-even: Delta acc >= -0.3 pp. Cheaper gamble than #8 for similar savings.

**8. Stride-2 conv1 in group 1.** `Conv2d(24, 128, 3, stride=2, padding=0)` maps 31 -> 15
directly, so group 1 loses its MaxPool: conv1 FLOPs 27 -> 7 GMAC (-4% FLOPs) and about
1 GB/step of traffic vanishes (252 MB written by conv1, read by the pool, written by the pool
backward, read by conv1's wgrad; the dgrad is already skipped after `whiten_bias_epochs`).
Expected -6..-8% (-0.5..-0.6 s). Accuracy risk is the loss of the max over 4 positions at the
highest resolution. Pool-before-conv1 in group 1 (pool on the 47 MB whitened tensor) saves the
same FLOPs but likely costs more accuracy; try only if the stride-2 version is close.

**9. Head LR multiplier.** The head sits in the "others" group at the conv LR; airbench94_muon
and Hiverge give it its own LR (12x the bias LR there). A `head_lr_scaler` param costs
nothing to add; expect +-0.2 pp, which is worth 0.3-0.5 s at the exchange rate.

**10. Whole-step CUDA graph.** After #2/#3 the eager remainder per step is the batch gather,
~4 fused-SGD launches, the LR Python loop and the EMA every 5 steps. `torch.compile(opt.step)`
with `lr` as a device tensor filled from a precomputed 408-entry schedule removes the
recompile-per-LR problem; the EMA stays a separate 2-launch foreach. Gain is bounded by #4's
gap measurement (likely < 0.1 s); risks are recompiles, cudagraph-tree memory pools and a
longer cold build. Fable's record used this kind of fusion, but on a 2 s run where overhead
is a much larger fraction than here.

**11. Batch ramp.** 512 for epochs 0-1 then 1024 (LR per-1024 scaling already decoupled). More
updates early is the mechanism; costs a second static graph (+150-230 s cold build in
max-autotune, use default mode for the small-batch graph) and ~10% lower GEMM efficiency on
those epochs. Only pays if it buys >= 0.5 epoch. Low confidence; airbench found 1024 best
for a constant batch.

**12. 12-channel whitening.** Halves conv1's FLOPs in group 1 (-13 GMAC, ~1%) but the
+-duplication followed by GELU is a CReLU-style nonlinearity on the whitened patches and the
paper credits whitening with the largest single saving (45 -> 21 epochs). Expect a small loss.

## Refinements of planned items (from the sources)

- **Progressive resizing**: Fable's 1.828 s record (7.6% over Hiverge) used a 3-stage
  24 -> 28 -> 32 curriculum with "the remainder" at 32; our 28x4 gave -0.70 s / -0.30 +- 0.17
  pp, which is roughly neutral in `dtime_adj`. Try 24x1 + 28x2 + 32x5.5 with `dynamic=False`
  (#5) and low-res graphs in default mode (each extra max-autotune graph adds ~230 s cold).
- **Jitter**: Hiverge's constants are brightness 0.14 / contrast 0.13 on normalized images
  (with label smoothing lowered to 0.09 on CIFAR-10); screen `jitter` 0.12-0.15 first.
- **Muon**: Newton-Schulz on 576x5184 weights every step would cost ~1 s over 408 steps.
  Hiverge runs the orthogonalization only every `2 + int(15 * progress)` steps, padded and
  batched across layers, Muon lr 0.205 / momentum 0.655 (CIFAR-10, batch 1536, 7.65 epochs);
  airbench94_muon keeps the biases, whitening bias and head on fused SGD.
- **Data filtering**: airbench96_faster keeps the 512 highest-loss examples of each 1024 batch
  using losses from a small proxy run and a no-grad pass on 3 of 4 steps; at a 45-epoch
  budget. For 8.5 epochs the only cheap form is to record per-example losses during the normal
  forward (`reduction="none"` then sum, free) and drop the easiest 25% from the index list in
  the last 2-3 epochs (-0.6 epoch of steps, about -0.55 s); the accuracy cost is unknown.
- **Batch 1536**: besides fewer steps, group 3's GEMMs tile better (M = 1024 x 9 rows gives
  3.3 waves on 108 SMs; 1536 x 9 gives 5 full waves). Hiverge used 1536 with fewer epochs.

## Checked and rejected

- BN in fp32: under inductor BN is decomposed into a fused Welford reduction + pointwise;
  the fp32 parameters add no traffic. Running stats need fp32. No gain. Folding BN into the
  conv is impossible while batch statistics are used (eval is untimed anyway).
- GELU tanh approximation / SiLU for speed: the fused pointwise kernels are memory-bound;
  activation choice is an accuracy question only (planned).
- Pinned staging buffer for the H2D copy: allowed (allocate in build, copy in prepare) but a
  CPU memcpy into pinned memory plus DMA is not faster than the pageable path for 154 MB;
  mean/std from a 5k subset saves ~3 ms and changes normalization. Both under 0.01 s.
- Lookahead EMA cost: 82 EMA steps x 0.33 ms = 0.027 s total; `ema_every` 10 (planned) halves
  it. Not worth more work.
- Last partial epoch: 408 steps is the schedule; the half epoch's crop+randperm wastes 3 ms.
  Using the 848 images dropped each epoch (50000 - 48 x 1024) changes nothing per step.
- Eval-batch warmup: inference is 0.10-0.19 s against a 5 s untimed limit.
- INT8 tensor cores, 2:4 sparsity, bf16: no torch 2.4 training path / no gain on A100.
- CIFAR-100 coarse labels as an auxiliary loss: the harness only hands over fine labels;
  reading the dataset files or hard-coding the 100 -> 20 mapping is label information outside
  the API. Treat as prohibited unless the organizers say otherwise.
- Anything Fable was called out for (setup moved outside the timer, instance selection,
  cooldowns) is explicitly banned by RULES.md section 3.

## Methodology notes for the orchestrator

- Calibrate d(acc)/d(epoch) once (n=8 paired, epochs 9.5); every entry above is ranked
  against an assumed 1 pp ~ 2 s.
- Systems items (#1-#3, #5) are accuracy-neutral: screen them at n=2 for time only, in one
  container, as default-off switches, then promote together. Keep `check_variants.py`'s
  bit-identical control test; #1 passes it, #2/#3 need a `fused`/`compiled_loss` switch.
- Architecture gambles (#6-#8) need n >= 4 paired and a hard stop rule: drop the variant if
  Delta acc < -0.6 pp at n=4 regardless of time.
- Run the variant before the control in every second `::ab` (or report per-trial-index
  times): the power cap makes the later run 1-2% slower.

## Sources

- Keller Jordan, "94% on CIFAR-10 in 3.29 Seconds on a Single GPU" (arXiv 2404.00498):
  additive epoch savings (whitening 45 -> 21, dirac -3, bias LR -4.5, lookahead -1.5,
  alternating flip -0.9; multi-crop -1.2 is TTA), architecture 31 -> 15 -> 7 -> 3,
  airbench96 on CIFAR-100 79.27% at its 96% budget. https://arxiv.org/abs/2404.00498
- KellerJordan/cifar10-airbench: airbench94_muon.py (fused SGD for biases/head, Muon for
  4D weights, batch 2000, 8 epochs) and airbench96_faster.py (proxy-run loss masks, 512 of
  1024 per batch). https://github.com/KellerJordan/cifar10-airbench
- hiverge/cifar10-speedrun (1.98 s): SiLU, SVD whitening, brightness/contrast jitter,
  vectorized crop, compiled forward+loss, periodic Muon normalization, batch 1536, 7.65
  epochs, selective TTA (not allowed here). https://github.com/hiverge/cifar10-speedrun and
  https://www.hiverge.ai/blog/cifar-speedrun
- Fulcrum, "Fable is SOTA at CIFAR Speedrun" (9 Jul 2026): 1.828 s via 24 -> 28 -> 32
  progressive resizing (~0.15 s), pooling / CUDA-graph / optimizer-fusion work, and the three
  rule violations. https://fulcrum.inc/2026/07/09/fable-cifar-speedrun.html
- tysam-code/hlb-CIFAR10: dirac init on non-transition layers, channels_last/fp16, conv-pool
  ordering lineage. https://github.com/tysam-code/hlb-CIFAR10
