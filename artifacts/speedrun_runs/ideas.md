# Ideas beyond the planned search space (3 Oct 2026)

Ranked by expected `dtime_adj` gain (time saved, accuracy changes converted to time) per GPU-minute
of screening. Baseline 8.0 s / 75.3% on the A100 80GB PCIe (300 W), 8.5 epochs x 48 steps, batch
1024. Checked read-only against `submission.py`, the eager profile
(`20261003-160840_phase0-noise/02_profile-control/profile.json`), torch 2.4.0 in WSL (CPU only) and
the airbench / Hiverge / Fable write-ups (web tools were available; sources at the end). No GPU
was used.

## Facts that frame every trade

- **Compute**: forward 0.948 GFLOP/image (FlopCounterMode) = 0.97 TFLOP per step, about 2.9
  TFLOP with the backward. 19.4 ms/step on PCIe = 150 TFLOP/s = 48% of fp16 peak (55% on SXM).
  FLOP cuts do turn into wall time here (the airbench paper's "FLOPs but not wallclock" remark is
  about its launch-bound 64/256/256 net), but roughly a quarter of the step is memory-bound glue.
- **Group 1 is inefficient**: 19.6% of forward FLOPs, 35.8% of eager forward time (1.8x group 2's
  time per FLOP, 2.6x group 3's). Its conv1 writes a 1024x128x31x31 fp16 tensor (252 MB) that the
  2x2 pool reads and whose gradient the pool backward writes again; N=128 GEMMs underfill tensor
  cores. Group 2: 48.4% FLOPs / 40.5% time; group 3: 31.9% / 22.6%. The two biggest single
  convs are the group-2 and group-3 `conv1`s (102 and 100 GMAC/step, 21% of forward FLOPs each):
  they run at the group's input resolution, before its pool.
- **Max-pool under compile**: torch 2.4 inductor lowers `max_pool2d_with_indices` and its
  backward to Triton for 2x2/3x3 windows, so the eager 4.5% + 2.6% is an upper bound; what
  remains is traffic (conv output -> pool -> indices -> backward).
- **Prepare** (0.07-0.15 s, < 2%): `reset` 63 ms is mostly `nn.init.dirac_`, a Python loop that
  launches one tiny kernel per channel: 2,712 launches per reset at widths 128/384/576 (counted).
  h2d+normalize 36 ms is the 154 MB pageable copy; the rest is 7 ms.
- **Power cap drift**: PCIe trials 0..7 went 7.88 -> 8.03 s at 296-306 W with SM clocks 1395 ->
  1290 MHz. The 40-trial mean sits 1.5-2% above a cold n=2, and in a sequential `::ab` the later
  variant runs hotter unless a long cold compile separates the runs. Fewer bytes/step also means
  less throttling.
- **Exchange rate unknown**: nothing in LOG.md measures d(acc)/d(epoch) for this recipe. A
  plausible local value is 0.4-0.6 pp per extra epoch (0.95 s), i.e. **1 pp ~ 1.6-2.4 s**: an idea
  costing 0.3 pp must save 0.5-0.7 s to break even. Spend one paired n=8 on `{"epochs": 9.5}`
  before any architecture gamble.

## Ranked list

| # | Idea | Time (PCIe) | Accuracy | Conf. | Screen cost | Verdict |
| --- | --- | ---: | ---: | --- | --- | --- |
| 1 | Eye-based dirac reset (2 ops per conv instead of 2,712 launches) | -0.04..-0.05 s | 0, bit-identical | high | 0 GPU-min; read prepare time of the next control | do now |
| 2 | `SGD(..., fused=True)` | -0.03..-0.05 s | ~0 | high | fold into #3's run | do now |
| 3 | Loss inside the compiled graph (fullgraph, returns loss) | -0.05..-0.10 s | ~0 | medium | one n=2, 3 GPU-min | do now |
| 4 | Compiled-mode profile (kernel sum vs wall) | information | 0 | high | 1-2 GPU-min | before #10 |
| 5 | `torch.compile(..., dynamic=False)` for multi-shape variants | 0..-0.2 s on those variants | 0 | medium | free with the low-res retry | do |
| 6 | Pool-first in group 3 (`pool -> conv1` at 3x3) | -1.0..-1.2 s | -0.3..-1.0 pp | low | n=4 paired, 6-8 GPU-min | biggest lever |
| 7 | 1x1 `conv2`/`conv3` in group 3 (3x3 maps) | -0.6..-0.7 s | -0.2..-0.6 pp | low | n=4 paired, 6-8 GPU-min | second lever |
| 8 | Stride-2 `conv1` (padding 0) in group 1, pool removed | -0.5..-0.6 s | -0.3..-1.0 pp | low | n=4 paired, 6-8 GPU-min | gamble |
| 9 | Head LR multiplier 0.5x / 2x | 0 | +-0.2 pp | low | n=4, 5 GPU-min | cheap knob |
| 10 | Whole-step CUDA graph (compiled optimizer, tensor LR) | -0.05..-0.15 s | 0 | low | 6-10 GPU-min of debugging | only if #4 shows >= 0.3 ms/step gaps |
| 11 | Batch ramp 512 (epochs 0-1) -> 1024 | +0.1 s raw; pays only if it buys ~0.5 epoch | +0.1..+0.4 pp | low | 2 graphs, 6 GPU-min | low |
| 12 | Whitening stem 24 -> 12 channels | -0.05..-0.10 s | -0.1..-0.4 pp | low | 3 GPU-min | skip unless #8 fails |

#1-#3 are a near-certain -0.12..-0.20 s. Reaching < 6.0 s (-25%) needs #6/#7 or the planned
resolution/width cuts to land with small accuracy costs, plus the planned accuracy gainers
(jitter, Muon) converted into fewer epochs.

## Details

1. **Dirac reset.** `Conv.reset_parameters` calls `nn.init.dirac_(w[:w.size(1)])`, whose torch
   2.4 source loops `tensor[d, d, 1, 1] = 1` per channel: 24+128+128+128+384+384+384+576+576 =
   2,712 launches at ~15-20 us = most of the 63 ms. Replace with `k = w.size(1); w[:k].zero_();
   w[:k, :, 1, 1].copy_(torch.eye(k, dtype=w.dtype, device=w.device))`; `torch.equal` to
   `dirac_` verified on CPU, no RNG use, so the control stays bit-identical and
   `check_variants.py` passes. An identity kernel is an init pattern, not a learned constant.
2. **Fused SGD.** torch 2.4 `SGD(fused=True)` supports Nesterov and the same coupled L2 decay;
   it groups by (device, dtype), so fp16 weights + fp32 BN biases become ~4 launches instead of
   ~12 (0.25 ms/step today). Not bit-identical (summation order); airbench94_muon and Hiverge
   use it.
3. **Loss in the graph.** Give `Net.forward` an optional `labels` argument returning
   `F.cross_entropy(logits.float(), labels, label_smoothing=..., reduction="sum")`, compile with
   `fullgraph=True` (Hiverge does this). Removes ~10 eager kernels per step and starts the
   backward inside the compiled region. The eval `Classifier` keeps calling `net(x)` (its graph
   is already warmed); keep `whiten_bias_grad` a Python bool so both training graphs stay static.
4. **Compiled profile.** `profile_recipe.py` forces eager, and its "busy fraction 2.238" summed
   record_function ranges with kernels. Profile 12 compiled steps, report kernel-sum/wall. My
   estimate is ~3 ms CPU vs 17-19 ms GPU per step (no gap); this bounds every overhead idea.
5. **`dynamic=False`.** `torch.compile(net, mode=...)` has no `dynamic`; torch 2.4 marks dims
   symbolic on the second distinct shape. Phase 1a warmed 28x28 first, then 32x32, so the 32-res
   epochs may have run a dynamic-shape graph (generic index math, weaker autotune) and the -0.70 s
   may be understated. One static graph per resolution or batch size, build cost unchanged.
6. **Pool-first in group 3.** Moving the 2x2 pool ahead of group 3's `conv1` (384 -> 576) runs it
   at 3x3: 100 -> 18 GMAC/step (-17% forward FLOPs), and the 58 MB conv output and its pool
   traffic vanish; downstream shapes are unchanged (3 -> 3 -> MaxPool(3)). Expected -14..-17% of
   step time. Risk: the group's first conv sees 4x fewer positions; airbench kept conv -> pool on
   CIFAR-10 but there the trade was free. Implement as a per-group `pool_first` list in DEFAULTS
   (`[False, False, True]`), one branch in `ConvGroup.forward`, same RNG consumption. Break-even
   at the assumed rate: Delta acc >= -0.5 pp. Try `[False, True, False]` (the other 100 GMAC
   conv) only if group 3 works.
7. **1x1 convs in group 3.** `conv2`/`conv3` are 3x3 convs on 3x3 maps with "same" padding: 27.5
   GMAC each (11% of FLOPs), mostly multiplying zero padding. 1x1 kernels: 3 GMAC each, about
   -0.6 s; `conv1` and the final MaxPool(3) still mix space; dirac init still works (center tap).
   Per-group `kernel_sizes` switch. Break-even Delta acc >= -0.3 pp.
8. **Stride-2 conv1 in group 1.** `Conv2d(24, 128, 3, stride=2, padding=0)` maps 31 -> 15, so the
   group loses its pool: conv1 27 -> 7 GMAC and ~1 GB/step of traffic disappears (the 252 MB
   written, pooled, written back by the pool backward and read by the wgrad; the dgrad is already
   skipped after `whiten_bias_epochs`). Expected -6..-8%. Risk: no max over 4 positions at full
   resolution. Pool-before-conv1 in group 1 saves the same FLOPs but likely costs more accuracy.
9. **Head LR multiplier.** The head sits in "others" at the conv LR; airbench94_muon and Hiverge
   give it its own LR (12x the bias LR). A `head_lr_scaler` param; +-0.2 pp is worth 0.3-0.5 s.
10. **Whole-step CUDA graph.** After #2/#3 the eager remainder is the batch gather, ~4 fused-SGD
    launches, the LR loop and the EMA every 5 steps. `torch.compile(opt.step)` with `lr` as a
    device tensor filled from a 408-entry schedule avoids recompiles; gain is bounded by #4
    (likely < 0.1 s); risks are recompiles, cudagraph pools and a longer cold build. Fable used
    this on a 2 s run where overhead matters far more than here.
11. **Batch ramp.** More updates early; needs a second static graph (+150-230 s cold in
    max-autotune, so compile it in default mode) and ~10% worse GEMM efficiency on those epochs.
    airbench found a constant 1024 best.
12. **12-channel whitening.** Halves conv1's FLOPs in group 1 (~1%), but the +-copies + GELU act
    as a CReLU on whitened patches, and whitening is the paper's largest saving (45 -> 21 epochs).

## Refinements of planned items (from the sources)

- **Progressive resizing**: Fable's 1.828 s record used 24 -> 28 -> 32 with "the remainder" at 32
  (~0.15 s of 1.98 s). Our 28x4 (-0.70 s, -0.30 +- 0.17 pp) is about neutral in `dtime_adj`; try
  24x1 + 28x2 + 32x5.5 with `dynamic=False` and low-res graphs in default mode (each extra
  max-autotune graph adds ~230 s cold).
- **Jitter**: Hiverge uses brightness 0.14 / contrast 0.13 on normalized images (and label
  smoothing 0.09 on CIFAR-10); screen `jitter` 0.12-0.15 first.
- **Muon**: Newton-Schulz on 576x5184 weights every step costs ~1 s over 408 steps. Hiverge
  orthogonalizes only every `2 + int(15 * progress)` steps, padded and batched across layers
  (Muon lr 0.205, momentum 0.655, batch 1536, 7.65 epochs); biases, whitening bias and head
  stay on fused SGD.
- **Data filtering**: airbench96_faster keeps the 512 highest-loss of each 1024 using a proxy
  run's losses at a 45-epoch budget. The only cheap form at 8.5 epochs: record per-example
  losses in the normal forward (`reduction="none"` then sum, free) and drop the easiest 25% from
  the index list in the last 2-3 epochs (-0.6 epoch, ~-0.55 s); accuracy cost unknown.
- **Batch 1536**: group 3's GEMMs tile better (1024 x 9 rows = 3.3 waves on 108 SMs; 1536 x 9 =
  5 full waves). Hiverge paired it with fewer epochs.

## Checked and rejected

- BN in fp32: inductor decomposes BN into a fused Welford reduction + pointwise; fp32 params add
  no traffic; running stats need fp32. Folding BN into convs is impossible with batch statistics
  (eval is untimed anyway). GELU-tanh/SiLU for speed: memory-bound kernels, accuracy question only.
- Pinned staging buffer for the 154 MB copy (allowed: allocate in build, fill in prepare): CPU
  memcpy + DMA is not faster than the pageable path; mean/std from 5k images saves ~3 ms.
- Lookahead cost 82 x 0.33 ms = 0.027 s (planned `ema_every` 10 halves it). Last half epoch
  wastes 3 ms of crop/randperm; the 848 images dropped per epoch change nothing per step.
  Eval warmup: inference 0.10-0.19 s against an untimed 5 s limit.
- INT8 tensor cores, 2:4 sparsity, bf16: no torch 2.4 training path or no gain on A100.
- CIFAR-100 coarse labels as an auxiliary loss: label information outside the harness API
  (files or a hard-coded 100 -> 20 map); treat as prohibited. Everything Fable was called out
  for (work outside the timer, instance selection, cooldowns) is banned by RULES.md section 3.

## Methodology notes

- Calibrate d(acc)/d(epoch) once (paired n=8, epochs 9.5); the ranking assumes 1 pp ~ 2 s.
- #1-#3, #5 are accuracy-neutral: screen at n=2 for time only, as default-off switches in one
  container, then promote together; #1 passes the bit-identical control test, #2/#3 need a switch.
- #6-#8 need n >= 4 paired and a stop rule: drop the variant if Delta acc < -0.6 pp at n=4.
- Alternate variant/control order between `::ab` launches (or compare per trial index): the
  power cap makes the later run 1-2% slower.

## Sources

- Jordan, "94% on CIFAR-10 in 3.29 Seconds on a Single GPU" (arXiv 2404.00498): whitening
  45 -> 21 epochs, dirac -3, bias LR -4.5, lookahead -1.5, alternating flip -0.9 (multi-crop
  -1.2 is TTA); 31 -> 15 -> 7 -> 3; airbench96 79.27% on CIFAR-100 at its 96% budget.
- KellerJordan/cifar10-airbench: airbench94_muon.py (fused SGD for biases/head, Muon for 4D
  weights) and airbench96_faster.py (proxy-run loss masks, 512 of 1024 per batch).
- hiverge/cifar10-speedrun and hiverge.ai/blog/cifar-speedrun (1.98 s): SiLU, SVD whitening,
  jitter 0.14/0.13, vectorized crop, compiled forward+loss, periodic Muon normalization, batch
  1536, 7.65 epochs, selective TTA (not allowed here).
- fulcrum.inc/2026/07/09/fable-cifar-speedrun.html: 1.828 s via 24 -> 28 -> 32 resizing
  (~0.15 s), pooling / CUDA-graph / optimizer-fusion work, three rule violations.
- tysam-code/hlb-CIFAR10: dirac init on non-transition layers, channels_last/fp16 lineage.
