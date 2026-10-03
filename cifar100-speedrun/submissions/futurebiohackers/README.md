# futurebiohackers: faster CIFAR-100 training

This is a convolutional image classifier based on Keller Jordan's
[airbench](https://github.com/KellerJordan/cifar10-airbench), trained from scratch.
The selected model uses three convolution blocks with 96, 256 and 768 channels.
It makes the first two blocks smaller, where images are largest and processing is
expensive, and gives the final block more capacity to distinguish the 100 classes.

Training uses 9.5 epochs, batches of 1024 images, half precision, channels-last
memory layout, Nesterov SGD and label smoothing. Convolution blocks use SiLU
activations, and the forward pass and training loss compile together. A moving average stabilizes the
weights. Training-image normalization and patch whitening happen inside the timer.
The final ordinary max pool covers the whole feature map, avoiding the slower
atomic backward kernel used by adaptive pooling. Experimental smaller-crop
training also works.

## Development results

All **40 fresh trials** of each recipe below completed successfully. Each pair
shared one Modal A100 SXM allocation; pairs used different allocations.

| Recipe | Trials | Mean accuracy | Mean preparation + training |
| --- | ---: | ---: | ---: |
| Pair A: frozen PR #5, GELU + adaptive pooling | 40 | 75.395% | 6.736 s |
| Pair A: ordinary full-map max pooling | 40 | 75.328% | 6.517 s |
| Pair A: ordinary pooling + compiled forward/loss | 40 | 75.33425% | 6.463 s |
| Pair B: frozen PR #5, GELU + adaptive pooling | 40 | 75.28775% | 6.755 s |
| Pair B: selected SiLU + ordinary pooling + compiled forward/loss | 40 | 75.12575% | 6.309 s |

The selected recipe was **6.6% faster** than the paired PR #5 control. Its
accuracy standard deviation was 0.241 percentage points, and its time standard
deviation was 0.015 seconds. Pair A used seeds 40000–40039, and pair B used seeds
50000–50039. No trials were discarded. SiLU has a smaller accuracy margin than
the GELU variant. This is development validation; official judging still
requires an A100 80GB PCIe and the organizer's private 40 seeds.

PyTorch Profiler found 0.335 seconds in the final adaptive max-pool backward
kernel across 456 training steps. Replacing that kernel motivated the pooling
experiment; these profiler times include diagnostic overhead and are separate
from the scored harness measurements above. SiLU and compiling the forward pass
with the loss were inspired by [Hiverge's CIFAR-10 speedrun](https://github.com/hiverge/cifar10-speedrun).
Its test-time augmentation is excluded because this challenge requires one view.

PR #5's earlier 96/256/768 model passed 40 trials at 75.2495% / 6.859 s,
8.0% faster than its three-trial original-width control. The
under-three-second target has not been reached. The original recipe in PR #3
previously reached 75.29% in 8.11 s across 40 trials on another Modal allocation.

The 8.5-epoch version of the new architecture initially looked promising over
three trials, but a fresh 40-trial run averaged 74.902% in 5.772 s and missed the
accuracy gate. Extending training to 9.5 epochs recovered the accuracy margin.
The failed check remains in the experiment log.

Run the selected defaults from the repository root with `just modal 40`.
The successful SiLU result is `20261003T171035Z-534b9a20`; its paired control
is `20261003T170512Z-da23dc47`. The archived run records contain the full source,
explicit parameter overrides, all ordered trial records and environment details.

## Experiments and progress

`--params` exposes smaller early crops (24 or 28 pixels followed by 32), different
block widths/depths, proxy-based hard-example selection, alternative pooling and
optimizer settings, per-image training brightness/contrast jitter, and an optional
Triton crop/flip kernel. Grouped Muon is an experimental optimizer, disabled in the
selected recipe. These other experiments are
turned off in the selected defaults. The 28px, nine-epoch candidate missed
the target over five fresh trials; proxy selection and fused optimizer settings
also failed to improve the qualifying result. The unsuccessful experimental
BN/GELU fusion was excluded from the submitted source.

`scripts/modal_experiments.py` runs bounded comparisons through the unchanged
competition harness and reserves spending against a $50 cap.
`scripts/track_speedrun.py` writes a live log under
`artifacts/runtime-optimization/`, sorted by time, with each experiment's main
changes, accuracy, trial count and paired control. It also generates a local
review dashboard under `.lavish/`.

## Data and evaluation checks

Build warms compilation and kernels using random synthetic images only. Every
trial resets model weights, BatchNorm statistics, gradients, optimizer state,
moving averages and any proxy masks. Preparation and training use only the
harness-provided training images and labels. Evaluation uses one image view and
frozen training statistics; it does not fit to test images, use test-batch
statistics or change registered model state.

The [rules](https://github.com/AIDDA-Institute/CIFAR-100-speedrun/blob/main/RULES.md)
allow comparing reported test accuracy during development. Test results never
choose a stopping point or hard-example masks within a trial. Source review found
no test-data leakage; official acceptance still requires organizer review and the
prescribed environment.

The correctness checks cover resets, immutable training inputs, evaluation state
and batch independence (27 CPU tests), plus exact crop/flip equivalence across
layouts and resolutions (36 GPU checks) and exact ordinary/adaptive pooling
outputs and gradients, including ties (24 GPU checks). The GPU benchmark applies the harness's
own output and state checks. Run the CPU checks from `cifar100-speedrun/` with
`uv run python -m pytest -q ../scripts/test_speedrun_recipe.py`.

Adapted under the MIT license, Copyright (c) 2024 Keller Jordan. The full original
permission notice is preserved in `LICENSE.airbench`.
The experimental Muon implementation adapts ideas and Newton-Schulz coefficients
from Hiverge; its original MIT permission notice is preserved in `LICENSE.hiverge`.
