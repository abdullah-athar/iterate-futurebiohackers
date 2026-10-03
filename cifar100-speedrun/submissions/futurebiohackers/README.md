# futurebiohackers: faster CIFAR-100 training

This is a convolutional image classifier based on Keller Jordan's
[airbench](https://github.com/KellerJordan/cifar10-airbench), trained from scratch.
The selected model uses three convolution blocks with 96, 256 and 768 channels.
It makes the first two blocks smaller, where images are largest and processing is
expensive, and gives the final block more capacity to distinguish the 100 classes.

Training uses 9.5 epochs, batches of 1024 images, half precision, channels-last
memory layout, Nesterov SGD and label smoothing. A moving average stabilizes the
weights. Training-image normalization and patch whitening happen inside the timer.
The final spatial pooling now covers the whole feature map, so experimental
smaller-crop training also works.

## Development results

All **40 fresh trials** of the selected 9.5-epoch recipe completed successfully:

| Recipe | Trials | Mean accuracy | Mean preparation + training |
| --- | ---: | ---: | ---: |
| Original 128/384/576 model, 8.5 epochs | 3 | 75.480% | 7.458 s |
| Selected 96/256/768 model, 9.5 epochs | 40 | 75.2495% | 6.859 s |

Both runs shared one Modal A100 SXM allocation. The selected recipe was **8.0%
faster** than the paired control. Its accuracy standard deviation was 0.264
percentage points, and its time standard deviation was 0.029 seconds. The new
recipe used seeds 20000–20039; the control used the first three of those seeds.
No trials were discarded. This is development validation; official judging still
requires an A100 80GB PCIe and the organizer's private 40 seeds.

The under-three-second target has not been reached. The original recipe in PR #3
previously reached 75.29% in 8.11 s across 40 trials on another Modal allocation.

The 8.5-epoch version of the new architecture initially looked promising over
three trials, but a fresh 40-trial run averaged 74.902% in 5.772 s and missed the
accuracy gate. Extending training to 9.5 epochs recovered the accuracy margin.
The failed check remains in the experiment log.

Run the selected defaults from the repository root with `just modal 40`.
The successful 40-trial result is `20261003T155840Z-bee7f356`; its paired control
is `20261003T155703Z-6e4f0e48`.

## Experiments and progress

`--params` exposes smaller early crops (24 or 28 pixels followed by 32), different
block widths/depths, proxy-based hard-example selection, alternative pooling and
optimizer settings, and an optional Triton crop/flip kernel. These experiments are
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
and batch independence (13 CPU tests), plus exact crop/flip equivalence across
layouts and resolutions (36 GPU checks). The GPU benchmark applies the harness's
own output and state checks. Run the CPU checks from `cifar100-speedrun/` with
`uv run python -m pytest -q ../scripts/test_speedrun_recipe.py`.

Adapted under the MIT license, Copyright (c) 2024 Keller Jordan. The full original
permission notice is preserved in `LICENSE.airbench`.
