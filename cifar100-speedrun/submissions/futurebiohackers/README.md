# futurebiohackers: faster CIFAR-100 training

This is a convolutional image classifier based on Keller Jordan's
[airbench](https://github.com/KellerJordan/cifar10-airbench), trained from scratch.
It uses three convolution blocks with 96, 256 and 768 channels. Smaller early
blocks reduce work where images are largest; the final block retains capacity
for distinguishing CIFAR-100's 100 classes.

The selected recipe trains for **eight epochs in batches of 512**. It uses half
precision, channels-last layout, SiLU activations, Nesterov SGD, label smoothing
and a moving average of model weights. The model's forward pass and loss compile
together. Ordinary max pooling covers the whole final feature map, avoiding the
slower adaptive-pool backward kernel.

Compared with PR #10, the main changes are smaller batches, fewer epochs and
PyTorch's fused SGD update, which combines optimizer operations into fewer GPU
kernels. Learning rate is 18, weight decay is 0.01425, and the whitening bias
trains for 1.5 epochs. Training-image normalization and patch whitening remain
inside the timer.

## Development validation

All **40 fresh trials** of both recipes completed successfully on the same Modal
A100 SXM allocation, using seeds 160000–160039.

| Recipe | Trials | Mean accuracy | Mean preparation + training |
| --- | ---: | ---: | ---: |
| Frozen PR #10: batch 1024, 9.5 epochs, ordinary SGD | 40 | 75.1825% | 6.285433 s |
| Selected: batch 512, eight epochs, fused SGD | 40 | 75.225% | 5.575874 s |

The selected recipe was **11.3% faster** than the paired control. Accuracy
standard deviation was 0.244 percentage points; time standard deviation was
0.019 seconds. Every ordered seed and trial succeeded; no trials were discarded.
The submission source matches the validated snapshot byte for byte (SHA-256
`354cc62f0dbd550005c3ad3ca5de5368ab8bd7786124f5b75a704d0a03066d89`).

This beats the 5.8-second development milestone. The under-three-second target
has not been reached. Official judging still requires an A100 80GB PCIe and the
organizer's private 40 seeds; an SXM result is not official acceptance.

Successful result: `20261003T183951Z-d023aca8`; paired control:
`20261003T183430Z-6993f0dc`. The archived records contain the source, all ordered
trial records and environment details. Run defaults with `just modal 40`.

The unfused smaller-batch recipe separately passed all 40 seeds 140000–140039 at
75.1245% / 6.314 s, versus its same-allocation PR #10 control at 75.1585% / 6.908 s.
Different pairs use different GPU allocations; compare their paired speedups.

## Profiling and experiments

PyTorch Profiler found 0.335 seconds in adaptive max-pool backward across 456
steps, motivating the whole-map pooling change. Optimizer GPU work consumed
another 0.231 seconds in that diagnostic, motivating the separate fused-SGD
screen. Profiler times include diagnostic overhead and are separate from scored
harness measurements. SiLU and compiling the model with the loss were inspired
by [Hiverge's CIFAR-10 speedrun](https://github.com/hiverge/cifar10-speedrun).
Its test-time augmentation is excluded because this challenge requires one view.

Optional parameters cover smaller early crops, block widths/depths, proxy-based
hard-example selection, alternative optimizers, per-image training color jitter,
and a Triton crop/flip kernel. These experiments are off in selected defaults.
A separate CNN/attention prototype failed accuracy and is not in this submission.

`scripts/modal_experiments.py` runs bounded comparisons through the unchanged
harness and reserves spending against a $50 cap. `scripts/track_speedrun.py`
writes `artifacts/runtime-optimization/progress.md` and a local review dashboard,
sorted by time with the main change, accuracy, trial count and paired control.
Unsuccessful and incomplete runs remain in that log.

## Data and evaluation checks

Build warms compilation and kernels using synthetic images only. Every trial
resets weights, BatchNorm statistics, gradients, optimizer state, moving averages
and any proxy masks. Preparation and training use only the harness-provided
training images and labels. Evaluation uses one view per image and frozen
training statistics; it never fits to test images, uses test-batch statistics or
changes model state. Predictions do not depend on other test images.

The [rules](https://github.com/AIDDA-Institute/CIFAR-100-speedrun/blob/main/RULES.md)
allow comparing reported test accuracy during development. Test results never
choose stopping points or hard-example masks within a trial. Source review found
no test-data leakage; official acceptance still requires organizer review.

27 CPU correctness checks pass in PyTorch 2.4, including resets, immutable inputs,
evaluation state and image independence. GPU checks cover 36 crop/flip cases and
24 exact pooling output/gradient cases, including ties. Each scored trial also
passes the unchanged harness's output and state checks.

Adapted under the MIT license, Copyright (c) 2024 Keller Jordan; the full notice is
preserved in `LICENSE.airbench`. The optional Muon implementation adapts ideas and
Newton–Schulz coefficients from Hiverge; its notice is in `LICENSE.hiverge`.
