"""Experimental grouped Muon updates, inspired by Hiverge's CIFAR speedrun.

Reference: https://github.com/hiverge/cifar10-speedrun (MIT; LICENSE.hiverge).
Group filters by output width to reduce padding in the batched Newton-Schulz
iteration. Timed prepare() replaces optimizer state; timed training initializes
fresh momentum buffers. No learned values persist between trials.
"""

import math

import torch
import torch.nn.functional as F


def orthogonalize(gradients):
    """Three quintic Newton-Schulz steps on equally wide padded matrices."""
    columns = max(g.numel() // len(g) for g in gradients)
    matrices = [F.pad(g.flatten(1), (0, columns - g.numel() // len(g))) for g in gradients]
    x = torch.stack(matrices)
    # Accumulate norms in float32 so fp16 gradients cannot overflow the reduction.
    x = (x / (x.float().norm(dim=(1, 2), keepdim=True) + 1e-5)).to(x.dtype)
    transpose = x.size(1) > x.size(2)
    if transpose:
        x = x.transpose(1, 2)
    for _ in range(3):
        a = x @ x.transpose(1, 2)
        b = -4.7391 * a + 2.0843 * (a @ a)
        x = 3.4576 * x + b @ x
    if transpose:
        x = x.transpose(1, 2)
    return [x[i, :, : g.numel() // len(g)].reshape_as(g) for i, g in enumerate(gradients)]


class Muon(torch.optim.Optimizer):
    """Nesterov filter updates with periodic weight normalization."""

    def __init__(self, params, lr, momentum, weight_decay, zeropower):
        groups = {}
        for p in params:
            groups.setdefault(len(p), []).append(p)
        super().__init__(
            [dict(params=values) for values in groups.values()],
            dict(lr=lr, initial_lr=lr, momentum=momentum, weight_decay=weight_decay),
        )
        self.zeropower = zeropower
        self.steps = 0
        self.last_normalized = 0

    @torch.no_grad()
    def step(self, progress):
        self.steps += 1
        normalize = self.steps - self.last_normalized >= 2 + int(15 * progress)
        if normalize:
            self.last_normalized = self.steps
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            buffers = []
            for p in params:
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buffers.append(state["momentum_buffer"])
            grads = [p.grad for p in params]
            torch._foreach_mul_(buffers, group["momentum"])
            torch._foreach_add_(buffers, grads)
            nesterov = torch._foreach_add(grads, buffers, alpha=group["momentum"])
            updates = self.zeropower(nesterov)
            if normalize:
                norms = torch._foreach_norm(params)
                scales = [
                    (math.sqrt(len(p)) / (norm + 1e-7)).to(p.dtype)
                    for p, norm in zip(params, norms)
                ]
                torch._foreach_mul_(params, scales)
            torch._foreach_add_(params, updates, alpha=-group["lr"])
            if group["weight_decay"]:
                torch._foreach_mul_(params, 1 - group["lr"] * group["weight_decay"])
