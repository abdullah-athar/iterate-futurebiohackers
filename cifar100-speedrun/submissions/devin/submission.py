"""CIFAR-100 speedrun recipe: an airbench-style network trained from scratch.

Adapted from Keller Jordan's airbench (https://github.com/KellerJordan/cifar10-airbench),
Copyright (c) 2024 Keller Jordan, released under the MIT License. Changes: 100-class
head, 96/256/768 blocks with global max pooling, label smoothing 0.25, an 8.75-epoch
schedule whose first quarter trains on 24x24 crops, per-image brightness/contrast
jitter, the harness build/prepare/train split, and no test-time augmentation.

Untimed build() compiles the network and warms up every kernel on synthetic data.
Timed prepare() resets all learned state, moves the images to the GPU, normalizes
them, and initializes the patch-whitening layer from training images.
"""

import math
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from benchmark.api import BuildContext, TrainingData

# Override any value with --params, e.g. '{"epochs": 9, "widths": [128, 384, 768]}'.
DEFAULTS = {
    "epochs": 8.25,
    "batch_size": 1024,
    "lr": 11.5,  # per 1024 examples, decoupled from momentum (airbench convention)
    "momentum": 0.85,
    "weight_decay": 0.017,  # per 1024 examples, decoupled from the learning rate
    "bias_scaler": 32.0,  # learning-rate multiplier for BatchNorm biases
    "label_smoothing": 0.25,
    "warmup": 0.23,  # fraction of steps spent ramping the learning rate up
    "final_lr": 0.07,  # learning-rate multiplier reached at the last step
    "whiten_bias_epochs": 3,
    "translate": 2,
    "cutout": 0,
    "widths": [128, 256, 768],
    "depth": 3,  # convs per group; the third adds a residual connection
    "depths": [2, 3, 3],  # per-group convs; None uses depth for every group
    "inner_widths": [0, 0, 0],  # >0 narrows a group's residual pair through that many channels
    "inner_kernels": [[3, 3], [3, 3], [3, 3]],  # per-group [conv2, conv3] kernel sizes (1 or 3)
    "bn_dtype": "float",  # "half" keeps BatchNorm in the fp16 network dtype
    "train_resolution": 24,  # reduced resolution for the first training stage
    "resolution_switch": 0.25,  # fraction of steps before returning to 32 pixels
    # [[resolution, end fraction], ...] before 32 px; overrides the two keys above if set
    "resolution_schedule": None,
    "crop_mode": "indexed",  # "indexed" preserves channels-last with one gather
    "fused_sgd": True,
    "optimizer": "sgd",  # "muon": orthogonalized updates for conv filters, SGD for the rest
    "muon_lr": 0.24,
    "muon_momentum": 0.6,
    "muon_ns_steps": 3,
    "muon_compile": "step",  # "step" compiles the per-bucket Muon update; "none" runs eagerly
    "compile_loss": True,
    "hard_fraction": 1.0,  # <1 enables a freshly trained small proxy
    "proxy_widths": [32, 64, 128],
    "proxy_every": 4,  # proxy backward/update period; scores every batch
    "proxy_mode": "offline",  # airbench-style prepass, or online selection
    "gelu_approximate": "none",  # "tanh" uses a cheaper approximation; "silu" swaps in SiLU
    "global_pool": "max",  # "max": plain full-map max pool, avoids adaptive atomics
    "autotune_backends": "ATEN,TRITON",  # ATen/cuDNN and Inductor Triton candidates
    "pool_first": [False, False, False],  # move selected group pools before conv1
    "scaling_factor": 1.25 / 9,  # logit scale (airbench uses 1/9)
    "bn_momentum": 0.5,
    "ema_every": 5,  # lookahead EMA period in steps; 0 disables it
    "compile": "max-autotune",  # torch.compile mode; "" runs eagerly
    # Accuracy-recovery and resizing switches.
    "jitter": 0.3,  # per-image brightness/contrast jitter strength (own RNG, seeded per trial)
    "low_res_compile": "default",  # torch.compile mode of the separate static low-res graph
    "count_nonfinite": False,  # count non-finite step losses on the GPU; train() prints the total
}


#############################################
#                  Network                  #
#############################################


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum):
        super().__init__(num_features, eps=1e-12, momentum=1 - momentum)
        self.weight.requires_grad = False

    def activate(self, x, approximate, residual=None):
        x = self(x)
        if residual is not None:
            x = x + residual
        return _activate(x, approximate)


class Conv(nn.Conv2d):
    def __init__(self, channels_in, channels_out, kernel=3):
        super().__init__(channels_in, channels_out, kernel, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        nn.init.dirac_(w[: w.size(1)])


class ConvGroup(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_out,
        depth,
        bn_momentum,
        gelu_approximate,
        pool_first,
        inner=0,
        kernels=(3, 3),
    ):
        super().__init__()
        inner = inner if inner and depth == 3 else channels_out
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.pool_first = pool_first
        self.norm1 = BatchNorm(channels_out, bn_momentum)
        self.conv2 = Conv(channels_out, inner, kernels[0])
        self.norm2 = BatchNorm(inner, bn_momentum)
        self.conv3 = Conv(inner, channels_out, kernels[1]) if depth == 3 else None
        self.norm3 = BatchNorm(channels_out, bn_momentum) if depth == 3 else None
        self.approximate = gelu_approximate

    def forward(self, x):
        x = self.conv1(self.pool(x)) if self.pool_first else self.pool(self.conv1(x))
        x = self.norm1.activate(x, self.approximate)
        if self.conv3 is None:
            return self.norm2.activate(self.conv2(x), self.approximate)
        x0 = x
        x = self.norm2.activate(self.conv2(x), self.approximate)
        return self.norm3.activate(self.conv3(x), self.approximate, x0)


def _activate(x, approximate):
    return F.silu(x) if approximate == "silu" else F.gelu(x, approximate=approximate)


class Activation(nn.Module):
    def __init__(self, approximate):
        super().__init__()
        self.approximate = approximate

    def forward(self, x):
        return _activate(x, self.approximate)


class GlobalMaxPool(nn.Module):
    def __init__(self, implementation):
        super().__init__()
        self.implementation = implementation

    def forward(self, x):
        if self.implementation == "max":
            return F.max_pool2d(x, x.shape[-2:])
        if self.implementation == "maxmean":
            return F.max_pool2d(x, x.shape[-2:]) + F.avg_pool2d(x, x.shape[-2:])
        return F.adaptive_max_pool2d(x, 1)


class Net(nn.Module):
    def __init__(self, hyp, num_classes):
        super().__init__()
        w1, w2, w3 = hyp["widths"]
        depths = hyp["depths"] or [hyp["depth"]] * 3
        bn_momentum = hyp["bn_momentum"]
        self.whiten = nn.Conv2d(3, 24, kernel_size=2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            Activation(hyp["gelu_approximate"]),
            ConvGroup(
                24,
                w1,
                depths[0],
                bn_momentum,
                hyp["gelu_approximate"],
                hyp["pool_first"][0],
                hyp["inner_widths"][0],
                hyp["inner_kernels"][0],
            ),
            ConvGroup(
                w1,
                w2,
                depths[1],
                bn_momentum,
                hyp["gelu_approximate"],
                hyp["pool_first"][1],
                hyp["inner_widths"][1],
                hyp["inner_kernels"][1],
            ),
            ConvGroup(
                w2,
                w3,
                depths[2],
                bn_momentum,
                hyp["gelu_approximate"],
                hyp["pool_first"][2],
                hyp["inner_widths"][2],
                hyp["inner_kernels"][2],
            ),
            GlobalMaxPool(hyp["global_pool"]),
        )
        self.head = nn.Linear(w3, num_classes, bias=False)
        self.scaling_factor = hyp["scaling_factor"]

    def reset(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d | nn.BatchNorm2d | nn.Linear):
                m.reset_parameters()
        self.whiten.bias.data.zero_()

    @torch.no_grad()
    def init_whiten(self, images, eps=5e-4):
        c, (h, w) = images.shape[1], self.whiten.weight.shape[2:]
        patches = images.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w)
        flat = patches.float().view(len(patches), -1)
        covariance = flat.T @ flat / len(flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance, UPLO="U")
        scaled = eigenvectors.T.reshape(-1, c, h, w) / torch.sqrt(
            eigenvalues.view(-1, 1, 1, 1) + eps
        )
        self.whiten.weight.copy_(torch.cat((scaled, -scaled)))

    def forward(self, x, whiten_bias_grad: bool = True):
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x).flatten(1)
        return self.head(x) * self.scaling_factor


class Classifier(nn.Module):
    """Evaluation wrapper: harness inputs are float32 RGB in [0, 1]."""

    def __init__(self, net, dtype):
        super().__init__()
        self.net = net
        self.dtype = dtype
        self.register_buffer("mean", torch.zeros(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.ones(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        x = ((x - self.mean) / self.std).to(self.dtype, memory_format=torch.channels_last)
        return self.net(x).float()


#############################################
#               Augmentation                #
#############################################


def batch_flip_lr(images):
    flip = (torch.rand(len(images), device=images.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip, images.flip(-1), images)


def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size) // 2
    shifts = torch.randint(-r, r + 1, size=(len(images), 2), device=images.device)
    out = torch.empty(
        (len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype
    )
    if r <= 2:
        for sy in range(-r, r + 1):
            for sx in range(-r, r + 1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                out[mask] = images[
                    mask, :, r + sy : r + sy + crop_size, r + sx : r + sx + crop_size
                ]
    else:
        tmp = torch.empty(
            (len(images), 3, crop_size, crop_size + 2 * r), device=images.device, dtype=images.dtype
        )
        for s in range(-r, r + 1):
            mask = shifts[:, 0] == s
            tmp[mask] = images[mask, :, r + s : r + s + crop_size, :]
        for s in range(-r, r + 1):
            mask = shifts[:, 1] == s
            out[mask] = tmp[mask, :, :, r + s : r + s + crop_size]
    return out


def batch_cutout(images, size):
    n, _, h, w = images.shape
    y = torch.randint(0, h - size + 1, size=(n, 1, 1, 1), device=images.device)
    x = torch.randint(0, w - size + 1, size=(n, 1, 1, 1), device=images.device)
    rows = torch.arange(h, device=images.device).view(1, 1, h, 1) - y
    cols = torch.arange(w, device=images.device).view(1, 1, 1, w) - x
    mask = (rows >= 0) & (rows < size) & (cols >= 0) & (cols < size)
    return images.masked_fill(mask, 0)


def batch_jitter(images, strength, generator):
    """Per-image contrast (x * c) and brightness (+ b) jitter, c in [1-s, 1+s], b in [-s, s]
    in normalized units; drawn from `generator` so the global RNG stream is untouched."""
    n = len(images)
    uniform = torch.rand(2, n, device=images.device, generator=generator) * 2 - 1
    contrast = (1 + strength * uniform[0]).to(images.dtype).view(-1, 1, 1, 1)
    brightness = (strength * uniform[1]).to(images.dtype).view(-1, 1, 1, 1)
    return images * contrast + brightness


def indexed_crop(images, crop_size):
    """One gather per epoch, with the same uniform crop distribution as batch_crop."""
    n, _, h, w = images.shape
    shifts = torch.randint(0, w - crop_size + 1, (n, 2), device=images.device)
    batch = torch.arange(n, device=images.device).view(n, 1, 1)
    rows = shifts[:, 0, None, None] + torch.arange(crop_size, device=images.device).view(1, -1, 1)
    cols = shifts[:, 1, None, None] + torch.arange(crop_size, device=images.device).view(1, 1, -1)
    return images.permute(0, 2, 3, 1)[batch, rows, cols].permute(0, 3, 1, 2)


#############################################
#                 Interface                 #
#############################################


def _resolution_schedule(hyp):
    """[(resolution, end fraction of steps), ...] for the stages before 32 px."""
    if hyp["resolution_schedule"] is not None:
        return [(int(r), float(end)) for r, end in hyp["resolution_schedule"]]
    if hyp["train_resolution"] < 32 and hyp["resolution_switch"] > 0:
        return [(hyp["train_resolution"], hyp["resolution_switch"])]
    return []


def build(context: BuildContext):
    hyp = {**DEFAULTS, **context.parameters}
    unknown = set(hyp) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown parameters: {sorted(unknown)}")
    if len(hyp["widths"]) != 3 or any(w <= 0 for w in hyp["widths"]):
        raise ValueError("widths must contain three positive channel counts")
    depths = hyp["depths"] or [hyp["depth"]] * 3
    if len(depths) != 3 or any(d not in (2, 3) for d in depths):
        raise ValueError("depths must contain three values of 2 or 3")
    if hyp["train_resolution"] not in (24, 28, 32) or not 0 <= hyp["resolution_switch"] <= 1:
        raise ValueError(
            "train_resolution must be 24, 28, or 32; resolution_switch must be in [0, 1]"
        )
    schedule = _resolution_schedule(hyp)
    ends = [end for _, end in schedule]
    if (
        any(r not in (16, 20, 24, 28) for r, _ in schedule)
        or ends != sorted(ends)
        or any(not 0 < end <= 1 for end in ends)
    ):
        raise ValueError(
            "resolution_schedule entries need resolutions 16-28 and increasing ends in (0, 1]"
        )
    if len(hyp["inner_widths"]) != 3 or any(w < 0 for w in hyp["inner_widths"]):
        raise ValueError("inner_widths must contain three non-negative channel counts")
    if len(hyp["inner_kernels"]) != 3 or any(
        len(k) != 2 or any(v not in (1, 3) for v in k) for k in hyp["inner_kernels"]
    ):
        raise ValueError("inner_kernels must contain three [k2, k3] pairs of 1 or 3")
    if hyp["bn_dtype"] not in ("float", "half"):
        raise ValueError("bn_dtype must be float or half")
    if hyp["global_pool"] not in ("max", "maxmean", "adaptive"):
        raise ValueError("global_pool must be max, maxmean or adaptive")
    if hyp["crop_mode"] not in ("masked", "indexed", "triton"):
        raise ValueError("crop_mode must be masked, indexed, or triton")
    if hyp["epochs"] <= 0 or hyp["batch_size"] <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if not 0 < hyp["hard_fraction"] <= 1 or hyp["proxy_every"] < 1:
        raise ValueError("hard_fraction must be in (0, 1]; proxy_every must be positive")
    if hyp["proxy_mode"] not in ("online", "offline"):
        raise ValueError("proxy_mode must be online or offline")
    if hyp["gelu_approximate"] not in ("none", "tanh", "silu"):
        raise ValueError("gelu_approximate must be none, tanh or silu")
    if hyp["autotune_backends"] not in ("ATEN", "TRITON", "ATEN,TRITON"):
        raise ValueError("autotune_backends must be ATEN, TRITON, or ATEN,TRITON")
    if len(hyp["pool_first"]) != 3 or any(type(v) is not bool for v in hyp["pool_first"]):
        raise ValueError("pool_first must contain three booleans")
    if hyp["jitter"] < 0:
        raise ValueError("jitter must be non-negative")
    device = context.device
    cuda = device.type == "cuda"
    dtype = torch.float16 if cuda else torch.float32
    crop_kernel = None
    if cuda and hyp["crop_mode"] == "triton":
        from .kernels import crop_flip

        crop_kernel = crop_flip
    torch.backends.cudnn.benchmark = True
    import torch._inductor.config as inductor_config

    inductor_config.max_autotune_gemm_backends = hyp["autotune_backends"]

    net = Net(hyp, context.num_classes).to(device, dtype, memory_format=torch.channels_last)
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d) and hyp["bn_dtype"] == "float":
            m.float()
    if cuda and hyp["compile"] and schedule:
        # Progressive resizing: one static graph per resolution over the same eager net.
        # dynamic=False keeps dynamo from switching to dynamic shapes on the second size, and
        # the low-res graph compiles in the cheaper low_res_compile mode (a second
        # max-autotune graph added about 230 s to the cold build).
        train_net = torch.compile(net, mode=hyp["compile"], dynamic=False)
        low_net = torch.compile(net, mode=hyp["low_res_compile"], dynamic=False)
    elif cuda and hyp["compile"]:
        train_net = low_net = torch.compile(net, mode=hyp["compile"])
    else:
        train_net = low_net = net

    def loss_fn(outputs, labels):
        return F.cross_entropy(
            outputs.float(), labels, label_smoothing=hyp["label_smoothing"], reduction="sum"
        )

    loss_fn = torch.compile(loss_fn) if cuda and hyp["compile_loss"] else loss_fn
    float_state = [t for t in net.state_dict().values() if t.is_floating_point()]
    state = SimpleNamespace(
        hyp=hyp,
        device=device,
        dtype=dtype,
        net=net,
        train_net=train_net,
        low_net=low_net,
        loss_fn=loss_fn,
        classifier=Classifier(net, dtype).to(device),
        float_state=float_state,
        ema=[t.clone() for t in float_state],
        proxy=None,
        crop_kernel=crop_kernel,
        aug_generator=None,
        nonfinite=None,
    )
    if hyp["hard_fraction"] < 1:
        proxy_hyp = {**hyp, "widths": hyp["proxy_widths"], "depths": [2, 2, 2]}
        proxy = Net(proxy_hyp, context.num_classes).to(
            device, dtype, memory_format=torch.channels_last
        )
        proxy.whiten.bias.requires_grad_(False)
        for m in proxy.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.float()
        state.proxy = proxy
        state.proxy_net = (
            torch.compile(proxy, mode=hyp["compile"]) if cuda and hyp["compile"] else proxy
        )

    # Untimed warmup on random synthetic images: compiles both whitening-bias graphs,
    # autotunes cuDNN, initializes cuBLAS/cuSOLVER, and warms evaluation shapes.
    # prepare() in each trial resets everything this changes.
    if cuda:
        count = 50_000
        synthetic = TrainingData(
            torch.randint(0, 256, (count, 3, 32, 32), dtype=torch.uint8),
            torch.randint(0, context.num_classes, (count,)),
        )
        for resolution in sorted({r for r, _ in schedule} | {32}):
            state.warmup_resolution = resolution
            for _ in range(2):
                prepare(state, synthetic, seed=0)
                state.whiten_bias_steps = 3
                _fit(state, total_steps=6)
        del state.warmup_resolution
        state.classifier.eval()
        with torch.inference_mode():
            for size in (context.eval_batch_size, 10_000 % context.eval_batch_size, 1):
                state.classifier(torch.rand(size, 3, 32, 32, device=device))
        state.classifier.train()
        torch.cuda.synchronize()
    return state


def prepare(state, data: TrainingData, seed: int) -> None:
    """Timed: reset every learned value and stage the training data on the GPU."""
    hyp, net, device = state.hyp, state.net, state.device
    net.reset()
    net.zero_grad(set_to_none=True)
    net.train()

    raw = data.images.to(device, non_blocking=True).float().div_(255)
    mean = raw.mean(dim=(0, 2, 3), keepdim=True)
    std = raw.std(dim=(0, 2, 3), keepdim=True)
    state.classifier.mean.copy_(mean)
    state.classifier.std.copy_(std)
    images = ((raw - mean) / std).to(state.dtype, memory_format=torch.channels_last)
    del raw
    net.init_whiten(images[:5000])
    if state.proxy is not None:
        state.proxy.reset()
        state.proxy.zero_grad(set_to_none=True)
        state.proxy.train()
        state.proxy.whiten.weight.data.copy_(net.whiten.weight)
    torch._foreach_copy_(state.ema, state.float_state)
    # Extra augmentation randomness comes from its own generator, re-seeded every trial, so
    # the global stream (init, data order, flips, crops) stays identical to the base recipe.
    state.aug_generator = (
        torch.Generator(device=device).manual_seed(seed) if hyp["jitter"] else None
    )

    # Alternating flip: flip a random half once, then mirror everything on odd epochs.
    images = batch_flip_lr(images)
    state.small_images = {}
    for resolution in {r for r, _ in _resolution_schedule(hyp)}:
        small = F.interpolate(images, size=(resolution,) * 2, mode="bilinear", align_corners=False)
        if hyp["translate"]:
            small = F.pad(small, (hyp["translate"],) * 4, "reflect")
        state.small_images[resolution] = small.to(memory_format=torch.channels_last)
    if hyp["translate"]:
        images = F.pad(images, (hyp["translate"],) * 4, "reflect")
    state.images = images
    state.labels = data.labels.to(device, non_blocking=True)

    batch_size = min(hyp["batch_size"], len(data.labels))
    momentum = hyp["momentum"]
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = hyp["lr"] / kilostep_scale
    wd = hyp["weight_decay"] * batch_size / kilostep_scale
    lr_biases = lr * hyp["bias_scaler"]
    state.optimizer = _make_optimizer(net, lr, lr_biases, wd * hyp["hard_fraction"], hyp, device)
    if state.proxy is not None:
        state.proxy_optimizer = _make_optimizer(state.proxy, lr, lr_biases, wd, hyp, device)

    state.batch_size = batch_size
    state.masks = []
    state.steps_per_epoch = len(data.labels) // batch_size
    state.total_steps = math.ceil(hyp["epochs"] * state.steps_per_epoch)
    state.whiten_bias_steps = math.ceil(hyp["whiten_bias_epochs"] * state.steps_per_epoch)


def _newton_schulz(g, steps: int, eps: float = 1e-7):
    """Batched quintic Newton-Schulz orthogonalization of [n, rows, cols] with rows <= cols."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    x = x / (x.norm(dim=(1, 2), keepdim=True) + eps)
    for _ in range(steps):
        m = x @ x.mT
        x = a * x + (b * m + c * m @ m) @ x
    return x


def _muon_updates(params, grads, buffers, momentum: float, ns_steps: int):
    """Nesterov momentum, filter renormalization and the orthogonalized update for one bucket."""
    torch._foreach_mul_(buffers, momentum)
    torch._foreach_add_(buffers, grads)
    g = torch.stack(torch._foreach_add(grads, buffers, alpha=momentum)).flatten(2)
    transpose = g.size(1) > g.size(2)
    u = _newton_schulz(g.mT if transpose else g, ns_steps)
    u = (u.mT if transpose else u).reshape(len(params), *params[0].shape)
    norms = torch._foreach_norm(params)
    torch._foreach_mul_(params, [len(p) ** 0.5 / n for p, n in zip(params, norms)])
    return list(u.to(params[0].dtype).unbind(0))


class Muon(torch.optim.Optimizer):
    """Keller Jordan's Muon (airbench94_muon): Nesterov momentum, then each filter bank's
    update is orthogonalized by a Newton-Schulz iteration; weights are renormalized.
    Filters of one shape share a batched Newton-Schulz call; compile="step" compiles the
    whole per-bucket update (the learning rate is applied outside, so it never recompiles)."""

    def __init__(self, params, lr, momentum, ns_steps, compile="none"):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps))
        shapes = {}
        for p in self.param_groups[0]["params"]:
            shapes.setdefault(tuple(p.shape), []).append(p)
        self.buckets = list(shapes.values())
        self.flat = [p for ps in self.buckets for p in ps]
        self.buffers = [[torch.zeros_like(p) for p in ps] for ps in self.buckets]
        self.updates = (
            torch.compile(_muon_updates, dynamic=False) if compile == "step" else (_muon_updates)
        )

    @torch.no_grad()
    def step(self):
        group = self.param_groups[0]
        updates = []
        for params, buffers in zip(self.buckets, self.buffers):
            grads = [p.grad for p in params]
            updates += self.updates(params, grads, buffers, group["momentum"], group["ns_steps"])
        torch._foreach_add_(self.flat, updates, alpha=-group["lr"])


class _Optimizers:
    def __init__(self, *optimizers):
        self.optimizers = optimizers
        self.param_groups = [g for o in optimizers for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.optimizers:
            o.step()


def _make_optimizer(net, lr, lr_biases, wd, hyp, device):
    if hyp["optimizer"] not in ("sgd", "muon"):
        raise ValueError("optimizer must be sgd or muon")
    trainable = [(name, p) for name, p in net.named_parameters() if p.requires_grad]
    filters = [p for _, p in trainable if hyp["optimizer"] == "muon" and p.ndim == 4]
    norm_biases = [p for name, p in trainable if "norm" in name]
    others = [p for name, p in trainable if "norm" not in name and not any(p is f for f in filters)]
    optimizer = torch.optim.SGD(
        [
            dict(params=norm_biases, lr=lr_biases, weight_decay=wd / lr_biases),
            dict(params=others, lr=lr, weight_decay=wd / lr),
        ],
        momentum=hyp["momentum"],
        nesterov=True,
        fused=hyp["fused_sgd"] and device.type == "cuda",
    )
    if filters:
        muon = Muon(
            filters,
            hyp["muon_lr"],
            hyp["muon_momentum"],
            hyp["muon_ns_steps"],
            compile=hyp["muon_compile"] if device.type == "cuda" else "none",
        )
        optimizer = _Optimizers(optimizer, muon)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer


def train(state) -> nn.Module:
    if state.proxy is not None and state.hyp["proxy_mode"] == "offline":
        # Replay the same augmentation/order stream for the main net. Only masks
        # persist within this trial; prepare discards them before the next one.
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(state.device) if state.device.type == "cuda" else None
        _fit(state, state.total_steps, proxy_only=True)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, state.device)
    _fit(state, state.total_steps)
    if hyp_count_nonfinite(state):
        # One host read at the very end (the harness synchronizes here anyway). The worker's
        # stderr is inherited, so the launcher can parse this line from the container log.
        state.nonfinite_losses = int(state.nonfinite)
        print(f"NONFINITE_LOSSES {state.nonfinite_losses}", file=sys.stderr, flush=True)
    return state.classifier


def hyp_count_nonfinite(state) -> bool:
    return bool(state.hyp["count_nonfinite"]) and state.nonfinite is not None


def _fit(state, total_steps, proxy_only=False):
    hyp, net, optimizer = state.hyp, state.net, state.optimizer
    labels, batch_size, steps_per_epoch = state.labels, state.batch_size, state.steps_per_epoch
    warmup_steps = int(total_steps * hyp["warmup"])
    ema_decay = (0.95**5 * (torch.arange(total_steps + 1) / total_steps) ** 3).tolist()
    nonfinite = (
        torch.zeros((), dtype=torch.int64, device=labels.device)
        if hyp["count_nonfinite"] and not proxy_only
        else None
    )
    step = 0
    stage_ends = [(r, int(total_steps * end)) for r, end in _resolution_schedule(hyp)]
    net.train()
    for epoch in range(math.ceil(total_steps / steps_per_epoch)):
        images = None
        current_resolution = None
        order = torch.randperm(len(labels), device=labels.device)
        for i in range(steps_per_epoch):
            if step >= total_steps:
                break
            resolution = getattr(state, "warmup_resolution", None)
            if resolution is None:
                resolution = next((r for r, end in stage_ends if step < end), 32)
            if resolution != current_resolution:
                source = state.small_images[resolution] if resolution < 32 else state.images
                if state.crop_kernel is not None:
                    images = state.crop_kernel(source, resolution, flip=epoch % 2 == 1)
                else:
                    crop = batch_crop if hyp["crop_mode"] == "masked" else indexed_crop
                    images = crop(source, resolution) if hyp["translate"] else source
                if epoch % 2 == 1 and state.crop_kernel is None:
                    images = images.flip(-1)
                if hyp["cutout"]:
                    images = batch_cutout(images, hyp["cutout"])
                if hyp["jitter"]:
                    images = batch_jitter(images, hyp["jitter"], state.aug_generator)
                current_resolution = resolution
                train_net = state.train_net if resolution == 32 else state.low_net
            idx = order[i * batch_size : (i + 1) * batch_size]
            inputs, targets = images[idx], labels[idx]
            offline_main = (
                state.proxy is not None
                and hyp["proxy_mode"] == "offline"
                and not proxy_only
                and not hasattr(state, "warmup_resolution")
            )
            if offline_main:
                chosen = state.masks[step]
                inputs, targets = inputs[chosen], targets[chosen]
            elif state.proxy is not None:
                update_proxy = step % hyp["proxy_every"] == 0
                with torch.set_grad_enabled(update_proxy):
                    proxy_outputs = state.proxy_net(inputs, False)
                    proxy_losses = F.cross_entropy(
                        proxy_outputs.float(),
                        targets,
                        reduction="none",
                        label_smoothing=hyp["label_smoothing"],
                    )
                kept = max(1, int(batch_size * hyp["hard_fraction"]))
                chosen = proxy_losses.detach().topk(kept, sorted=False).indices
                if update_proxy:
                    state.proxy_optimizer.zero_grad(set_to_none=True)
                    proxy_losses[chosen].sum().backward()
                    proxy_frac = step / max(1, total_steps)
                    proxy_scale = min(1.0, 0.2 + 8 * proxy_frac) * (1 - proxy_frac)
                    for group in state.proxy_optimizer.param_groups:
                        group["lr"] = group["initial_lr"] * proxy_scale
                    state.proxy_optimizer.step()
                inputs, targets = inputs[chosen], targets[chosen]
                if proxy_only:
                    state.masks.append(chosen)
                    step += 1
                    continue
            outputs = train_net(inputs, step < state.whiten_bias_steps)
            loss = state.loss_fn(outputs, targets)
            if nonfinite is not None:
                nonfinite += (~torch.isfinite(loss.detach())).to(nonfinite.dtype)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if step < warmup_steps:
                frac = step / warmup_steps
                scale = 0.2 * (1 - frac) + frac
            else:
                frac = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                scale = (1 - frac) + hyp["final_lr"] * frac
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * scale
            optimizer.step()
            step += 1
            if hyp["ema_every"] and step % hyp["ema_every"] == 0:
                _lookahead(state, ema_decay[step])
    if hyp["ema_every"] and not proxy_only:
        _lookahead(state, 1.0)
    if not proxy_only:
        state.nonfinite = nonfinite


@torch.no_grad()
def _lookahead(state, decay):
    torch._foreach_lerp_(state.ema, state.float_state, 1 - decay)
    torch._foreach_copy_(state.float_state, state.ema)
