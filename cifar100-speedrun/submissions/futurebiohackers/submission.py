"""CIFAR-100 speedrun recipe: an airbench-style network trained from scratch.

Adapted from Keller Jordan's airbench (https://github.com/KellerJordan/cifar10-airbench),
Copyright (c) 2024 Keller Jordan, released under the MIT License. Changes: 100-class
head with a wider last block, label smoothing 0.3, an 8.5-epoch schedule, the
harness build/prepare/train split, and no test-time augmentation.

Untimed build() compiles the network and warms up every kernel on synthetic data.
Timed prepare() resets all learned state, moves the images to the GPU, normalizes
them, and initializes the patch-whitening layer from training images.
"""

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from benchmark.api import BuildContext, TrainingData

# Override any value with --params, e.g. '{"epochs": 9, "widths": [128, 384, 768]}'.
DEFAULTS = {
    "epochs": 8.5,
    "batch_size": 1024,
    "lr": 9.0,  # per 1024 examples, decoupled from momentum (airbench convention)
    "momentum": 0.85,
    "weight_decay": 0.012,  # per 1024 examples, decoupled from the learning rate
    "bias_scaler": 64.0,  # learning-rate multiplier for BatchNorm biases
    "label_smoothing": 0.3,
    "warmup": 0.23,  # fraction of steps spent ramping the learning rate up
    "final_lr": 0.07,  # learning-rate multiplier reached at the last step
    "whiten_bias_epochs": 3,
    "translate": 2,
    "cutout": 0,
    "widths": [128, 384, 576],
    "depth": 3,  # convs per group; the third adds a residual connection
    "scaling_factor": 1 / 9,
    "bn_momentum": 0.6,
    "ema_every": 5,  # lookahead EMA period in steps; 0 disables it
    "compile": "max-autotune",  # torch.compile mode; "" runs eagerly
    # Experimental switches (the defaults reproduce the recipe above exactly).
    "depths": None,  # per-group convs, e.g. [3, 2, 4]; None uses "depth" for every group
    "activation": "gelu",  # or "silu"
    "whiten_svd": False,  # whitening basis from an SVD instead of eigh
    "crop_gather": False,  # vectorized random crop (one gather, same shift distribution)
    "low_res": 0,  # progressive resizing: train epochs with index < low_res_epochs at this size
    "low_res_epochs": 0,
    "jitter": 0.0,  # per-image brightness/contrast jitter strength (own RNG, seeded per trial)
}


#############################################
#                  Network                  #
#############################################


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum):
        super().__init__(num_features, eps=1e-12, momentum=1 - momentum)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, channels_in, channels_out):
        super().__init__(channels_in, channels_out, kernel_size=3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        nn.init.dirac_(w[: w.size(1)])


def _activation(name):
    return {"gelu": nn.GELU, "silu": nn.SiLU}[name]()


class ConvGroup(nn.Module):
    """conv-pool-BN-act, then depth-1 more convs: a residual pair (depth 3), a plain conv
    (depth 2), or a plain conv followed by a residual pair (depth 4)."""

    def __init__(self, channels_in, channels_out, depth, bn_momentum, activation="gelu"):
        super().__init__()
        if depth not in (2, 3, 4):
            raise ValueError(f"depth must be 2, 3 or 4, got {depth}")
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out, bn_momentum)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out, bn_momentum)
        self.conv3 = Conv(channels_out, channels_out) if depth >= 3 else None
        self.norm3 = BatchNorm(channels_out, bn_momentum) if depth >= 3 else None
        self.conv4 = Conv(channels_out, channels_out) if depth == 4 else None
        self.norm4 = BatchNorm(channels_out, bn_momentum) if depth == 4 else None
        self.activ = _activation(activation)

    def forward(self, x):
        x = self.activ(self.norm1(self.pool(self.conv1(x))))
        if self.conv3 is None:
            return self.activ(self.norm2(self.conv2(x)))
        if self.conv4 is None:
            x0 = x
            x = self.activ(self.norm2(self.conv2(x)))
            return self.activ(self.norm3(self.conv3(x)) + x0)
        x = self.activ(self.norm2(self.conv2(x)))
        x0 = x
        x = self.activ(self.norm3(self.conv3(x)))
        return self.activ(self.norm4(self.conv4(x)) + x0)


class Net(nn.Module):
    def __init__(self, hyp, num_classes):
        super().__init__()
        w1, w2, w3 = hyp["widths"]
        d1, d2, d3 = hyp["depths"] or [hyp["depth"]] * 3
        bn_momentum, act = hyp["bn_momentum"], hyp["activation"]
        self.whiten = nn.Conv2d(3, 24, kernel_size=2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            _activation(act),
            ConvGroup(24, w1, d1, bn_momentum, act),
            ConvGroup(w1, w2, d2, bn_momentum, act),
            ConvGroup(w2, w3, d3, bn_momentum, act),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(w3, num_classes, bias=False)
        self.scaling_factor = hyp["scaling_factor"]
        self.whiten_svd = hyp["whiten_svd"]

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
        if self.whiten_svd:
            eigenvectors, eigenvalues, _ = torch.linalg.svd(covariance)
        else:
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


def batch_crop_gather(images, crop_size):
    """Same shift distribution (one randint draw) as batch_crop, done with a single gather
    instead of 25 masked copies, each of which forces a GPU->CPU sync."""
    n, c, h, w = images.shape
    r = (w - crop_size) // 2
    shifts = torch.randint(-r, r + 1, size=(n, 2), device=images.device)
    offsets = torch.arange(crop_size, device=images.device)
    rows = r + shifts[:, 0:1] + offsets  # (n, crop)
    cols = r + shifts[:, 1:2] + offsets
    index = (rows[:, :, None] * w + cols[:, None, :]).view(n, 1, crop_size * crop_size)
    flat = images.reshape(n, c, h * w).gather(2, index.expand(n, c, -1))
    return flat.view(n, c, crop_size, crop_size).contiguous(memory_format=torch.channels_last)


def batch_jitter(images, strength, generator):
    """Per-image contrast (x * c) and brightness (+ b) jitter, c in [1-s, 1+s], b in [-s, s]
    in normalized units; drawn from `generator` so the global RNG stream is untouched."""
    n = len(images)
    uniform = torch.rand(2, n, device=images.device, generator=generator) * 2 - 1
    contrast = (1 + strength * uniform[0]).to(images.dtype).view(-1, 1, 1, 1)
    brightness = (strength * uniform[1]).to(images.dtype).view(-1, 1, 1, 1)
    return images * contrast + brightness


def batch_cutout(images, size):
    n, _, h, w = images.shape
    y = torch.randint(0, h - size + 1, size=(n, 1, 1, 1), device=images.device)
    x = torch.randint(0, w - size + 1, size=(n, 1, 1, 1), device=images.device)
    rows = torch.arange(h, device=images.device).view(1, 1, h, 1) - y
    cols = torch.arange(w, device=images.device).view(1, 1, 1, w) - x
    mask = (rows >= 0) & (rows < size) & (cols >= 0) & (cols < size)
    return images.masked_fill(mask, 0)


#############################################
#                 Interface                 #
#############################################


def build(context: BuildContext):
    hyp = {**DEFAULTS, **context.parameters}
    unknown = set(hyp) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown parameters: {sorted(unknown)}")
    # The 2x2 whitening conv and three pools need (size - 1) // 8 >= 3 for the final 3x3 pool.
    if hyp["low_res"] and (hyp["low_res"] - 1) // 8 < 3:
        raise ValueError("low_res must be at least 25 (28 or 26 recommended)")
    device = context.device
    cuda = device.type == "cuda"
    dtype = torch.float16 if cuda else torch.float32
    torch.backends.cudnn.benchmark = True

    net = Net(hyp, context.num_classes).to(device, dtype, memory_format=torch.channels_last)
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.float()
    train_net = torch.compile(net, mode=hyp["compile"]) if cuda and hyp["compile"] else net
    float_state = [t for t in net.state_dict().values() if t.is_floating_point()]
    state = SimpleNamespace(
        hyp=hyp,
        device=device,
        dtype=dtype,
        net=net,
        train_net=train_net,
        classifier=Classifier(net, dtype).to(device),
        float_state=float_state,
        ema=[t.clone() for t in float_state],
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
        # Every training resolution gets its own compiled graph: warm each one up.
        for resolution in _resolutions(hyp):
            for _ in range(2):
                prepare(state, synthetic, seed=0)
                state.whiten_bias_steps = 3
                _fit(state, total_steps=6, force_resolution=resolution)
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
    net.train()

    raw = data.images.to(device, non_blocking=True).float().div_(255)
    mean = raw.mean(dim=(0, 2, 3), keepdim=True)
    std = raw.std(dim=(0, 2, 3), keepdim=True)
    state.classifier.mean.copy_(mean)
    state.classifier.std.copy_(std)
    images = ((raw - mean) / std).to(state.dtype, memory_format=torch.channels_last)
    del raw
    net.init_whiten(images[:5000])
    torch._foreach_copy_(state.ema, state.float_state)
    # Extra augmentation randomness comes from its own generator, reseeded every trial, so
    # the global stream (init, data order, flips, crops) stays identical to the base recipe.
    state.aug_generator = (
        torch.Generator(device=device).manual_seed(seed) if hyp["jitter"] else None
    )

    # Alternating flip: flip a random half once, then mirror everything on odd epochs.
    images = batch_flip_lr(images)
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
    norm_biases = [p for name, p in net.named_parameters() if "norm" in name and p.requires_grad]
    others = [p for name, p in net.named_parameters() if "norm" not in name and p.requires_grad]
    state.optimizer = torch.optim.SGD(
        [
            dict(params=norm_biases, lr=lr_biases, weight_decay=wd / lr_biases),
            dict(params=others, lr=lr, weight_decay=wd / lr),
        ],
        momentum=momentum,
        nesterov=True,
    )
    for group in state.optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    state.batch_size = batch_size
    state.steps_per_epoch = len(data.labels) // batch_size
    state.total_steps = math.ceil(hyp["epochs"] * state.steps_per_epoch)
    state.whiten_bias_steps = math.ceil(hyp["whiten_bias_epochs"] * state.steps_per_epoch)


def train(state) -> nn.Module:
    _fit(state, state.total_steps)
    return state.classifier


def _resolutions(hyp):
    """Training resolutions in use: the low one first (if any), then the native 32."""
    return ([hyp["low_res"]] if hyp["low_res"] and hyp["low_res_epochs"] else []) + [32]


def _epoch_resolution(hyp, epoch):
    return hyp["low_res"] if hyp["low_res"] and epoch < hyp["low_res_epochs"] else 32


def _fit(state, total_steps, force_resolution=None):
    hyp, net, optimizer = state.hyp, state.net, state.optimizer
    labels, batch_size, steps_per_epoch = state.labels, state.batch_size, state.steps_per_epoch
    warmup_steps = int(total_steps * hyp["warmup"])
    ema_decay = 0.95**5 * (torch.arange(total_steps + 1) / total_steps) ** 3
    step = 0
    net.train()
    for epoch in range(math.ceil(total_steps / steps_per_epoch)):
        if hyp["translate"]:
            crop = batch_crop_gather if hyp["crop_gather"] else batch_crop
            images = crop(state.images, 32)
        else:
            images = state.images
        if epoch % 2 == 1:
            images = images.flip(-1)
        if hyp["cutout"]:
            images = batch_cutout(images, hyp["cutout"])
        if hyp["jitter"]:
            images = batch_jitter(images, hyp["jitter"], state.aug_generator)
        resolution = force_resolution or _epoch_resolution(hyp, epoch)
        if resolution != images.size(-1):
            images = F.interpolate(
                images, size=(resolution, resolution), mode="bilinear", antialias=True
            ).contiguous(memory_format=torch.channels_last)
        order = torch.randperm(len(labels), device=labels.device)
        for i in range(steps_per_epoch):
            if step >= total_steps:
                break
            idx = order[i * batch_size : (i + 1) * batch_size]
            outputs = state.train_net(images[idx], step < state.whiten_bias_steps)
            loss = F.cross_entropy(
                outputs.float(),
                labels[idx],
                label_smoothing=hyp["label_smoothing"],
                reduction="sum",
            )
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
                _lookahead(state, ema_decay[step].item())
    if hyp["ema_every"]:
        _lookahead(state, 1.0)


@torch.no_grad()
def _lookahead(state, decay):
    torch._foreach_lerp_(state.ema, state.float_state, 1 - decay)
    torch._foreach_copy_(state.float_state, state.ema)
