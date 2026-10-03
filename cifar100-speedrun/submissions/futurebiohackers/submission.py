"""CIFAR-100 speedrun recipe: an airbench-style network trained from scratch.

Adapted from Keller Jordan's airbench (https://github.com/KellerJordan/cifar10-airbench),
Copyright (c) 2024 Keller Jordan, released under the MIT License. Changes: 100-class
head, the harness build/prepare/train split, no test-time augmentation, and tunable
widths, depth, and schedule.

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

# Override any value with --params, e.g. '{"epochs": 14, "widths": [128, 384, 384]}'.
DEFAULTS = {
    "epochs": 12.0,
    "batch_size": 1024,
    "lr": 11.5,  # per 1024 examples, decoupled from momentum (airbench convention)
    "momentum": 0.85,
    "weight_decay": 0.0153,  # per 1024 examples, decoupled from the learning rate
    "bias_scaler": 64.0,  # learning-rate multiplier for BatchNorm biases
    "label_smoothing": 0.2,
    "warmup": 0.23,  # fraction of steps spent ramping the learning rate up
    "final_lr": 0.07,  # learning-rate multiplier reached at the last step
    "whiten_bias_epochs": 3,
    "translate": 2,
    "cutout": 0,
    "widths": [64, 256, 256],
    "depth": 2,
    "scaling_factor": 1 / 9,
    "bn_momentum": 0.6,
    "ema_every": 5,  # lookahead EMA period in steps; 0 disables it
    "compile": "max-autotune",  # torch.compile mode; "" runs eagerly
    # "muon" follows airbench94_muon: Muon on conv filters, SGD on biases and the head.
    "optimizer": "sgd",
    "muon_lr": 0.24,
    "muon_momentum": 0.6,
    "bias_lr": 0.053,
    "head_lr": 0.67,
    "muon_wd": 2e-6,  # per example
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


class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out, depth, bn_momentum):
        super().__init__()
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out, bn_momentum)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out, bn_momentum)
        self.conv3 = Conv(channels_out, channels_out) if depth == 3 else None
        self.norm3 = BatchNorm(channels_out, bn_momentum) if depth == 3 else None
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.activ(self.norm1(self.pool(self.conv1(x))))
        if self.conv3 is None:
            return self.activ(self.norm2(self.conv2(x)))
        x0 = x
        x = self.activ(self.norm2(self.conv2(x)))
        return self.activ(self.norm3(self.conv3(x)) + x0)


class Net(nn.Module):
    def __init__(self, hyp, num_classes):
        super().__init__()
        w1, w2, w3 = hyp["widths"]
        depth, bn_momentum = hyp["depth"], hyp["bn_momentum"]
        self.whiten = nn.Conv2d(3, 24, kernel_size=2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(24, w1, depth, bn_momentum),
            ConvGroup(w1, w2, depth, bn_momentum),
            ConvGroup(w2, w3, depth, bn_momentum),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(w3, num_classes, bias=False)
        self.scaling_factor = hyp["scaling_factor"]
        self.muon_head = hyp["optimizer"] == "muon"

    def reset(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d | nn.BatchNorm2d | nn.Linear):
                m.reset_parameters()
        self.whiten.bias.data.zero_()
        if self.muon_head:
            self.head.weight.data /= self.head.weight.data.std()

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
        if self.muon_head:
            return self.head(x) / x.size(-1)
        return self.head(x) * self.scaling_factor


#############################################
#                   Muon                    #
#############################################


def zeropower_via_newtonschulz5(G, steps: int = 3, eps: float = 1e-7):
    """Approximately orthogonalize G with a quintic Newton-Schulz iteration."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Nesterov momentum, then an orthogonalized update on normalized conv filters."""

    def __init__(self, params, lr, momentum, zeropower):
        super().__init__(params, dict(lr=lr, momentum=momentum))
        self.zeropower = zeropower

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(p.grad)
                g = p.grad.add(buf, alpha=group["momentum"])
                p.mul_(len(p) ** 0.5 / p.norm())
                update = self.zeropower(g.reshape(len(g), -1)).view(g.shape)
                p.add_(update, alpha=-group["lr"])


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


#############################################
#                 Interface                 #
#############################################


def build(context: BuildContext):
    hyp = {**DEFAULTS, **context.parameters}
    unknown = set(hyp) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown parameters: {sorted(unknown)}")
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
        zeropower=torch.compile(zeropower_via_newtonschulz5)
        if cuda
        else zeropower_via_newtonschulz5,
    )

    # Untimed warmup on random synthetic images: compiles both whitening-bias graphs,
    # autotunes cuDNN, initializes cuBLAS/cuSOLVER, and warms evaluation shapes.
    if cuda:
        count = 50_000
        synthetic = TrainingData(
            torch.randint(0, 256, (count, 3, 32, 32), dtype=torch.uint8),
            torch.randint(0, context.num_classes, (count,)),
        )
        for _ in range(2):
            prepare(state, synthetic, seed=0)
            state.whiten_bias_steps = 3
            _fit(state, total_steps=6)
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

    # Alternating flip: flip a random half once, then mirror everything on odd epochs.
    images = batch_flip_lr(images)
    if hyp["translate"]:
        images = F.pad(images, (hyp["translate"],) * 4, "reflect")
    state.images = images
    state.labels = data.labels.to(device, non_blocking=True)

    batch_size = min(hyp["batch_size"], len(data.labels))
    momentum = hyp["momentum"]
    norm_biases = [p for name, p in net.named_parameters() if "norm" in name and p.requires_grad]
    if hyp["optimizer"] == "muon":
        wd = hyp["muon_wd"] * batch_size
        bias_lr, head_lr = hyp["bias_lr"], hyp["head_lr"]
        filters = [p for p in net.parameters() if p.ndim == 4 and p.requires_grad]
        groups = [
            dict(params=[net.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr, whiten=True),
            dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr),
            dict(params=[net.head.weight], lr=head_lr, weight_decay=wd / head_lr),
        ]
        state.optimizers = [
            torch.optim.SGD(groups, momentum=momentum, nesterov=True),
            Muon(filters, hyp["muon_lr"], hyp["muon_momentum"], state.zeropower),
        ]
    else:
        kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
        lr = hyp["lr"] / kilostep_scale
        wd = hyp["weight_decay"] * batch_size / kilostep_scale
        lr_biases = lr * hyp["bias_scaler"]
        others = [p for name, p in net.named_parameters() if "norm" not in name and p.requires_grad]
        groups = [
            dict(params=norm_biases, lr=lr_biases, weight_decay=wd / lr_biases),
            dict(params=others, lr=lr, weight_decay=wd / lr),
        ]
        state.optimizers = [torch.optim.SGD(groups, momentum=momentum, nesterov=True)]
    for optimizer in state.optimizers:
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]

    state.batch_size = batch_size
    state.steps_per_epoch = len(data.labels) // batch_size
    state.total_steps = math.ceil(hyp["epochs"] * state.steps_per_epoch)
    state.whiten_bias_steps = math.ceil(hyp["whiten_bias_epochs"] * state.steps_per_epoch)


def train(state) -> nn.Module:
    _fit(state, state.total_steps)
    return state.classifier


def _fit(state, total_steps):
    hyp, net, optimizers = state.hyp, state.net, state.optimizers
    labels, batch_size, steps_per_epoch = state.labels, state.batch_size, state.steps_per_epoch
    warmup_steps = int(total_steps * hyp["warmup"])
    ema_decay = 0.95**5 * (torch.arange(total_steps + 1) / total_steps) ** 3
    crop = 32
    step = 0
    net.train()
    for epoch in range(math.ceil(total_steps / steps_per_epoch)):
        images = batch_crop(state.images, crop) if hyp["translate"] else state.images
        if epoch % 2 == 1:
            images = images.flip(-1)
        if hyp["cutout"]:
            images = batch_cutout(images, hyp["cutout"])
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
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if step < warmup_steps:
                frac = step / warmup_steps
                scale = 0.2 * (1 - frac) + frac
            else:
                frac = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                scale = (1 - frac) + hyp["final_lr"] * frac
            whiten_scale = max(0.0, 1 - step / max(1, state.whiten_bias_steps))
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * (
                        whiten_scale if "whiten" in group else scale
                    )
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
