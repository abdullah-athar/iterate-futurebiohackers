"""CIFAR-100 speedrun recipe: an airbench-style network trained from scratch.

Adapted from Keller Jordan's airbench (https://github.com/KellerJordan/cifar10-airbench),
Copyright (c) 2024 Keller Jordan, released under the MIT License. Changes: 100-class
head, 64/256/768 blocks with global max pooling, label smoothing 0.25, a 9.5-epoch
schedule whose first quarter trains on 24x24 crops, per-image brightness/contrast
jitter, a compiled loss and fused SGD, the harness build/prepare/train split, and no
test-time augmentation.

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
    "epochs": 9.5,
    "batch_size": 1024,
    "lr": 12.0,  # per 1024 examples, decoupled from momentum (airbench convention)
    "momentum": 0.85,
    "weight_decay": 0.0168,  # per 1024 examples, decoupled from the learning rate
    "bias_scaler": 16.0,  # learning-rate multiplier for BatchNorm biases
    "label_smoothing": 0.25,
    "warmup": 0.23,  # fraction of steps spent ramping the learning rate up
    "final_lr": 0.07,  # learning-rate multiplier reached at the last step
    "whiten_bias_epochs": 3,
    "translate": 2,
    "cutout": 0,
    "widths": [64, 256, 768],
    "depth": 3,  # convs per group; the third adds a residual connection
    "depths": None,  # optional per-group depths, e.g. [2, 3, 3]
    "train_resolution": 24,  # reduced resolution for the first training stage
    "resolution_switch": 0.25,  # fraction of steps before returning to 32 pixels
    "crop_mode": "masked",  # "indexed" preserves channels-last with one gather
    "fused_sgd": True,
    "compile_loss": True,
    "hard_fraction": 1.0,  # <1 enables a freshly trained small proxy
    "proxy_widths": [32, 64, 128],
    "proxy_every": 4,  # proxy backward/update period; scores every batch
    "proxy_mode": "offline",  # airbench-style prepass, or online selection
    "gelu_approximate": "none",  # "tanh" uses a cheaper approximation
    "autotune_backends": "ATEN,TRITON",  # ATen/cuDNN and Inductor Triton candidates
    "pool_first": [False, False, False],  # move selected group pools before conv1
    "depth2_residual": False,  # depth-2 groups keep a skip connection over their second conv
    "stem": "conv",  # how the frozen whitening feeds group 1; the alternatives are in STEMS
    "scaling_factor": 1.25 / 9,  # logit scale (airbench uses 1/9)
    "bn_momentum": 0.7,
    "ema_every": 5,  # lookahead EMA period in steps; 0 disables it
    "compile": "max-autotune",  # torch.compile mode; "" runs eagerly
    # Accuracy-recovery and resizing switches.
    "jitter": 0.3,  # per-image brightness/contrast jitter strength (own RNG, seeded per trial)
    "low_res_compile": "default",  # torch.compile mode of the static low-res graphs; "" = eager
    "count_nonfinite": False,  # count non-finite step losses on the GPU; train() prints the total
    # Staged resizing: [[resolution, until_fraction], ...] with increasing fractions, e.g.
    # [[20, 0.2], [24, 0.4], [28, 0.6]]; steps past the last stage run at 32. When set, it
    # replaces the train_resolution/resolution_switch pair above.
    "res_schedule": None,
    # Hard-example filtering: from epoch filter_start on, every epoch trains only on the
    # filter_keep fraction of the training set with the highest last training loss.
    "filter_start": 0,  # 0 disables filtering
    "filter_keep": 0.75,
}

# Stems: cheaper ways to reach many channels early than group 1's trainable 3x3 conv at 31x31.
# Each entry is (space-to-depth factor applied to the input, whitening kernel, whitening
# stride, group 1's first conv kernel or None to drop that conv, whether group 1 keeps its 2x2
# pool). The frozen whitening conv maps every c*k*k patch to 2*c*k*k channels (the covariance
# eigenvectors and their negatives), so the group widths downstream are unchanged.
STEMS = {
    "conv": (1, 2, 1, 3, True),  # airbench: 2x2 whitening, 3x3 conv at 31x31, pool
    "space_to_depth": (2, 2, 1, 3, True),  # pixel_unshuffle(2): 16x16x12 -> 15x15x96, then as conv
    "space_to_depth_nopool": (2, 2, 1, 3, False),  # as above, but group 1 keeps its 15x15 maps
    "whiten4s2": (1, 4, 2, None, False),  # 4x4 stride-2 whitening to 15x15x96 straight into conv2
    "whiten3s2": (1, 3, 2, 1, False),  # 3x3 stride-2 whitening to 15x15x54, then a 1x1 conv
    "conv1x1": (1, 2, 1, 1, True),  # group 1's first conv is 1x1 (24 -> widths[0]), then pool
}


def _stem_channels(stem):
    """Output channels of the whitening conv: twice the dimension of the patches it sees."""
    unshuffle, kernel = STEMS[stem][:2]
    return 2 * 3 * unshuffle**2 * kernel**2


def _stem_sizes(stem, resolution):
    """Spatial sizes after the stem and after each group for one input resolution. Raises
    when a 2x2 pool would see a map smaller than 2x2, i.e. before any map reaches 0x0."""
    unshuffle, kernel, stride, _, first_pool = STEMS[stem]
    if resolution % unshuffle:
        raise ValueError(f"stem {stem} needs even input sizes, not {resolution} px")
    size = (resolution // unshuffle - kernel) // stride + 1
    sizes = [size]
    for pool in (first_pool, True, True):
        if pool:
            if size < 2:
                raise ValueError(
                    f"stem {stem}: a {size}x{size} map meets a 2x2 pool at {resolution} px"
                )
            size //= 2
        sizes.append(size)
    return sizes


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
        return F.gelu(x, approximate=approximate)


class Conv(nn.Conv2d):
    def __init__(self, channels_in, channels_out, kernel_size=3):
        super().__init__(
            channels_in, channels_out, kernel_size=kernel_size, padding="same", bias=False
        )

    def reset_parameters(self):
        super().reset_parameters()
        # Same values as nn.init.dirac_(w[: w.size(1)]) (identity kernel on the first `in`
        # output channels) with one fill and one index_put instead of a kernel launch per
        # channel: that loop was most of prepare's 63 ms reset time.
        w = self.weight.data
        first = w[: w.size(1)]
        first.zero_()
        n = min(first.size(0), first.size(1))
        idx = torch.arange(n, device=w.device)
        first[idx, idx, w.size(2) // 2, w.size(3) // 2] = 1


class ConvGroup(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_out,
        depth,
        bn_momentum,
        gelu_approximate,
        pool_first,
        depth2_residual=False,
        first_kernel=3,
        first_pool=True,
    ):
        super().__init__()
        if first_kernel is None:
            # The stem already delivers channels_out channels at this group's resolution: the
            # group starts at conv2 and its residual is the group input (stem whiten4s2).
            if channels_in != channels_out:
                raise ValueError("a group without a first conv needs channels_in == channels_out")
            self.conv1 = self.pool = self.norm1 = None
        else:
            self.conv1 = Conv(channels_in, channels_out, first_kernel)
            self.pool = nn.MaxPool2d(2) if first_pool else nn.Identity()
            self.norm1 = BatchNorm(channels_out, bn_momentum)
        self.pool_first = pool_first
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out, bn_momentum)
        self.conv3 = Conv(channels_out, channels_out) if depth == 3 else None
        self.norm3 = BatchNorm(channels_out, bn_momentum) if depth == 3 else None
        self.approximate = gelu_approximate
        self.depth2_residual = depth2_residual  # depth 2: skip connection over conv2

    def forward(self, x):
        if self.conv1 is not None:
            x = self.conv1(self.pool(x)) if self.pool_first else self.pool(self.conv1(x))
            x = self.norm1.activate(x, self.approximate)
        if self.conv3 is None:
            residual = x if self.depth2_residual else None
            return self.norm2.activate(self.conv2(x), self.approximate, residual)
        x0 = x
        x = self.norm2.activate(self.conv2(x), self.approximate)
        return self.norm3.activate(self.conv3(x), self.approximate, x0)


class Net(nn.Module):
    def __init__(self, hyp, num_classes):
        super().__init__()
        w1, w2, w3 = hyp["widths"]
        depths = hyp["depths"] or [hyp["depth"]] * 3
        bn_momentum = hyp["bn_momentum"]
        unshuffle, kernel, stride, first_kernel, first_pool = STEMS[hyp["stem"]]
        self.unshuffle = unshuffle  # space-to-depth factor applied before whitening; 1 = none
        channels = _stem_channels(hyp["stem"])
        self.whiten = nn.Conv2d(
            3 * unshuffle**2, channels, kernel_size=kernel, stride=stride, padding=0, bias=True
        )
        self.whiten.weight.requires_grad = False
        skip = hyp["depth2_residual"]
        self.layers = nn.Sequential(
            nn.GELU(approximate=hyp["gelu_approximate"]),
            ConvGroup(
                channels,
                w1,
                depths[0],
                bn_momentum,
                hyp["gelu_approximate"],
                hyp["pool_first"][0],
                skip,
                first_kernel=first_kernel,
                first_pool=first_pool,
            ),
            ConvGroup(
                w1, w2, depths[1], bn_momentum, hyp["gelu_approximate"], hyp["pool_first"][1], skip
            ),
            ConvGroup(
                w2, w3, depths[2], bn_momentum, hyp["gelu_approximate"], hyp["pool_first"][2], skip
            ),
            nn.AdaptiveMaxPool2d(1),
        )
        self.head = nn.Linear(w3, num_classes, bias=False)
        self.scaling_factor = hyp["scaling_factor"]

    def reset(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d | nn.BatchNorm2d | nn.Linear):
                m.reset_parameters()
        self.whiten.bias.data.zero_()

    def space_to_depth(self, x):
        """The whitening conv's input: pixel_unshuffle folds each 2x2 block into the channels."""
        return F.pixel_unshuffle(x, self.unshuffle) if self.unshuffle > 1 else x

    @torch.no_grad()
    def init_whiten(self, images, eps=5e-4):
        """Whitening filters from the covariance of the patches the conv sees at its own stride;
        `images` are space_to_depth(training images)."""
        c, (h, w) = images.shape[1], self.whiten.weight.shape[2:]
        sh, sw = self.whiten.stride
        patches = images.unfold(2, h, sh).unfold(3, w, sw).transpose(1, 3).reshape(-1, c, h, w)
        flat = patches.float().view(len(patches), -1)
        covariance = flat.T @ flat / len(flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance, UPLO="U")
        scaled = eigenvectors.T.reshape(-1, c, h, w) / torch.sqrt(
            eigenvalues.view(-1, 1, 1, 1) + eps
        )
        self.whiten.weight.copy_(torch.cat((scaled, -scaled)))

    def stem(self, x, whiten_bias_grad: bool = True):
        """Frozen whitening (after space-to-depth when the stem uses it); only the bias trains,
        and only while whiten_bias_grad is set."""
        b = self.whiten.bias
        bias = b if whiten_bias_grad else b.detach()
        return F.conv2d(self.space_to_depth(x), self.whiten.weight, bias, self.whiten.stride)

    def forward(self, x, whiten_bias_grad: bool = True):
        x = self.layers(self.stem(x, whiten_bias_grad)).flatten(1)
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
#                 Schedules                 #
#############################################


def _resolutions(hyp):
    """Sorted training resolutions, 32 included: the res_schedule stages, or the
    train_resolution/resolution_switch pair when res_schedule is null."""
    if hyp["res_schedule"] is None:
        return sorted({hyp["train_resolution"], 32})
    return sorted({resolution for resolution, _ in hyp["res_schedule"]} | {32})


def _validate_schedule(schedule):
    if schedule is None:
        return
    pairs = [stage for stage in schedule if isinstance(stage, list | tuple) and len(stage) == 2]
    fractions = [fraction for _, fraction in pairs]
    if (
        not pairs
        or len(pairs) != len(schedule)
        or any(resolution not in (16, 20, 24, 28) for resolution, _ in pairs)
        or any(not 0 < fraction < 1 for fraction in fractions)
        or fractions != sorted(set(fractions))
    ):
        raise ValueError(
            "res_schedule must list [resolution, until_fraction] pairs with resolutions in"
            " {16, 20, 24, 28} and strictly increasing fractions in (0, 1)"
        )


def _stages(hyp, total_steps):
    """(resolution, first step of the next stage) pairs; steps past the last pair run at 32."""
    if hyp["res_schedule"] is None:
        return [(hyp["train_resolution"], int(total_steps * hyp["resolution_switch"]))]
    return [
        (resolution, int(total_steps * fraction)) for resolution, fraction in hyp["res_schedule"]
    ]


def _step_resolution(stages, step):
    for resolution, until in stages:
        if step < until:
            return resolution
    return 32


def _epoch_steps(hyp, epoch, count, batch_size):
    """Optimizer steps in one epoch; filtered epochs see round(filter_keep * count) examples."""
    if hyp["filter_start"] and epoch >= hyp["filter_start"]:
        count = round(hyp["filter_keep"] * count)
    return count // batch_size


def _steps_until(hyp, epochs, count, batch_size):
    """Steps in the first `epochs` epochs, a fractional last epoch rounded up. Without
    filtering this is the base recipe's math.ceil(epochs * steps_per_epoch)."""
    if not hyp["filter_start"]:
        return math.ceil(epochs * (count // batch_size))
    full = math.floor(epochs)
    steps = sum(_epoch_steps(hyp, epoch, count, batch_size) for epoch in range(full))
    return steps + math.ceil((epochs - full) * _epoch_steps(hyp, full, count, batch_size))


#############################################
#                 Interface                 #
#############################################


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
    if hyp["crop_mode"] not in ("masked", "indexed", "triton"):
        raise ValueError("crop_mode must be masked, indexed, or triton")
    if hyp["epochs"] <= 0 or hyp["batch_size"] <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if not 0 < hyp["hard_fraction"] <= 1 or hyp["proxy_every"] < 1:
        raise ValueError("hard_fraction must be in (0, 1]; proxy_every must be positive")
    if hyp["proxy_mode"] not in ("online", "offline"):
        raise ValueError("proxy_mode must be online or offline")
    if hyp["gelu_approximate"] not in ("none", "tanh"):
        raise ValueError("gelu_approximate must be none or tanh")
    if hyp["autotune_backends"] not in ("ATEN", "TRITON", "ATEN,TRITON"):
        raise ValueError("autotune_backends must be ATEN, TRITON, or ATEN,TRITON")
    if type(hyp["depth2_residual"]) is not bool:
        raise ValueError("depth2_residual must be a boolean")
    if len(hyp["pool_first"]) != 3 or any(type(v) is not bool for v in hyp["pool_first"]):
        raise ValueError("pool_first must contain three booleans")
    if hyp["jitter"] < 0:
        raise ValueError("jitter must be non-negative")
    _validate_schedule(hyp["res_schedule"])
    stem = hyp["stem"]
    if stem not in STEMS:
        raise ValueError(f"stem must be one of {sorted(STEMS)}")
    stem_widths = [hyp["widths"]] + ([hyp["proxy_widths"]] if hyp["hard_fraction"] < 1 else [])
    if STEMS[stem][3] is None and any(w[0] != _stem_channels(stem) for w in stem_widths):
        raise ValueError(
            f"stem {stem} starts group 1 at its {_stem_channels(stem)} whitened channels:"
            f" widths[0] (and proxy_widths[0] with a proxy) must be {_stem_channels(stem)}"
        )
    if not STEMS[stem][4] and hyp["pool_first"][0]:
        raise ValueError(f"stem {stem} removes group 1's pool: pool_first[0] must be false")
    for resolution in _resolutions(hyp):
        _stem_sizes(stem, resolution)  # every resolution must keep a map >= 2x2 at each pool
    if type(hyp["filter_start"]) is not int or hyp["filter_start"] < 0:
        raise ValueError("filter_start must be a non-negative epoch index")
    if not 0 < hyp["filter_keep"] <= 1:
        raise ValueError("filter_keep must be in (0, 1]")
    if hyp["filter_start"] and (hyp["hard_fraction"] < 1 or hyp["compile_loss"]):
        raise ValueError("filter_start cannot be combined with hard_fraction < 1 or compile_loss")
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
        if isinstance(m, nn.BatchNorm2d):
            m.float()
    resolutions = _resolutions(hyp)
    if cuda and hyp["compile"] and resolutions[0] < 32:
        # Progressive resizing: one static graph per resolution over the same eager net.
        # dynamic=False keeps dynamo from switching to dynamic shapes on the second size, and
        # the low-res graphs compile in the cheaper low_res_compile mode (a second
        # max-autotune graph added about 230 s to the cold build); "" runs them eagerly.
        # Every resolution adds two entries (whitening bias trained / frozen) to the dynamo
        # cache of Net.forward, whose default limit of 8 holds at most four resolutions.
        import torch._dynamo.config as dynamo_config

        dynamo_config.cache_size_limit = max(dynamo_config.cache_size_limit, 2 * len(resolutions))
        nets = {32: torch.compile(net, mode=hyp["compile"], dynamic=False)}
        for resolution in resolutions[:-1]:
            nets[resolution] = (
                torch.compile(net, mode=hyp["low_res_compile"], dynamic=False)
                if hyp["low_res_compile"]
                else net
            )
    elif cuda and hyp["compile"]:
        nets = {32: torch.compile(net, mode=hyp["compile"])}
    else:
        nets = dict.fromkeys(resolutions, net)

    def loss_fn(outputs, labels):
        return F.cross_entropy(
            outputs.float(), labels, label_smoothing=hyp["label_smoothing"], reduction="sum"
        )

    def example_loss_fn(outputs, labels):
        return F.cross_entropy(
            outputs.float(), labels, label_smoothing=hyp["label_smoothing"], reduction="none"
        )

    loss_fn = torch.compile(loss_fn) if cuda and hyp["compile_loss"] else loss_fn
    float_state = [t for t in net.state_dict().values() if t.is_floating_point()]
    state = SimpleNamespace(
        hyp=hyp,
        device=device,
        dtype=dtype,
        net=net,
        train_net=nets[32],
        low_net=nets[resolutions[0]],
        nets=nets,
        loss_fn=loss_fn,
        example_loss_fn=example_loss_fn,
        classifier=Classifier(net, dtype).to(device),
        float_state=float_state,
        ema=[t.clone() for t in float_state],
        proxy=None,
        crop_kernel=crop_kernel,
        aug_generator=None,
        nonfinite=None,
        scores=None,
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
        for resolution in resolutions:
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
    net.init_whiten(net.space_to_depth(images[:5000]))
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
    # One downscaled, reflect-padded copy of the flipped set per reduced training resolution.
    state.small_images = {}
    for resolution in _resolutions(hyp):
        if resolution < 32:
            small = F.interpolate(
                images, size=(resolution,) * 2, mode="bilinear", align_corners=False
            )
            if hyp["translate"]:
                small = F.pad(small, (hyp["translate"],) * 4, "reflect")
            state.small_images[resolution] = small.to(memory_format=torch.channels_last)
    if hyp["translate"]:
        images = F.pad(images, (hyp["translate"],) * 4, "reflect")
    state.images = images
    state.labels = data.labels.to(device, non_blocking=True)
    # Last training loss of every example (+inf until seen): the hard-example filter's key.
    state.scores = (
        torch.full((len(data.labels),), math.inf, device=device) if hyp["filter_start"] else None
    )

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
    state.total_steps = _steps_until(hyp, hyp["epochs"], len(data.labels), batch_size)
    state.whiten_bias_steps = _steps_until(
        hyp, hyp["whiten_bias_epochs"], len(data.labels), batch_size
    )


def _make_optimizer(net, lr, lr_biases, wd, hyp, device):
    norm_biases = [p for name, p in net.named_parameters() if "norm" in name and p.requires_grad]
    others = [p for name, p in net.named_parameters() if "norm" not in name and p.requires_grad]
    optimizer = torch.optim.SGD(
        [
            dict(params=norm_biases, lr=lr_biases, weight_decay=wd / lr_biases),
            dict(params=others, lr=lr, weight_decay=wd / lr),
        ],
        momentum=hyp["momentum"],
        nesterov=True,
        fused=hyp["fused_sgd"] and device.type == "cuda",
    )
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
    labels, batch_size = state.labels, state.batch_size
    warmup_steps = int(total_steps * hyp["warmup"])
    ema_decay = (0.95**5 * (torch.arange(total_steps + 1) / total_steps) ** 3).tolist()
    nonfinite = (
        torch.zeros((), dtype=torch.int64, device=labels.device)
        if hyp["count_nonfinite"] and not proxy_only
        else None
    )
    stages = _stages(hyp, total_steps)
    filtering = bool(hyp["filter_start"])
    step = 0
    epoch = 0
    net.train()
    while step < total_steps:
        images = None
        current_resolution = None
        order = torch.randperm(len(labels), device=labels.device)
        if filtering and epoch >= hyp["filter_start"]:
            # Keep the examples with the highest last training loss. The permutation is drawn
            # exactly as in the unfiltered recipe, so the global RNG stream stays paired with it.
            hardest = state.scores.topk(round(hyp["filter_keep"] * len(labels)), sorted=False)
            keep = torch.zeros_like(order, dtype=torch.bool)
            keep[hardest.indices] = True
            order = order[keep[order]]
        for i in range(len(order) // batch_size):
            if step >= total_steps:
                break
            resolution = getattr(state, "warmup_resolution", None)
            if resolution is None:
                resolution = _step_resolution(stages, step)
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
                train_net = state.nets[resolution]
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
            if filtering:
                losses = state.example_loss_fn(outputs, targets)
                state.scores[idx] = losses.detach()
                loss = losses.sum()
            else:
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
        epoch += 1
    if hyp["ema_every"] and not proxy_only:
        _lookahead(state, 1.0)
    if not proxy_only:
        state.nonfinite = nonfinite
        state.steps_taken = step


@torch.no_grad()
def _lookahead(state, decay):
    torch._foreach_lerp_(state.ema, state.float_state, 1 - decay)
    torch._foreach_copy_(state.float_state, state.ema)
