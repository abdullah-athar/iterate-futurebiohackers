"""futurebiohackers: ResNet9-style CIFAR-100 baseline (40 epochs, width 64).

Modelled on the organizers' verified reference (repository README: ResNet9-style,
40 epochs, width 64, 75.36% mean accuracy, 59.30 s prepare + train on an A100 80GB
PCIe). Plain PyTorch 2.4, nothing beyond torch and benchmark.api is imported.

Harness contract (submission_template/README.md):
- build(context)        once, untimed: model structure, settings, synthetic warmup only.
- prepare(state, data, seed)  per trial, timed: reset everything, data to the device.
- train(state)          per trial, timed: the training loop; returns the eager nn.Module.

The defaults below ARE the baseline. Override any key for experiments with
--params '{"epochs": 10, "width": 32}'; the official run uses no --params.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from benchmark.api import BuildContext, TrainingData

from .model import ResNet9

DEFAULTS: dict = {
    "epochs": 40,
    "width": 64,
    "batch_size": 512,
    "lr": 0.4,  # peak learning rate for the mean-reduced loss
    "momentum": 0.9,  # SGD with Nesterov momentum
    "weight_decay": 5e-4,  # on every parameter, as in the DAWNBench ResNet9 recipe
    "label_smoothing": 0.1,
    "warmup_fraction": 0.15,  # linear warmup to lr, then linear decay to 0 (one cycle)
    # auto: bf16 autocast on Ampere or newer CUDA GPUs, fp16 autocast + GradScaler on
    # older CUDA GPUs (free Colab/Kaggle T4, P100), fp32 on CPU. Also bf16/fp16/fp32.
    "precision": "auto",
    "bn_weight_decay": True,  # False: no weight decay on BatchNorm scale/shift (1-D params)
    "use_compile": False,  # torch.compile the training forward/backward (compiled in build)
    "compile_mode": "default",  # default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs
}
COMPILE_MODES = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")
PAD = 4  # random-crop padding in pixels
WARMUP_STEPS = 3  # synthetic forward/backward/optimizer steps in build (autotuning only)
TEST_IMAGES = 10_000  # size of the CIFAR-100 test split; only used to pick eval warmup shapes
AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}


# ------------------------------------------------------------------ configuration


def _config(parameters: dict) -> SimpleNamespace:
    unknown = sorted(set(parameters) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown recipe parameters {unknown}; known: {sorted(DEFAULTS)}")
    cfg = SimpleNamespace(**{**DEFAULTS, **parameters})
    if not all(
        isinstance(getattr(cfg, k), int) and getattr(cfg, k) >= 1
        for k in ("epochs", "width", "batch_size")
    ):
        raise ValueError("epochs, width and batch_size must be integers >= 1")
    if cfg.lr <= 0 or not 0 <= cfg.warmup_fraction < 1:
        raise ValueError("lr must be positive and warmup_fraction in [0, 1)")
    if cfg.precision not in ("auto", *AMP_DTYPES):
        raise ValueError(f"precision must be one of auto, {', '.join(AMP_DTYPES)}")
    if cfg.compile_mode not in COMPILE_MODES:
        raise ValueError(f"compile_mode must be one of {', '.join(COMPILE_MODES)}")
    if not isinstance(cfg.use_compile, bool) or not isinstance(cfg.bn_weight_decay, bool):
        raise ValueError("use_compile and bn_weight_decay must be JSON booleans")
    if cfg.compile_mode != "default" and not cfg.use_compile:
        raise ValueError("compile_mode has no effect without use_compile: true")
    return cfg


def _precision(cfg: SimpleNamespace, device: torch.device) -> str:
    if device.type != "cuda":
        return "fp32"
    if cfg.precision != "auto":
        return cfg.precision
    return "bf16" if torch.cuda.get_device_capability(device)[0] >= 8 else "fp16"


def _lr_at(step: int, total_steps: int, cfg: SimpleNamespace) -> float:
    """Linear warmup over the first warmup_fraction of steps, then linear decay to 0."""
    warmup = max(1, round(cfg.warmup_fraction * total_steps))
    if step < warmup:
        return cfg.lr * (step + 1) / warmup
    return cfg.lr * max(0.0, (total_steps - step) / max(1, total_steps - warmup))


# ------------------------------------------------------------------ trial-specific objects


def _make_optimizer(state) -> torch.optim.SGD:
    cfg = state.cfg
    params = list(state.model.parameters())
    if cfg.bn_weight_decay:
        groups = [{"params": params}]
    else:  # BatchNorm scale/shift are the only 1-D parameters (convs and the linear have no bias)
        groups = [
            {"params": [p for p in params if p.ndim > 1]},
            {"params": [p for p in params if p.ndim <= 1], "weight_decay": 0.0},
        ]
    return torch.optim.SGD(
        groups,
        lr=0.0,  # set per step by _lr_at
        momentum=cfg.momentum,
        nesterov=cfg.momentum > 0,
        weight_decay=cfg.weight_decay,
    )


def _make_scaler(state):
    return torch.amp.GradScaler("cuda") if state.precision == "fp16" else None


def _channel_stats(images_u8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean and std of the given images on the [0, 1] scale (chunked, any device)."""
    n, c, h, w = images_u8.shape
    total = torch.zeros(c, dtype=torch.float64, device=images_u8.device)
    total_sq = torch.zeros_like(total)
    for chunk in images_u8.split(8192):
        x = chunk.to(torch.float32).div_(255)
        total += x.sum(dim=(0, 2, 3), dtype=torch.float64)
        total_sq += x.square().sum(dim=(0, 2, 3), dtype=torch.float64)
    count = n * h * w
    mean = total / count
    std = (total_sq / count - mean.square()).clamp_min_(1e-12).sqrt()
    return mean.float(), std.float()


# ------------------------------------------------------------------ data pipeline


def _augment(xb: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Random 32x32 crop from the PAD-pixel zero-padded image + random horizontal flip.

    Runs on the device, one gather per batch. xb: uint8 [B, 3, H, W]. Returns a
    uint8 [B, 3, H, W] view with channels_last strides.
    """
    b, _, h, w = xb.shape
    dev = xb.device
    padded = F.pad(xb, (PAD, PAD, PAD, PAD))
    dy = torch.randint(0, 2 * PAD + 1, (b,), device=dev, generator=generator)
    dx = torch.randint(0, 2 * PAD + 1, (b,), device=dev, generator=generator)
    flip = torch.rand(b, device=dev, generator=generator) < 0.5
    rows = dy[:, None] + torch.arange(h, device=dev)  # [B, H]
    cols = dx[:, None] + torch.arange(w, device=dev)  # [B, W]
    cols = torch.where(flip[:, None], cols.flip(1), cols)  # reversed columns = flipped image
    batch = torch.arange(b, device=dev)[:, None, None]
    out = padded[batch, :, rows[:, :, None], cols[:, None, :]]  # [B, H, W, C]
    return out.permute(0, 3, 1, 2)


def _train_step(state, xb_u8: torch.Tensor, yb: torch.Tensor, lr: float) -> None:
    for group in state.optimizer.param_groups:
        group["lr"] = lr
    x = xb_u8.to(torch.float32).div_(255).contiguous(memory_format=state.memory_format)
    autocast = (
        torch.autocast(device_type="cuda", dtype=state.amp_dtype)
        if state.amp_dtype is not None
        else contextlib.nullcontext()
    )
    with autocast:
        logits = state.train_model(x)
        loss = F.cross_entropy(logits, yb, label_smoothing=state.cfg.label_smoothing)
    state.optimizer.zero_grad(set_to_none=True)
    if state.scaler is None:
        loss.backward()
        state.optimizer.step()
    else:
        state.scaler.scale(loss).backward()
        state.scaler.step(state.optimizer)
        state.scaler.update()


# ------------------------------------------------------------------ harness entry points


def build(context: BuildContext):
    """Model structure, settings and synthetic warmup. Untimed. No real data, no trial seed."""
    cfg = _config(context.parameters)
    device = context.device
    cuda = device.type == "cuda"
    if cuda:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")  # TF32 for fp32 matmuls (eval forward)
    model = ResNet9(context.num_classes, cfg.width, channels_last=cuda).to(device)
    if cuda:
        model = model.to(memory_format=torch.channels_last)
    precision = _precision(cfg, device)
    state = SimpleNamespace(
        cfg=cfg,
        device=device,
        num_classes=context.num_classes,
        eval_batch_size=context.eval_batch_size,
        model=model,  # the eager module that train() returns
        train_model=torch.compile(model, mode=cfg.compile_mode) if cfg.use_compile else model,
        precision=precision,
        amp_dtype=AMP_DTYPES[precision],
        memory_format=torch.channels_last if cuda else torch.contiguous_format,
        # Per-trial objects, (re)created in prepare():
        optimizer=None,
        scaler=None,
        generator=None,
        images=None,
        labels=None,
    )
    print(
        f"[futurebiohackers] device={device} precision={precision} config={vars(cfg)}", flush=True
    )
    _synthetic_warmup(state)
    return state


def _synthetic_warmup(state) -> None:
    """Allocate memory and autotune kernels on random inputs. prepare() resets all of this."""
    device, cfg = state.device, state.cfg
    generator = torch.Generator(device=device).manual_seed(0)  # fixed; not a trial seed
    state.optimizer = _make_optimizer(state)
    state.scaler = _make_scaler(state)
    state.train_model.train()
    for _ in range(WARMUP_STEPS):
        xb = torch.randint(
            0,
            256,
            (cfg.batch_size, 3, 32, 32),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )
        yb = torch.randint(
            0, state.num_classes, (cfg.batch_size,), device=device, generator=generator
        )
        _train_step(state, _augment(xb, generator), yb, cfg.lr)
    # The evaluator sends fp32 [0, 1] batches of eval_batch_size, plus one smaller final batch.
    state.model.eval()
    with torch.inference_mode():
        for n in {
            state.eval_batch_size,
            TEST_IMAGES % state.eval_batch_size or state.eval_batch_size,
        }:
            state.model(torch.rand(n, 3, 32, 32, device=device, generator=generator))
    state.model.train()


def prepare(state, data: TrainingData, seed: int) -> None:
    """Start a trial from scratch. Timed. Nothing from the warmup or an earlier trial survives."""
    model, device = state.model, state.device
    # 1. Fresh weights and BatchNorm statistics. The harness seeded torch with the trial
    #    seed just before this call, so PyTorch's default initializers are seed-determined.
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()  # BatchNorm: also running_mean/var, num_batches_tracked
    model.zero_grad(set_to_none=True)
    model.train()
    # 2. Fresh optimizer (empty momentum buffers), loss scaler and data-order generator.
    state.optimizer = _make_optimizer(state)
    state.scaler = _make_scaler(state)
    state.generator = torch.Generator(device=device).manual_seed(seed)
    # 3. Training split to the device (uint8; cast per batch) and its normalization statistics.
    state.images = data.images.to(device, non_blocking=True)
    state.labels = data.labels.to(device, non_blocking=True)
    mean, std = _channel_stats(state.images)
    model.mean.copy_(mean.view(1, 3, 1, 1))
    model.std.copy_(std.view(1, 3, 1, 1))


def train(state) -> nn.Module:
    """The training loop. Timed. Returns the eager model for the harness's evaluation."""
    cfg = state.cfg
    n = state.images.shape[0]
    batch_size = min(cfg.batch_size, n)  # synthetic fixtures have only 64 images
    steps_per_epoch = n // batch_size  # drop the last partial batch: fixed shapes
    total_steps = cfg.epochs * steps_per_epoch
    state.train_model.train()
    step = 0
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, device=state.device, generator=state.generator)
        for i in range(steps_per_epoch):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            xb = _augment(state.images[idx], state.generator)
            _train_step(state, xb, state.labels[idx], _lr_at(step, total_steps, cfg))
            step += 1
    return state.model
