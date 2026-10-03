"""CPU correctness checks; run with the pinned harness environment."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cifar100-speedrun"))
from benchmark.api import BuildContext
from benchmark.data import synthetic_split
from benchmark.worker import load_submission, seed_everything

RECIPE = (
    Path(__file__).resolve().parents[1]
    / "cifar100-speedrun/submissions/futurebiohackers"
)


@pytest.fixture
def recipe():
    torch.set_num_threads(4)
    return load_submission(RECIPE)


@pytest.mark.parametrize("radius", [0, 1, 2])
def test_indexed_crop_matches_original(recipe, radius):
    images = torch.arange(11 * 3 * 32 * 32).reshape(11, 3, 32, 32).float()
    images = F.pad(images, (radius,) * 4, "reflect").to(
        memory_format=torch.channels_last
    )
    torch.manual_seed(7)
    expected = recipe.batch_crop(images, 32)
    torch.manual_seed(7)
    actual = recipe.indexed_crop(images, 32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.is_contiguous(memory_format=torch.channels_last)


@pytest.mark.parametrize("resolution", [24, 28, 32])
@pytest.mark.parametrize("hard_fraction", [0.5, 1.0])
def test_transition_reset_and_evaluation(recipe, resolution, hard_fraction):
    data = synthetic_split(train=True)
    original = data.images.clone()
    original_labels = data.labels.clone()
    state = recipe.build(
        BuildContext(
            torch.device("cpu"),
            {
                "widths": [8, 16, 32],
                "depths": [2, 3, 3],
                "batch_size": 16,
                "epochs": 1.5,
                "train_resolution": resolution,
                "crop_mode": "indexed",
                "whiten_bias_epochs": 0.5,
                "hard_fraction": hard_fraction,
                "proxy_widths": [4, 8, 16],
                "pool_first": [False, True, True],
                "compile_loss": True,
                "fused_sgd": True,
            },
        )
    )
    seed_everything(5)
    recipe.prepare(state, data, 5)
    initialized = {k: v.clone() for k, v in state.net.state_dict().items()}
    proxy_initial = (
        {k: v.clone() for k, v in state.proxy.state_dict().items()}
        if state.proxy
        else {}
    )
    model = recipe.train(state)
    before_eval = {k: v.clone() for k, v in model.state_dict().items()}
    all_buffers = {k: v.clone() for k, v in model.named_buffers()}
    model.eval()
    with torch.inference_mode():
        x = data.images[:7].float() / 255
        logits = model(x)
        single = model(x[:1])
    assert logits.shape == (7, 100) and torch.isfinite(logits).all()
    torch.testing.assert_close(logits[:1], single, rtol=1e-4, atol=1e-5)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before_eval[name], rtol=0, atol=0)
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, all_buffers[name], rtol=0, atol=0)
    seed_everything(5)
    recipe.prepare(state, data, 5)
    for name, value in state.net.state_dict().items():
        torch.testing.assert_close(value, initialized[name], rtol=0, atol=0)
    assert not state.optimizer.state
    assert all(p.grad is None for p in state.net.parameters())
    if state.proxy is not None:
        assert not state.proxy_optimizer.state
        assert all(p.grad is None for p in state.proxy.parameters())
        assert state.masks == []
        for name, value in state.proxy.state_dict().items():
            torch.testing.assert_close(value, proxy_initial[name], rtol=0, atol=0)
    for ema, value in zip(state.ema, state.float_state):
        torch.testing.assert_close(ema, value, rtol=0, atol=0)
    torch.testing.assert_close(data.images, original, rtol=0, atol=0)
    torch.testing.assert_close(data.labels, original_labels, rtol=0, atol=0)


@pytest.mark.parametrize(
    "params",
    [
        {"depths": [2, 2]},
        {"train_resolution": 16},
        {"crop_mode": "wrong"},
        {"batch_size": 0},
    ],
)
def test_invalid_parameters(recipe, params):
    with pytest.raises(ValueError):
        recipe.build(BuildContext(torch.device("cpu"), params))
