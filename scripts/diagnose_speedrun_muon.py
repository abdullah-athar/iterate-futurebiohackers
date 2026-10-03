"""Training-only numerical diagnosis. Synchronizations make timings unscored."""

import json
import sys
from pathlib import Path

import torch
from benchmark.api import BuildContext
from benchmark.data import load_split
from benchmark.worker import load_submission, seed_everything


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    module = load_submission(Path("/root/speedrun/submissions/futurebiohackers"))
    params = {
        "optimizer": "muon",
        "activation": "gelu",
        "global_pool": "max",
        "compile_forward_loss": True,
        "fused_sgd": True,
        "ema_every": 0,
        "whiten_bias_epochs": 0.2,
        "epochs": 8.5,
    }
    state = module.build(BuildContext(torch.device("cuda"), params))
    data = load_split(Path("/data"), train=True)
    seed_everything(60000)
    module.prepare(state, data, 60000)
    forward = state.train_loss
    history = []

    def checked_forward(inputs, labels, whiten_bias_grad):
        step = len(history)
        for name, value in list(state.net.named_parameters()) + list(
            state.net.named_buffers()
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f"step {step}: nonfinite parameter/buffer {name}")
        loss = forward(inputs, labels, whiten_bias_grad)
        value = loss.item()
        history.append({"step": step, "loss": value})
        if step % 20 == 0:
            print(json.dumps(history[-1]), flush=True)
        if not torch.isfinite(loss):
            raise ValueError(f"step {step}: nonfinite loss")
        return loss

    state.train_loss = checked_forward
    update = state.muon_optimizer.zeropower

    def checked_update(grads):
        step = len(history) - 1
        for i, g in enumerate(grads):
            if not torch.isfinite(g).all():
                raise ValueError(
                    f"step {step}: nonfinite Muon gradient, shape {list(g.shape)}"
                )
        outputs = update(grads)
        for g, output in zip(grads, outputs):
            if not torch.isfinite(output).all():
                raise ValueError(
                    f"step {step}: nonfinite Muon update, shape {list(g.shape)}; grad max {g.abs().max().item()}"
                )
        return outputs

    state.muon_optimizer.zeropower = checked_update
    result = {"parameters": params, "scored": False}
    try:
        module.train(state)
        result["status"] = "finite training completed"
    except ValueError as error:
        result["status"] = str(error)
    result["last_losses"] = history[-8:]
    Path(sys.argv[1], "muon-diagnostic.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
