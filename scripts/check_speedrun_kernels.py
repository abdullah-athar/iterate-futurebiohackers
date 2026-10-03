"""GPU equivalence checks for custom kernels, separate from scored harness runs."""

import importlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from benchmark.worker import load_submission


def main():
    load_submission(Path("/root/speedrun/submissions/futurebiohackers"))
    kernels = importlib.import_module("benchmark._submission.kernels")
    for size in (24, 28, 32):
        for n in (1, 33, 1025):
            for channels_last in (True, False):
                images = torch.randn(
                    n, 3, size, size, device="cuda", dtype=torch.float16
                )
                images = F.pad(images, (2,) * 4, "reflect")
                if channels_last:
                    images = images.to(memory_format=torch.channels_last)
                shifts = torch.randint(0, 5, (n, 2), device="cuda")
                batch = torch.arange(n, device="cuda").view(n, 1, 1)
                rows = shifts[:, 0, None, None] + torch.arange(
                    size, device="cuda"
                ).view(1, -1, 1)
                cols = shifts[:, 1, None, None] + torch.arange(
                    size, device="cuda"
                ).view(1, 1, -1)
                expected = images.permute(0, 2, 3, 1)[batch, rows, cols].permute(
                    0, 3, 1, 2
                )
                for flip in (False, True):
                    actual = kernels.crop_flip(images, size, flip, shifts)
                    torch.testing.assert_close(
                        actual, expected.flip(-1) if flip else expected, rtol=0, atol=0
                    )
                    assert actual.is_contiguous(memory_format=torch.channels_last)
    torch.cuda.synchronize()
    Path(sys.argv[1]).write_text(
        "36 crop/flip checks passed; both layouts, all resolutions.\n"
    )


if __name__ == "__main__":
    main()
