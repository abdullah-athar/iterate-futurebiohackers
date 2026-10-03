"""Search and held-out benchmark suites.

A non-zero seed offset regenerates every instance of a tier with the same generator
settings but different random seeds, giving a held-out suite the loop never sees.
"""

from __future__ import annotations

from median_string.benchmarks import generate_planted_instance, get_benchmark_suite
from median_string.instance import ProblemInstance

from .core import SuiteSpec

HELDOUT_SEED_OFFSET = 10_000


def build_suite(spec: SuiteSpec) -> list[ProblemInstance]:
    suite = get_benchmark_suite(spec.tier)
    if spec.seed_offset == 0:
        return suite
    return [
        generate_planted_instance(
            name=f"{inst.name}_s{spec.seed_offset}",
            alphabet=inst.alphabet,
            target_length=len(inst.planted_consensus),
            num_strings=len(inst.strings),
            mutation_rate=inst.metadata["mutation_rate"],
            indel_rate=inst.metadata["indel_rate"],
            seed=inst.metadata["seed"] + spec.seed_offset,
            metric=inst.metric,
            description=inst.description,
        )
        for inst in suite
    ]
