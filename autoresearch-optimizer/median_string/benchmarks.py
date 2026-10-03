"""Benchmark datasets and test suites for the Median String problem."""

from __future__ import annotations

import random
from typing import Literal
from .instance import ProblemInstance


def generate_planted_instance(
    name: str,
    alphabet: str = "ACGT",
    target_length: int = 20,
    num_strings: int = 8,
    mutation_rate: float = 0.2,
    indel_rate: float = 0.05,
    seed: int = 42,
    metric: str = "levenshtein",
    description: str = "",
) -> ProblemInstance:
    """Generate a synthetic Median String instance from a known planted consensus sequence.

    Each sequence is derived from the planted consensus by applying substitutions,
    insertions, and deletions according to the specified rates.
    """
    rng = random.Random(seed)
    alpha_list = list(alphabet)

    # 1. Generate planted consensus
    planted = "".join(rng.choice(alpha_list) for _ in range(target_length))

    # 2. Generate noisy variants
    variants: list[str] = []
    for _ in range(num_strings):
        chars: list[str] = []
        for ch in planted:
            # Possible deletion
            if indel_rate > 0 and rng.random() < indel_rate:
                continue

            # Possible insertion before
            if indel_rate > 0 and rng.random() < indel_rate:
                chars.append(rng.choice(alpha_list))

            # Possible substitution
            if rng.random() < mutation_rate:
                subs = [c for c in alpha_list if c != ch]
                chars.append(rng.choice(subs) if subs else ch)
            else:
                chars.append(ch)

        # Fallback to prevent empty variant
        if not chars:
            chars.append(rng.choice(alpha_list))

        variants.append("".join(chars))

    from .metrics import sum_distance
    planted_score = sum_distance(planted, variants, metric=metric)

    return ProblemInstance(
        name=name,
        strings=variants,
        alphabet=alphabet,
        # No length constraint: the optimal median can be shorter or longer than the planted sequence.
        target_length=None,
        metric=metric,
        description=description or f"Planted motif (len={target_length}, k={num_strings}, mut={mutation_rate})",
        planted_consensus=planted,
        known_best_score=planted_score,
        metadata={
            "seed": seed,
            "mutation_rate": mutation_rate,
            "indel_rate": indel_rate,
            "planted_score": planted_score,
        },
    )


def get_small_suite() -> list[ProblemInstance]:
    """Fast smoke-test suite (3 instances, small strings, executes in < 0.1s)."""
    return [
        # Instance 1: Small DNA motif (fixed length, pure substitutions)
        generate_planted_instance(
            name="dna_planted_small_fixed",
            alphabet="ACGT",
            target_length=12,
            num_strings=6,
            mutation_rate=0.15,
            indel_rate=0.0,
            seed=101,
            metric="levenshtein",
            description="Small DNA motif, 6 sequences of length 12 with substitutions only",
        ),
        # Instance 2: Small DNA motif with indels (variable length Steiner string)
        generate_planted_instance(
            name="dna_planted_small_indels",
            alphabet="ACGT",
            target_length=16,
            num_strings=8,
            mutation_rate=0.20,
            indel_rate=0.08,
            seed=102,
            metric="levenshtein",
            description="DNA sequences with insertions/deletions, testing Steiner consensus",
        ),
        # Instance 3: Protein/Amino acid small motif
        generate_planted_instance(
            name="protein_short_motif",
            alphabet="ACDEFGHIKLMNPQRSTVWY",
            target_length=10,
            num_strings=5,
            mutation_rate=0.25,
            indel_rate=0.0,
            seed=103,
            metric="levenshtein",
            description="Short protein motif across 20-letter amino acid alphabet",
        ),
    ]


def get_medium_suite() -> list[ProblemInstance]:
    """Standard benchmark suite (5 instances, medium length 25-50, executes in ~1s)."""
    return [
        generate_planted_instance(
            name="dna_promoter_25bp",
            alphabet="ACGT",
            target_length=25,
            num_strings=12,
            mutation_rate=0.20,
            indel_rate=0.04,
            seed=201,
            description="Synthetic promoter motif (25bp) across 12 sequences with moderate indels",
        ),
        generate_planted_instance(
            name="dna_regulatory_40bp",
            alphabet="ACGT",
            target_length=40,
            num_strings=15,
            mutation_rate=0.25,
            indel_rate=0.05,
            seed=202,
            description="Regulatory element (40bp) across 15 noisy sequences",
        ),
        generate_planted_instance(
            name="dna_high_noise_30bp",
            alphabet="ACGT",
            target_length=30,
            num_strings=10,
            mutation_rate=0.35,
            indel_rate=0.08,
            seed=203,
            description="High-noise DNA consensus problem (35% mutation, 8% indels)",
        ),
        generate_planted_instance(
            name="dna_hamming_exact_30bp",
            alphabet="ACGT",
            target_length=30,
            num_strings=12,
            mutation_rate=0.25,
            indel_rate=0.0,
            seed=204,
            metric="hamming",
            description="Equal-length DNA sequences evaluated with Hamming distance",
        ),
        generate_planted_instance(
            name="protein_domain_20aa",
            alphabet="ACDEFGHIKLMNPQRSTVWY",
            target_length=20,
            num_strings=10,
            mutation_rate=0.30,
            indel_rate=0.03,
            seed=205,
            description="Conserved protein functional region (20 residues) across 10 sequences",
        ),
    ]


def get_hard_suite() -> list[ProblemInstance]:
    """Challenging benchmark suite for thorough optimization (5 instances, lengths 50-100)."""
    return [
        generate_planted_instance(
            name="dna_hard_60bp_k20",
            alphabet="ACGT",
            target_length=60,
            num_strings=20,
            mutation_rate=0.28,
            indel_rate=0.06,
            seed=301,
            description="Long DNA sequence (60bp) across 20 sequences with indels",
        ),
        generate_planted_instance(
            name="dna_hard_80bp_k25",
            alphabet="ACGT",
            target_length=80,
            num_strings=25,
            mutation_rate=0.25,
            indel_rate=0.05,
            seed=302,
            description="80bp consensus across 25 sequences",
        ),
        generate_planted_instance(
            name="dna_hard_noisy_50bp",
            alphabet="ACGT",
            target_length=50,
            num_strings=30,
            mutation_rate=0.38,
            indel_rate=0.10,
            seed=303,
            description="Heavy noise Steiner string search (38% substitution, 10% indel, 30 sequences)",
        ),
        generate_planted_instance(
            name="protein_hard_40aa",
            alphabet="ACDEFGHIKLMNPQRSTVWY",
            target_length=40,
            num_strings=15,
            mutation_rate=0.35,
            indel_rate=0.05,
            seed=304,
            description="40-residue protein domain across 15 divergent sequences",
        ),
        generate_planted_instance(
            name="dna_large_cohort_35bp",
            alphabet="ACGT",
            target_length=35,
            num_strings=40,
            mutation_rate=0.25,
            indel_rate=0.05,
            seed=305,
            description="Large cohort test: 40 sequences of length ~35bp",
        ),
    ]


def get_confirm_suite(seed_offset: int = 5000) -> list[ProblemInstance]:
    """Fresh medium-difficulty instances used by the autoresearch loop to re-test a claimed
    new best before accepting it (guards against selecting on evaluation noise/overfitting)."""
    specs = [
        ("confirm_dna_25bp_k12", "ACGT", 25, 12, 0.20, 0.04, "levenshtein"),
        ("confirm_dna_40bp_k15", "ACGT", 40, 15, 0.25, 0.05, "levenshtein"),
        ("confirm_dna_noisy_30bp_k10", "ACGT", 30, 10, 0.35, 0.08, "levenshtein"),
        ("confirm_dna_hamming_30bp_k12", "ACGT", 30, 12, 0.25, 0.0, "hamming"),
        ("confirm_protein_20aa_k10", "ACDEFGHIKLMNPQRSTVWY", 20, 10, 0.30, 0.03, "levenshtein"),
    ]
    return _fresh_suite(specs, seed_offset, "Confirmation instance")


def get_holdout_suite(seed_offset: int = 9000) -> list[ProblemInstance]:
    """Fresh instances with the medium/hard generator settings but unseen seeds.

    Used by the autoresearch loop for final reporting so that search-time feedback
    (small/medium tiers) is separated from the final evaluation.
    """
    specs = [
        ("holdout_dna_25bp_k12", "ACGT", 25, 12, 0.20, 0.04, "levenshtein"),
        ("holdout_dna_40bp_k15", "ACGT", 40, 15, 0.25, 0.05, "levenshtein"),
        ("holdout_dna_noisy_30bp_k10", "ACGT", 30, 10, 0.35, 0.08, "levenshtein"),
        ("holdout_dna_hamming_30bp_k12", "ACGT", 30, 12, 0.25, 0.0, "hamming"),
        ("holdout_protein_20aa_k10", "ACDEFGHIKLMNPQRSTVWY", 20, 10, 0.30, 0.03, "levenshtein"),
        ("holdout_dna_60bp_k20", "ACGT", 60, 20, 0.28, 0.06, "levenshtein"),
    ]
    return _fresh_suite(specs, seed_offset, "Held-out instance")


def _fresh_suite(specs, seed_offset: int, label: str) -> list[ProblemInstance]:
    return [
        generate_planted_instance(
            name=name,
            alphabet=alphabet,
            target_length=length,
            num_strings=k,
            mutation_rate=mut,
            indel_rate=indel,
            seed=seed_offset + i,
            metric=metric,
            description=f"{label} (len={length}, k={k}, mut={mut}, indel={indel})",
        )
        for i, (name, alphabet, length, k, mut, indel, metric) in enumerate(specs, 1)
    ]


BenchmarkTier = Literal["small", "medium", "hard", "confirm", "holdout"]


def get_benchmark_suite(tier: BenchmarkTier = "small") -> list[ProblemInstance]:
    """Retrieve benchmark instances for the given difficulty tier."""
    if tier == "small":
        return get_small_suite()
    elif tier == "medium":
        return get_medium_suite()
    elif tier == "hard":
        return get_hard_suite()
    elif tier == "confirm":
        return get_confirm_suite()
    elif tier == "holdout":
        return get_holdout_suite()
    else:
        raise ValueError(f"Unknown benchmark tier '{tier}'. Choose from 'small', 'medium', 'hard', 'confirm', 'holdout'.")
