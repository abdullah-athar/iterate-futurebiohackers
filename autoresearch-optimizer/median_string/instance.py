"""Problem instance definition for the Median String (Steiner String) problem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProblemInstance:
    """Represents an instance of the Median String (Steiner String) problem.

    Attributes:
        name: Unique identifier for the instance.
        strings: The collection of target sequences S = {s_1, s_2, ..., s_k}.
        alphabet: Permitted characters (e.g., "ACGT" for DNA).
        target_length: Desired length of the median string, or None if variable length is allowed.
        metric: Distance metric to use ("levenshtein" or "hamming").
        description: Human-readable context (e.g. biological source, motif model).
        known_best_score: Known optimal or best-known sum of distances, if available.
        planted_consensus: Planted ground-truth string if generated synthetically.
    """

    name: str
    strings: list[str]
    alphabet: str = "ACGT"
    target_length: int | None = None
    metric: str = "levenshtein"
    description: str = ""
    known_best_score: int | None = None
    planted_consensus: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.strings:
            raise ValueError(f"Instance '{self.name}' must contain at least one string.")
        self.alphabet_set = set(self.alphabet)

    @property
    def num_strings(self) -> int:
        return len(self.strings)

    @property
    def min_length(self) -> int:
        return min(len(s) for s in self.strings)

    @property
    def max_length(self) -> int:
        return max(len(s) for s in self.strings)

    @property
    def mean_length(self) -> float:
        return sum(len(s) for s in self.strings) / len(self.strings)

    def validate_candidate(self, candidate: str) -> tuple[bool, str]:
        """Validate whether a candidate solution complies with instance constraints.

        Returns:
            (is_valid, error_message)
        """
        if not isinstance(candidate, str):
            return False, f"Candidate must be a string, got {type(candidate).__name__}."
        if len(candidate) == 0:
            return False, "Candidate string cannot be empty."

        invalid_chars = set(candidate) - self.alphabet_set
        if invalid_chars:
            return False, f"Candidate contains characters not in alphabet '{self.alphabet}': {sorted(invalid_chars)}"

        if self.target_length is not None and len(candidate) != self.target_length:
            return False, f"Candidate length {len(candidate)} does not match target length {self.target_length}."

        return True, ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "strings": self.strings,
            "alphabet": self.alphabet,
            "target_length": self.target_length,
            "metric": self.metric,
            "description": self.description,
            "known_best_score": self.known_best_score,
            "planted_consensus": self.planted_consensus,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProblemInstance:
        return cls(
            name=data["name"],
            strings=data["strings"],
            alphabet=data.get("alphabet", "ACGT"),
            target_length=data.get("target_length"),
            metric=data.get("metric", "levenshtein"),
            description=data.get("description", ""),
            known_best_score=data.get("known_best_score"),
            planted_consensus=data.get("planted_consensus"),
            metadata=data.get("metadata", {}),
        )
