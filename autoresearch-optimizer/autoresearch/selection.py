"""Selectors: decide which results to accept and which records to build on next."""

from __future__ import annotations

from .core import Memory, Record, Result


class GreedySelector:
    """Hill climbing: accept only valid strict improvements; always build on the champion."""

    def accept(self, memory: Memory, result: Result) -> bool:
        champ = memory.champion()
        return result.valid and (champ is None or result.score < champ.result.score)

    def parents(self, memory: Memory) -> list[Record]:
        champ = memory.champion()
        return [champ] if champ else []
