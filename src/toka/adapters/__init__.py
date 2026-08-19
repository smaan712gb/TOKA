"""Adapter registry and format auto-detection.

`toka analyze <path>` should work on any supported transcript without the
user naming a format, so every file is sniffed and routed to the adapter
that recognises it. Adding an agent means appending one entry to ADAPTERS.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterator

from ..record import Request
from .base import Adapter, sniff
from .claude_code import ClaudeCodeAdapter
from .cline import ClineAdapter
from .gemini import GeminiAdapter
from .openai_compat import OpenAICompatAdapter

ADAPTERS: list[Adapter] = [
    ClaudeCodeAdapter(),
    ClineAdapter(),
    OpenAICompatAdapter(),
    GeminiAdapter(),
]

# Below this, we assume nothing recognised the file rather than guessing.
MIN_CONFIDENCE = 0.5


def adapter_for(path: Path) -> Adapter | None:
    """Best adapter for a file, or None if no format is recognised."""
    sample = sniff(path)
    if not sample:
        return None
    best: Adapter | None = None
    best_score = 0.0
    for adapter in ADAPTERS:
        score = adapter.detect(sample)
        if score > best_score:
            best, best_score = adapter, score
    return best if best_score >= MIN_CONFIDENCE else None


def parse(path: Path) -> Iterator[Request]:
    adapter = adapter_for(path)
    if adapter is None:
        return iter(())
    return adapter.parse(path)


def parse_all(paths: list[Path]) -> tuple[list[Request], Counter, list[Path]]:
    """Parse many files.

    Returns (records, per-adapter file counts, unrecognised files) so the
    CLI can report coverage instead of silently dropping input.
    """
    records: list[Request] = []
    used: Counter = Counter()
    skipped: list[Path] = []
    for path in paths:
        adapter = adapter_for(path)
        if adapter is None:
            skipped.append(path)
            continue
        before = len(records)
        records.extend(adapter.parse(path))
        if len(records) > before:
            used[adapter.name] += 1
        else:
            skipped.append(path)
    return records, used, skipped


__all__ = ["ADAPTERS", "adapter_for", "parse", "parse_all", "Request"]
