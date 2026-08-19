"""Parse agent transcripts into normalised per-request records.

Currently reads Claude Code session transcripts (.jsonl). Each assistant
message carries a `usage` object with the cache accounting we need.

One transcript line can contain a `usage.iterations` array when a single
logical turn made several model requests. We use the top-level totals and
count the line as one request, which keeps the arithmetic consistent with
what was actually billed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class Request:
    session: str
    seq: int  # position within the session, 0-based
    timestamp: str | None
    model: str | None
    fresh_input: int
    cache_write_5m: int
    cache_write_1h: int
    cache_read: int
    output: int
    thinking: int

    @property
    def context_size(self) -> int:
        """Total prompt tokens the model saw for this request."""
        return (
            self.fresh_input
            + self.cache_write_5m
            + self.cache_write_1h
            + self.cache_read
        )


def _usage_of(record: dict) -> dict | None:
    message = record.get("message")
    if not isinstance(message, dict):
        return None
    usage = message.get("usage")
    return usage if isinstance(usage, dict) else None


def _split_cache_writes(usage: dict) -> tuple[int, int]:
    """Return (5m tokens, 1h tokens).

    Prefer the explicit `cache_creation` breakdown. Older transcripts only
    carry the `cache_creation_input_tokens` total; attribute those to the
    5m tier, which is the cheaper of the two, so the estimate stays
    conservative rather than inflating measured cost.
    """
    total = int(usage.get("cache_creation_input_tokens") or 0)
    breakdown = usage.get("cache_creation")
    if isinstance(breakdown, dict):
        five = int(breakdown.get("ephemeral_5m_input_tokens") or 0)
        hour = int(breakdown.get("ephemeral_1h_input_tokens") or 0)
        if five + hour > 0:
            return five, hour
    return total, 0


def read_transcript(path: Path) -> Iterator[Request]:
    """Yield one Request per billed model call in a transcript."""
    session = path.stem
    seq = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            usage = _usage_of(record)
            if usage is None:
                continue

            five, hour = _split_cache_writes(usage)
            details = usage.get("output_tokens_details")
            thinking = 0
            if isinstance(details, dict):
                thinking = int(details.get("thinking_tokens") or 0)

            message = record["message"]
            yield Request(
                session=session,
                seq=seq,
                timestamp=record.get("timestamp"),
                model=message.get("model"),
                fresh_input=int(usage.get("input_tokens") or 0),
                cache_write_5m=five,
                cache_write_1h=hour,
                cache_read=int(usage.get("cache_read_input_tokens") or 0),
                output=int(usage.get("output_tokens") or 0),
                thinking=thinking,
            )
            seq += 1


def read_all(paths: list[Path]) -> list[Request]:
    out: list[Request] = []
    for path in paths:
        out.extend(read_transcript(path))
    return out


def find_transcripts(root: Path) -> list[Path]:
    """All .jsonl transcripts under `root`, newest first."""
    files = [p for p in root.rglob("*.jsonl") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files
