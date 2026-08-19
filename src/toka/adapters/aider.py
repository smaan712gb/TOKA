"""Aider — `.aider.chat.history.md` in the project directory.

UNVERIFIED. Built from Aider's documented history format, not from real
traffic. Every other adapter in this package was written against actual
files on disk; this one was not, and building Cline blind would have
shipped two silent bugs. Treat its numbers as provisional until someone
runs it against a real `.aider.chat.history.md` and confirms them.

Aider writes a markdown transcript with per-message accounting lines:

    > Tokens: 3.1k sent, 226 received. Cost: $0.01 message, $0.10 session.
    > Model: gpt-4o with diff edit format

The counts are human-formatted (`3.1k`, `1.2M`), so they are rounded at
the source — a 3,149-token request and a 3,051-token one both log as
`3.1k`. That is fine for spotting a pattern and wrong for an invoice, so
this adapter is for shape, not for cost reconciliation.

Aider does not report cache accounting in this file, so records are
marked cache_visible=False and stay out of the recoverable figure.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from ..record import Request

VERIFIED = False

# > Tokens: 3.1k sent, 226 received.
_TOKENS = re.compile(
    r"Tokens:\s*([\d.,]+\s*[kKmM]?)\s*sent,\s*([\d.,]+\s*[kKmM]?)\s*received",
)
# > Model: gpt-4o with diff edit format
_MODEL = re.compile(r"^>\s*Model:\s*([^\s]+)")
# # aider chat started at 2025-01-01 10:00:00
_STARTED = re.compile(r"aider chat started at\s*(.+?)\s*$")


class AiderAdapter:
    name = "aider"
    provider = "unknown"
    verified = VERIFIED

    def detect(self, sample: list[dict]) -> float:
        # Markdown, so the JSON sniffer yields nothing — routing is by
        # filename instead. See adapters.adapter_for.
        return 0.0

    def claims(self, path: Path) -> bool:
        return path.name.endswith(".aider.chat.history.md") or path.name == (
            "aider.chat.history.md"
        )

    def parse(self, path: Path) -> Iterator[Request]:
        session = path.parent.name or path.stem
        seq = 0
        model: str | None = None
        started: str | None = None

        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return

        for line in lines:
            m = _STARTED.search(line)
            if m:
                started = m.group(1)
                # A new "chat started" banner begins a new session.
                seq = 0
                continue

            m = _MODEL.match(line.strip())
            if m:
                model = m.group(1)
                continue

            m = _TOKENS.search(line)
            if not m:
                continue

            yield Request(
                source=self.name,
                provider=_provider_of(model),
                session=f"{session}@{started}" if started else session,
                seq=seq,
                timestamp=started,
                model=model,
                fresh_input=_count(m.group(1)),
                cache_write_5m=0,
                cache_write_1h=0,
                cache_read=0,
                output=_count(m.group(2)),
                # Aider's history has no cache fields; claiming these as
                # misses would report the whole session as waste.
                cache_visible=False,
            )
            seq += 1


def _count(raw: str) -> int:
    """Parse Aider's human-formatted counts: '3.1k' -> 3100."""
    text = raw.strip().replace(",", "")
    mult = 1
    if text[-1:] in "kK":
        mult, text = 1_000, text[:-1]
    elif text[-1:] in "mM":
        mult, text = 1_000_000, text[:-1]
    try:
        return int(float(text) * mult)
    except ValueError:
        return 0


def _provider_of(model: str | None) -> str:
    if not model:
        return "unknown"
    m = model.lower()
    if "claude" in m or m.startswith("anthropic"):
        return "anthropic"
    if "gpt" in m or m.startswith(("openai", "o1", "o3")):
        return "openai"
    if "gemini" in m or m.startswith("google"):
        return "google"
    return "unknown"
