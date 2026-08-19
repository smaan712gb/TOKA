"""Records written by `toka.log`.

The only format here that Toka writes itself, which makes this the one
adapter that never has to infer anything: the fields are already the
`Request` fields, the provider is recorded per record rather than assumed
for the file, and `cache_visible` was decided at the call site where the
response shape was actually in hand.

The per-record provider matters for the same reason it mattered for
Cline. One process can call Anthropic and DeepSeek in the same session,
and a file-level provider would price half of it wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..record import Request
from .base import as_int, iter_objects, scoped_session


class TokaLogAdapter:
    name = "toka-log"
    # Nominal only — every record carries its own, and that is what is used.
    provider = "anthropic"

    def detect(self, sample: list[dict]) -> float:
        for obj in sample:
            if "toka" in obj and "fresh_input" in obj:
                return 1.0  # our own marker; no other format has it
        return 0.0

    def parse(self, path: Path) -> Iterator[Request]:
        for record in iter_objects(path):
            if "toka" not in record:
                continue
            yield Request(
                source=self.name,
                provider=record.get("provider") or "unknown",
                session=str(record.get("session") or scoped_session(path, path.stem)),
                seq=as_int(record.get("seq")),
                timestamp=record.get("timestamp"),
                model=record.get("model"),
                fresh_input=as_int(record.get("fresh_input")),
                cache_write_5m=as_int(record.get("cache_write_5m")),
                cache_write_1h=as_int(record.get("cache_write_1h")),
                cache_read=as_int(record.get("cache_read")),
                output=as_int(record.get("output")),
                thinking=as_int(record.get("thinking")),
                # Absent means an older record from before the field
                # existed. Assuming it was visible would invent a
                # recoverable figure out of a file that never claimed one.
                cache_visible=bool(record.get("cache_visible", False)),
            )
