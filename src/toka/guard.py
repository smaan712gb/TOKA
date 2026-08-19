"""Find out what broke your prompt cache.

The measurement side of Toka says prefix churn is where the money goes.
This is the other half: given the request you are about to send, it tells
you whether the cached prefix survived since last turn, and if not, which
bytes killed it.

Caching is a prefix match — one changed byte invalidates everything after
it. Providers render the prompt in a fixed order (tools, then system, then
messages), so the prefix is that concatenation and a break is the first
position where this turn diverges from the last.

Nothing here calls a network or needs an API key. You hand it the same
arguments you are about to hand your client:

    guard = PrefixGuard()

    report = guard.check(system=system, tools=tools, messages=messages)
    if not report.stable:
        print(report.explain())

Output looks like:

    prefix broke at turn 7 — 94% of the cached prefix invalidated
      segment: system[0]
      cause:   looks like a timestamp
      before:  ...Current date: 2026-08-19T10:33:01Z\\nYou are...
      after:   ...Current date: 2026-08-19T10:34:15Z\\nYou are...
                              ^ first difference at offset 1204

      Move volatile content after the last cache breakpoint.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# Patterns whose presence in a changed span usually explains the break
# outright. Ordered most-specific first.
_INVALIDATORS: list[tuple[str, re.Pattern]] = [
    ("looks like a timestamp", re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")),
    ("looks like a UUID", re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-", re.I)),
    ("looks like a date", re.compile(r"\d{4}-\d{2}-\d{2}")),
    ("looks like an epoch timestamp", re.compile(r"\b1[6-9]\d{8,11}\b")),
    ("looks like a token/session id", re.compile(r"\b[A-Za-z0-9_-]{24,}\b")),
    ("looks like a counter", re.compile(r"\b\d{1,6}\b")),
]

CONTEXT_CHARS = 48


@dataclass
class Segment:
    """One addressable piece of the rendered prefix."""

    label: str
    text: str


@dataclass
class Break:
    label: str
    kind: str  # "changed" | "added" | "removed"
    offset: int  # first differing character within the segment
    before: str | None
    after: str | None
    cause: str | None

    def context(self) -> tuple[str, str]:
        """Windows around the divergence, for display."""
        start = max(0, self.offset - CONTEXT_CHARS)
        end = self.offset + CONTEXT_CHARS
        b = (self.before or "")[start:end]
        a = (self.after or "")[start:end]
        return b, a


@dataclass
class CheckReport:
    turn: int
    stable: bool
    prefix_chars: int
    reusable_chars: int
    break_: Break | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def invalidated_chars(self) -> int:
        return max(0, self.prefix_chars - self.reusable_chars)

    @property
    def invalidated_pct(self) -> float:
        if self.prefix_chars == 0:
            return 0.0
        return 100.0 * self.invalidated_chars / self.prefix_chars

    def explain(self) -> str:
        if self.stable:
            line = (
                f"prefix stable at turn {self.turn} — "
                f"{self.reusable_chars:,} chars reusable"
            )
            return "\n".join([line] + [f"  note:    {n}" for n in self.notes])

        out = [
            f"prefix broke at turn {self.turn} — "
            f"{self.invalidated_pct:.0f}% of the cached prefix invalidated "
            f"({self.invalidated_chars:,} chars)"
        ]
        b = self.break_
        if b is not None:
            out.append(f"  segment: {b.label}")
            if b.cause:
                out.append(f"  cause:   {b.cause}")
            if b.kind == "changed":
                before, after = b.context()
                out.append(f"  before:  ...{_escape(before)}...")
                out.append(f"  after:   ...{_escape(after)}...")
            else:
                out.append(f"  change:  segment was {b.kind}")
        for note in self.notes:
            out.append(f"  note:    {note}")
        out.append("")
        out.append("  " + _advice(self.break_))
        return "\n".join(out)


class PrefixGuard:
    """Tracks one conversation's prefix across turns.

    Use one guard per conversation — it compares each call against the
    previous one, so sharing an instance across conversations reports
    breaks that did not happen.
    """

    def __init__(self) -> None:
        self._previous: list[Segment] | None = None
        self._turn = 0

    def check(
        self,
        *,
        system=None,
        tools=None,
        messages=None,
    ) -> CheckReport:
        segments = render(system=system, tools=tools, messages=messages)
        total = sum(len(s.text) for s in segments)
        self._turn += 1

        if self._previous is None:
            self._previous = segments
            return CheckReport(
                turn=self._turn,
                stable=True,
                prefix_chars=total,
                reusable_chars=0,
                notes=["first turn — nothing cached yet"],
            )

        brk, reusable = _first_divergence(self._previous, segments)
        self._previous = segments

        if brk is None:
            return CheckReport(
                turn=self._turn,
                stable=True,
                prefix_chars=total,
                reusable_chars=reusable,
            )

        notes = []
        if brk.label.startswith("tools"):
            notes.append(
                "tools render before everything else, so any change here "
                "invalidates the entire prompt"
            )
        elif brk.label.startswith("system"):
            notes.append("system renders before messages, so all history was lost too")

        return CheckReport(
            turn=self._turn,
            stable=False,
            prefix_chars=total,
            reusable_chars=reusable,
            break_=brk,
            notes=notes,
        )


def render(*, system=None, tools=None, messages=None) -> list[Segment]:
    """Flatten a request into labelled prefix segments, in render order.

    Order is tools -> system -> messages, which is how the prompt is
    assembled for cache-key purposes. Labels are addressable so a break
    points at something you can actually go and find.
    """
    segments: list[Segment] = []

    for i, tool in enumerate(tools or []):
        name = _tool_name(tool) or str(i)
        segments.append(Segment(f"tools[{i}]:{name}", _canonical(tool)))

    for i, block in enumerate(_as_blocks(system)):
        segments.append(Segment(f"system[{i}]", _text_of(block)))

    for i, message in enumerate(messages or []):
        role = message.get("role", "?") if isinstance(message, dict) else "?"
        content = message.get("content") if isinstance(message, dict) else message
        for j, block in enumerate(_as_blocks(content)):
            segments.append(
                Segment(f"messages[{i}].{role}.content[{j}]", _text_of(block))
            )

    return segments


def _first_divergence(
    old: list[Segment], new: list[Segment]
) -> tuple[Break | None, int]:
    """Return (break, reusable chars before it)."""
    reusable = 0
    for i in range(max(len(old), len(new))):
        o = old[i] if i < len(old) else None
        n = new[i] if i < len(new) else None

        if o is None:
            # New segment appended past the old prefix. Appending is how
            # conversations grow — it does not invalidate anything.
            return None, reusable
        if n is None:
            return (
                Break(o.label, "removed", 0, o.text, None, None),
                reusable,
            )
        if o.label != n.label or o.text != n.text:
            offset = _first_diff_offset(o.text, n.text)
            kind = "changed" if o.label == n.label else "added"
            return (
                Break(n.label, kind, offset, o.text, n.text, _classify(o.text, n.text, offset)),
                reusable,
            )
        reusable += len(n.text)
    return None, reusable


def _first_diff_offset(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit


def _classify(before: str, after: str, offset: int) -> str | None:
    """Guess why two segments diverged, from the text around the break.

    The lookback has to be wide enough to contain the whole token the
    change sits inside. A timestamp diverging at its minutes digit is
    ~20 characters in, and a window that clips the year matches the bare
    number pattern instead — reporting "a counter" for the single most
    common invalidator there is.
    """
    window = slice(max(0, offset - CONTEXT_CHARS), offset + CONTEXT_CHARS)
    span_before, span_after = before[window], after[window]
    for label, pattern in _INVALIDATORS:
        if pattern.search(span_before) and pattern.search(span_after):
            return label
    if len(before) != len(after) and before.strip() == after.strip():
        return "whitespace only"
    return None


def _advice(brk: Break | None) -> str:
    if brk is None:
        return ""
    if brk.label.startswith("tools"):
        return (
            "Keep the tool list fixed and serialise it deterministically "
            "(sort by name). If the set must change, look for a "
            "mid-conversation tool-change API rather than editing `tools`."
        )
    if brk.label.startswith("system"):
        return (
            "Move volatile content out of the system prompt and into a "
            "later message, after the last cache breakpoint."
        )
    return (
        "Something rewrote earlier history. Append rather than edit — "
        "compaction and context editing both break the prefix."
    )


def _as_blocks(content) -> list:
    if content is None:
        return []
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return content
    return [content]


def _text_of(block) -> str:
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        text = block.get("text")
        if isinstance(text, str):
            return text
        return _canonical(block)
    return str(block)


def _tool_name(tool) -> str | None:
    if isinstance(tool, dict):
        name = tool.get("name")
        if isinstance(name, str):
            return name
    return None


def _without_directives(obj):
    """Drop `cache_control` wherever it appears.

    It tells the provider where to put a breakpoint; the model never sees
    it. Leaving it in would make moving a breakpoint look like the prompt
    changed — and would make `toka.repair` fail its own safety check the
    moment it placed one on a tool.
    """
    if isinstance(obj, dict):
        return {
            k: _without_directives(v) for k, v in obj.items() if k != "cache_control"
        }
    if isinstance(obj, list):
        return [_without_directives(v) for v in obj]
    return obj


def _canonical(obj) -> str:
    """Stable serialisation, so key order never registers as a break.

    A dict whose keys iterate in a different order is not a real prompt
    change, but it is a real cache break for anyone serialising without
    sort_keys. Canonicalising here means the guard reports genuine
    content changes; unsorted serialisation is reported separately by
    `sorted_keys_warning`.
    """
    try:
        return json.dumps(
            _without_directives(obj),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except (TypeError, ValueError):
        return str(obj)


def sorted_keys_warning(tools) -> str | None:
    """Flag tool definitions that would serialise unstably.

    The guard canonicalises before comparing, so it will not itself see a
    break from key order — but the provider sees whatever your client
    sends. If your serialiser does not sort keys, the cache breaks even
    though nothing changed.
    """
    if not tools:
        return None
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if list(tool.keys()) != sorted(tool.keys()):
            return (
                "tool definitions are not in sorted key order — if your "
                "client serialises without sort_keys, the prefix changes "
                "between turns even when the tools do not"
            )
    return None


def _escape(text: str) -> str:
    return text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
