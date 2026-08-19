"""Fix the cache breaks the guard finds — safely.

`PrefixGuard` tells you the prefix broke and why. This applies the fixes
that carry no risk, and refuses the ones that do.

The tiers exist because "make the cache work better" spans two very
different kinds of change:

  Tier 1 (default, automatic)
      Changes bytes on the wire without changing a single character the
      model reads. Deterministic key ordering in tool definitions;
      placing `cache_control` at the real stability boundary. These are
      provably meaning-preserving — `verify()` re-renders both versions
      and asserts the model-visible text is identical.

  Tier 2 (opt-in, per change)
      Changes what the model reads or the order it reads it in. Hoisting
      a timestamp out of the system prompt genuinely fixes the largest
      single cause of churn, but it also moves text the model was
      conditioned on. That is a judgment call, so it is proposed, never
      applied.

The distinction is the whole point. A repair pass that saves 20% and
breaks one task in fifty is a bad trade, and token metrics alone will
happily call it a win.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .guard import render

# Serialising a dict without a stable key order produces different bytes
# for identical content, which breaks the cache on a prompt that never
# actually changed.
SORT_KEYS = True


@dataclass
class Change:
    tier: int
    where: str
    what: str
    # Tier 2 only: why we are not doing it for you.
    requires_consent: bool = False

    def __str__(self) -> str:
        mark = "proposed" if self.requires_consent else "applied"
        return f"[tier {self.tier}] {mark}: {self.where} — {self.what}"


@dataclass
class RepairResult:
    system: Any
    tools: Any
    messages: Any
    changes: list[Change] = field(default_factory=list)

    @property
    def applied(self) -> list[Change]:
        return [c for c in self.changes if not c.requires_consent]

    @property
    def proposed(self) -> list[Change]:
        return [c for c in self.changes if c.requires_consent]

    def explain(self) -> str:
        if not self.changes:
            return "nothing to repair — prefix construction is already stable"
        lines = [str(c) for c in self.applied]
        if self.proposed:
            lines.append("")
            lines.append("Not applied — these change what the model reads:")
            lines.extend(f"  {c.where} — {c.what}" for c in self.proposed)
        return "\n".join(lines)


def repair(
    *,
    system=None,
    tools=None,
    messages=None,
    place_cache_control: bool = True,
) -> RepairResult:
    """Apply tier-1 repairs; propose tier-2 ones.

    Returns new objects — the inputs are never mutated, so a caller can
    compare, log, or discard the result without side effects.
    """
    changes: list[Change] = []

    new_tools = _stabilise_tools(tools, changes)
    new_system = _copy(system)
    new_messages = _copy(messages)

    if place_cache_control:
        new_system, new_tools = _place_breakpoint(new_system, new_tools, changes)

    _propose_volatile_hoists(new_system, changes)

    return RepairResult(
        system=new_system, tools=new_tools, messages=new_messages, changes=changes
    )


def verify(original: dict, repaired: RepairResult) -> bool:
    """Assert a tier-1 repair changed no model-visible text.

    This is the safety property, checked rather than claimed. If it ever
    returns False the repair is unsafe and must not ship — which is why
    `repair_safely` raises instead of returning a bad result.
    """
    before = [(s.label, s.text) for s in render(**original)]
    after = [
        (s.label, s.text)
        for s in render(
            system=repaired.system, tools=repaired.tools, messages=repaired.messages
        )
    ]
    return before == after


def repair_safely(*, system=None, tools=None, messages=None) -> RepairResult:
    """`repair` with the invariant enforced.

    Raises rather than returning a result that would alter the prompt.
    Prefer this at call sites that apply the output automatically.
    """
    original = {"system": system, "tools": tools, "messages": messages}
    result = repair(**original)
    if not verify(original, result):
        raise AssertionError(
            "tier-1 repair altered model-visible text; refusing to return it"
        )
    return result


def _stabilise_tools(tools, changes: list[Change]):
    """Give every tool definition a deterministic key order.

    Key order is invisible to the model — it reads the rendered values —
    but it is very visible to the cache, which keys on exact bytes. A
    client serialising without sort_keys can break the prefix on a turn
    where nothing changed at all.
    """
    if not tools:
        return tools

    out = []
    unstable = []
    for tool in tools:
        if not isinstance(tool, dict):
            out.append(tool)
            continue
        if list(tool.keys()) != sorted(tool.keys()):
            unstable.append(tool.get("name") or "<unnamed>")
        out.append(_sorted_deep(tool))

    if unstable:
        changes.append(
            Change(
                tier=1,
                where=f"tools ({', '.join(unstable[:3])}"
                + (f" +{len(unstable) - 3} more" if len(unstable) > 3 else "")
                + ")",
                what="key order normalised so serialisation is byte-stable",
            )
        )
    return out


def _sorted_deep(obj):
    if isinstance(obj, dict):
        return {k: _sorted_deep(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [_sorted_deep(v) for v in obj]
    return obj


def _place_breakpoint(system, tools, changes: list[Change]):
    """Put `cache_control` on the last stable block.

    Tools render before system, so a breakpoint on the final system block
    caches both together. Adding the directive changes nothing the model
    reads — it is an instruction to the provider, not content.
    """
    blocks = _as_block_list(system)
    if not blocks:
        return system, tools

    if any(isinstance(b, dict) and b.get("cache_control") for b in blocks):
        return system, tools  # caller already placed one; leave it alone

    last = blocks[-1]
    if not isinstance(last, dict):
        last = {"type": "text", "text": str(last)}
    else:
        last = dict(last)
    last["cache_control"] = {"type": "ephemeral"}
    blocks = blocks[:-1] + [last]

    changes.append(
        Change(
            tier=1,
            where="system (last block)",
            what="cache_control added at the tools+system boundary",
        )
    )
    return blocks, tools


def _as_block_list(system) -> list:
    if system is None:
        return []
    if isinstance(system, str):
        return [{"type": "text", "text": system}]
    if isinstance(system, list):
        return list(system)
    return [system]


def _propose_volatile_hoists(system, changes: list[Change]) -> None:
    """Flag volatile content in the system prompt without moving it.

    This is the single largest cause of churn, and also the one fix that
    reorders text the model was conditioned on. Proposing it keeps the
    decision with the person who knows whether position matters here.
    """
    from .guard import _INVALIDATORS

    for i, block in enumerate(_as_block_list(system)):
        text = block.get("text") if isinstance(block, dict) else str(block)
        if not isinstance(text, str):
            continue
        for label, pattern in _INVALIDATORS[:4]:  # timestamps, uuids, dates
            m = pattern.search(text)
            if m:
                changes.append(
                    Change(
                        tier=2,
                        where=f"system[{i}] at offset {m.start()}",
                        what=(
                            f"{label} ({m.group(0)[:32]!r}) — every change to it "
                            "invalidates the whole prompt. Move it into a later "
                            "message, after the breakpoint."
                        ),
                        requires_consent=True,
                    )
                )
                break


def _copy(obj):
    """Deep-ish copy so callers' objects are never mutated."""
    try:
        return json.loads(json.dumps(obj, default=str)) if obj is not None else obj
    except (TypeError, ValueError):
        return obj
