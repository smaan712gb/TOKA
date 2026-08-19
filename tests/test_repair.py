"""Repair safety.

The one property that matters: a tier-1 repair must not change a single
character the model reads. Everything else here exists to make sure that
property cannot silently regress.
"""

import pytest

from toka.guard import PrefixGuard, render
from toka.repair import repair, repair_safely, verify

SYSTEM = "You are a helpful assistant. " + ("context " * 60)
TOOLS = [
    {"name": "get_weather", "input_schema": {"type": "object"}, "description": "W"},
    {"description": "S", "name": "search", "input_schema": {"type": "object"}},
]
MESSAGES = [{"role": "user", "content": "hi"}]


def test_tier1_never_changes_what_the_model_reads():
    """The safety invariant, checked rather than claimed."""
    original = {"system": SYSTEM, "tools": TOOLS, "messages": MESSAGES}
    result = repair(**original)
    assert verify(original, result)


def test_repair_safely_raises_rather_than_returning_unsafe_output():
    """Call sites that auto-apply need a hard failure, not a flag."""
    original = {"system": SYSTEM, "tools": TOOLS, "messages": MESSAGES}
    result = repair_safely(**original)  # must not raise
    assert verify(original, result)


def test_tool_key_order_is_normalised():
    result = repair(system=SYSTEM, tools=TOOLS, messages=MESSAGES)
    for tool in result.tools:
        assert list(tool.keys()) == sorted(tool.keys())
    assert any("key order" in c.what for c in result.applied)


def test_normalising_key_order_stops_the_guard_firing():
    """The actual payoff: two turns whose tools differ only by key order
    must stop registering as a break once repaired."""
    a = [{"name": "t", "description": "d", "input_schema": {}}]
    b = [{"input_schema": {}, "description": "d", "name": "t"}]

    guard = PrefixGuard()
    guard.check(system=SYSTEM, tools=repair(tools=a, system=SYSTEM).tools)
    report = guard.check(system=SYSTEM, tools=repair(tools=b, system=SYSTEM).tools)
    assert report.stable


def test_cache_control_is_placed_on_the_last_system_block():
    result = repair(system=SYSTEM, tools=TOOLS, messages=MESSAGES)
    assert result.system[-1]["cache_control"] == {"type": "ephemeral"}
    assert any("cache_control" in c.what for c in result.applied)


def test_existing_cache_control_is_left_alone():
    """The caller placed it deliberately; second-guessing that could move
    the breakpoint somewhere worse."""
    system = [
        {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "b"},
    ]
    result = repair(system=system, tools=None, messages=MESSAGES)
    assert result.system[0].get("cache_control")
    assert not result.system[1].get("cache_control")


def test_volatile_content_is_proposed_never_applied():
    """Hoisting a timestamp is the biggest single win and the one change
    that reorders text the model saw. It must not happen automatically."""
    system = "Current date: 2026-08-19T10:33:01Z\nYou are helpful."
    result = repair(system=system, tools=None, messages=MESSAGES)

    assert result.proposed, "volatile content should be flagged"
    hoist = result.proposed[0]
    assert hoist.tier == 2 and hoist.requires_consent
    assert "timestamp" in hoist.what

    # and the text is untouched
    assert "2026-08-19T10:33:01Z" in result.system[0]["text"]


def test_inputs_are_never_mutated():
    tools = [{"name": "t", "description": "d"}]
    system = [{"type": "text", "text": "s"}]
    before_tools = [dict(t) for t in tools]
    before_system = [dict(b) for b in system]

    repair(system=system, tools=tools, messages=MESSAGES)

    assert tools == before_tools
    assert system == before_system


def test_repairing_nothing_reports_nothing():
    result = repair(system=None, tools=None, messages=None)
    assert result.changes == []
    assert "already stable" in result.explain()


def test_verify_catches_a_repair_that_alters_text():
    """Guards the guard: if a future change makes tier 1 rewrite content,
    verify must fail rather than wave it through."""
    original = {"system": "abc", "tools": None, "messages": None}
    result = repair(**original)
    result.system = [{"type": "text", "text": "TAMPERED"}]
    assert not verify(original, result)
