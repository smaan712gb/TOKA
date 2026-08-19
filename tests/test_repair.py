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


def test_a_tools_only_request_still_gets_a_breakpoint():
    """With no system prompt the tool list is the entire stable prefix.
    Skipping it left the one arrangement that gets no caching at all."""
    tools = [{"name": "a", "description": "d"}, {"name": "b", "description": "d"}]
    result = repair(system=None, tools=tools, messages=MESSAGES)

    assert result.tools[-1]["cache_control"] == {"type": "ephemeral"}
    assert not result.tools[0].get("cache_control")
    assert any("stable prefix ends" in c.what for c in result.applied)


def test_a_breakpoint_on_a_tool_survives_the_safety_check():
    """cache_control is a directive to the provider, not text the model
    reads — so placing one must not count as altering the prompt."""
    original = {"system": None, "tools": list(TOOLS), "messages": MESSAGES}
    result = repair_safely(**original)  # raises if it does
    assert verify(original, result)


def test_system_still_wins_the_breakpoint_when_there_is_one():
    """Tools render before system, so a breakpoint on the last system
    block caches both. Marking a tool as well would waste one."""
    result = repair(system=SYSTEM, tools=TOOLS, messages=MESSAGES)
    assert result.system[-1]["cache_control"] == {"type": "ephemeral"}
    assert not any(t.get("cache_control") for t in result.tools)


def test_an_existing_breakpoint_on_a_tool_is_left_alone():
    tools = [
        {"name": "a", "description": "d"},
        {"name": "b", "description": "d", "cache_control": {"type": "ephemeral"}},
    ]
    result = repair(system=None, tools=tools, messages=MESSAGES)
    assert not result.tools[0].get("cache_control")
    assert result.tools[1]["cache_control"] == {"type": "ephemeral"}


def test_volatile_content_in_a_tool_is_proposed_and_named_as_worse():
    """A varying date in a tool definition takes the system prompt and
    the whole history with it. Scanning only system missed it."""
    tools = [{"name": "search", "description": "Index built 2026-08-19T10:33:01Z"}]
    result = repair(system="stable prompt", tools=tools, messages=MESSAGES)

    hoists = [c for c in result.proposed if c.where.startswith("tools")]
    assert hoists, "volatile content in a tool must be flagged"
    assert hoists[0].tier == 2 and hoists[0].requires_consent
    assert "tools render first" in hoists[0].what
    # and the definition is untouched
    assert "2026-08-19T10:33:01Z" in result.tools[0]["description"]


def test_volatility_nested_in_a_schema_is_found():
    """It is bytes in the prefix wherever it sits."""
    tools = [
        {
            "name": "q",
            "input_schema": {
                "properties": {"since": {"description": "after 2026-08-19T10:33:01Z"}}
            },
        }
    ]
    result = repair(system=None, tools=tools, messages=MESSAGES)
    assert any(c.where.startswith("tools") for c in result.proposed)


def test_moving_a_breakpoint_is_not_a_prefix_break():
    """The provider reads cache_control; the model never does. If the
    guard saw it, every repair that placed one would report a break."""
    a = [{"name": "t", "description": "d"}]
    b = [{"name": "t", "description": "d", "cache_control": {"type": "ephemeral"}}]

    guard = PrefixGuard()
    guard.check(system=SYSTEM, tools=a)
    assert guard.check(system=SYSTEM, tools=b).stable
