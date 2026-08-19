"""Prefix guard behaviour.

The guard's whole value is that it points at the right thing. A guard
that cries wolf on normal conversation growth, or that blames the wrong
segment, is worse than nothing — so most of these tests are about what it
must *not* report.
"""

from toka.guard import PrefixGuard, render, sorted_keys_warning

SYSTEM = "You are a helpful assistant. " + ("x" * 500)
TOOLS = [
    {"name": "get_weather", "description": "Get weather", "input_schema": {}},
    {"name": "search", "description": "Search", "input_schema": {}},
]


def _msgs(*texts):
    roles = ("user", "assistant")
    return [
        {"role": roles[i % 2], "content": t} for i, t in enumerate(texts)
    ]


def test_first_turn_is_stable_with_nothing_cached():
    guard = PrefixGuard()
    report = guard.check(system=SYSTEM, tools=TOOLS, messages=_msgs("hi"))
    assert report.stable
    assert report.reusable_chars == 0
    assert "nothing cached" in report.explain()


def test_appending_a_turn_does_not_break_the_prefix():
    """Conversations grow by appending. If that registered as a break the
    guard would fire on every single turn."""
    guard = PrefixGuard()
    guard.check(system=SYSTEM, tools=TOOLS, messages=_msgs("hi"))
    report = guard.check(
        system=SYSTEM, tools=TOOLS, messages=_msgs("hi", "hello", "how are you")
    )
    assert report.stable
    assert report.reusable_chars > 500


def test_changed_system_prompt_is_attributed_to_system():
    guard = PrefixGuard()
    guard.check(system=SYSTEM, tools=TOOLS, messages=_msgs("hi"))
    report = guard.check(
        system=SYSTEM + " Also be terse.", tools=TOOLS, messages=_msgs("hi")
    )
    assert not report.stable
    assert report.break_.label.startswith("system")
    assert "history was lost" in " ".join(report.notes)


def test_changed_tool_invalidates_everything():
    guard = PrefixGuard()
    guard.check(system=SYSTEM, tools=TOOLS, messages=_msgs("hi"))
    changed = [dict(TOOLS[0], description="Get the weather now"), TOOLS[1]]
    report = guard.check(system=SYSTEM, tools=changed, messages=_msgs("hi"))
    assert not report.stable
    assert report.break_.label.startswith("tools")
    assert report.reusable_chars == 0  # tools render first
    assert report.invalidated_pct > 99


def test_timestamp_in_system_prompt_is_named_as_the_cause():
    """The payoff case — this is the single most common invalidator."""
    guard = PrefixGuard()
    guard.check(system="Current date: 2026-08-19T10:33:01Z\nYou are helpful.")
    report = guard.check(system="Current date: 2026-08-19T10:34:15Z\nYou are helpful.")
    assert not report.stable
    assert report.break_.cause == "looks like a timestamp"
    assert "timestamp" in report.explain()


def test_key_order_alone_is_not_reported_as_a_content_change():
    """Serialisation order is a real cache break, but it is not a prompt
    change — reporting it here would point the user at the wrong fix."""
    guard = PrefixGuard()
    guard.check(system=SYSTEM, tools=[{"name": "a", "description": "d"}])
    report = guard.check(system=SYSTEM, tools=[{"description": "d", "name": "a"}])
    assert report.stable


def test_unsorted_tool_keys_are_flagged_separately():
    assert sorted_keys_warning([{"name": "a", "description": "d"}]) is not None
    assert sorted_keys_warning([{"description": "d", "name": "a"}]) is None
    assert sorted_keys_warning([]) is None


def test_render_order_is_tools_then_system_then_messages():
    segments = render(system="S", tools=[{"name": "t"}], messages=_msgs("m"))
    labels = [s.label for s in segments]
    assert labels[0].startswith("tools")
    assert labels[1].startswith("system")
    assert labels[2].startswith("messages")


def test_editing_history_is_attributed_to_messages_not_system():
    guard = PrefixGuard()
    guard.check(system=SYSTEM, messages=_msgs("hi", "hello"))
    report = guard.check(system=SYSTEM, messages=_msgs("hi", "HELLO THERE"))
    assert not report.stable
    assert report.break_.label.startswith("messages[1]")
    assert "Append rather than edit" in report.explain()


def test_guard_is_per_conversation():
    """Two conversations sharing one guard would report phantom breaks."""
    a, b = PrefixGuard(), PrefixGuard()
    a.check(system="conversation A")
    report = b.check(system="conversation B")
    assert report.stable  # b's first turn, not a break against A
