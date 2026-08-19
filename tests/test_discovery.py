"""Finding agents, including ones nobody has written an adapter for.

A built-in list of locations is wrong the week after it ships. These
tests are about the three ways a user extends it without waiting for a
release, and about the built-ins staying honest in the meantime.
"""

import json
import os
from pathlib import Path

from toka import discovery


def test_a_user_can_add_an_agent_without_a_code_change(tmp_path, monkeypatch):
    """The whole point of the config file: a tool Toka has never heard of
    becomes a row in the comparison."""
    logs = tmp_path / "my-agent-logs"
    logs.mkdir()
    config = tmp_path / "agents.json"
    config.write_text(json.dumps({"My Agent": str(logs)}), encoding="utf-8")

    monkeypatch.setattr(discovery, "toka_home", lambda: tmp_path)
    found = discovery.candidates()
    assert logs in found["My Agent"]


def test_a_config_entry_may_list_several_paths(tmp_path, monkeypatch):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    (tmp_path / "agents.json").write_text(
        json.dumps({"Split": [str(first), str(second)]}), encoding="utf-8"
    )

    monkeypatch.setattr(discovery, "toka_home", lambda: tmp_path)
    assert discovery.candidates()["Split"] == [first, second]


def test_config_entries_use_the_same_templates_as_the_builtins(tmp_path, monkeypatch):
    """Otherwise the documented syntax only works for us."""
    target = tmp_path / "home" / "custom"
    target.mkdir(parents=True)
    (tmp_path / "agents.json").write_text(
        json.dumps({"Templated": "{home}/custom"}), encoding="utf-8"
    )

    monkeypatch.setattr(discovery, "toka_home", lambda: tmp_path)
    monkeypatch.setattr(discovery, "_home", lambda: tmp_path / "home")
    assert target in discovery.candidates()["Templated"]


def test_a_broken_config_warns_rather_than_crashing_or_going_quiet(
    tmp_path, monkeypatch, capsys
):
    """Crashing loses the other agents; silence leaves the user wondering
    why theirs never shows up."""
    (tmp_path / "agents.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(discovery, "toka_home", lambda: tmp_path)

    found = discovery.candidates()
    assert isinstance(found, dict)
    assert "agents.json" in capsys.readouterr().err


def test_the_env_var_splits_the_way_this_platform_splits_path(tmp_path, monkeypatch):
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv(discovery.SCAN_ENV, os.pathsep.join([str(first), str(second)]))
    monkeypatch.setattr(discovery, "toka_home", lambda: tmp_path / "none")

    assert discovery.candidates()["Scanned"] == [first, second]


def test_user_configuration_extends_the_builtins_rather_than_replacing_them(
    tmp_path, monkeypatch
):
    """Adding your own tool must not cost you Claude Code."""
    logs = tmp_path / "mine"
    logs.mkdir()
    monkeypatch.setenv(discovery.SCAN_ENV, str(logs))

    claude = tmp_path / "fakehome" / ".claude" / "projects"
    claude.mkdir(parents=True)
    monkeypatch.setattr(discovery, "_home", lambda: tmp_path / "fakehome")
    monkeypatch.setattr(discovery, "toka_home", lambda: tmp_path / "none")

    found = discovery.candidates()
    assert "Scanned" in found and "Claude Code" in found


def test_every_builtin_location_is_a_valid_template():
    """A typo in a template would silently drop an agent from discovery,
    and nothing else would ever fail."""
    for agent, template in discovery.builtin_locations():
        assert agent
        discovery.expand(template)  # raises on an unknown placeholder


def test_vs_code_forks_are_covered_by_data_not_by_code():
    """Cursor and Windsurf host the same extensions as VS Code, so they
    cost an entry in a tuple, not a branch."""
    templates = [t for _agent, t in discovery.builtin_locations()]
    for editor in ("Cursor", "Windsurf", "Code"):
        assert any(f"/{editor}/" in t for t in templates), editor
    assert any("saoudrizwan.claude-dev" in t for t in templates)


def test_only_existing_paths_are_offered(tmp_path, monkeypatch):
    monkeypatch.setenv(discovery.SCAN_ENV, str(tmp_path / "does-not-exist"))
    monkeypatch.setattr(discovery, "toka_home", lambda: tmp_path / "none")
    assert "Scanned" not in discovery.candidates()


def test_explicit_extra_paths_are_merged_in(tmp_path):
    logs = tmp_path / "adhoc"
    logs.mkdir()
    found = discovery.candidates({"Ad hoc": [logs]})
    assert found["Ad hoc"] == [logs]
