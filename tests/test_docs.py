"""Documentation, checked against the code rather than by eye.

Toka's whole argument is that a number you cannot substantiate should not
be printed. Documentation is subject to the same rule: a README that
claims a flag, an adapter, or a version the code does not have is a
confident wrong statement, and it is the kind that survives for months
because nothing fails when it drifts.

So the drift is a test failure. Every claim checked here is one that has
already gone stale at least once.
"""

import re
import tomllib
from pathlib import Path

import pytest

from toka import __version__
from toka.adapters import ADAPTERS
from toka.cli import build_parser

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _section(heading: str) -> str:
    """The README text under one heading, up to the next one.

    Sliced by heading rather than by a marker that happens to follow it,
    so restructuring the README does not make these tests error out
    instead of reporting what they check.
    """
    start = README.index(heading) + len(heading)
    rest = README[start:]
    end = rest.find("\n## ")
    return rest if end == -1 else rest[:end]


def test_the_package_version_matches_itself():
    """pyproject and __init__ have drifted apart twice already — once
    shipping 0.8.0 that reported itself as 0.7.0."""
    assert PYPROJECT["project"]["version"] == __version__


def test_every_cli_flag_is_documented():
    """A flag nobody can find is a flag that does not exist."""
    flags = {
        opt
        for action in build_parser()._actions
        for opt in action.option_strings
        if opt.startswith("--") and opt != "--help"
    }
    missing = sorted(f for f in flags if f not in README)
    assert not missing, f"undocumented flags: {missing}"


def test_every_documented_command_actually_parses():
    """The README's usage block is executable documentation, so it is run
    against the real parser instead of being read."""
    commands = re.findall(r"^\s*(toka(?: [^\n#]*)?)", README, flags=re.M)
    assert commands, "the README should show at least one invocation"

    parser = build_parser()
    for command in commands:
        argv = command.split()[1:]
        argv = [a for a in argv if not a.startswith("#")]
        try:
            parser.parse_args(argv)
        except SystemExit:  # pragma: no cover - only on a bad doc
            pytest.fail(f"README documents an invalid invocation: {command!r}")


def test_every_adapter_appears_in_the_supported_agents_table():
    """A new adapter that nobody knows about helps nobody."""
    table = _section("## Supported agents")
    missing = [a.name for a in ADAPTERS if f"`{a.name}`" not in table]
    assert not missing, f"adapters missing from the README table: {missing}"


def test_the_table_does_not_claim_adapters_that_do_not_exist():
    """The other direction — a row for an adapter that was renamed or
    removed reads as coverage the tool does not have."""
    table = _section("## Supported agents")
    claimed = set(re.findall(r"^\| `([a-z0-9-]+)`", table, flags=re.M))
    real = {a.name for a in ADAPTERS}
    assert claimed <= real, f"README claims adapters that do not exist: {claimed - real}"


def test_the_readme_documents_every_public_export():
    """`__all__` is the promised surface; an undocumented export is a
    feature the user has no way to discover."""
    import toka

    # Whole-word, not substring: a short export like `log` matches inside
    # "logs" and would pass without ever being documented.
    undocumented = [
        name
        for name in toka.__all__
        if not name.startswith("__")
        and not re.search(rf"\b{re.escape(name)}\b", README)
    ]
    assert not undocumented, f"public exports missing from the README: {undocumented}"
