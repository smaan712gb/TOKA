"""Where agents keep their logs — and how to teach Toka about new ones.

`toka --compare` has to find agents without being told, which means
knowing some locations up front. But a built-in list is a list that is
wrong the week after it ships: new agents appear, editors fork, and
people put their logs wherever they like.

So the built-in locations are *data*, not code, and they are only the
starting point. Three things extend them, none of which require touching
this file:

  TOKA_SCAN     one or more paths, separated the way your platform
                separates PATH entries. Highest precedence, per-run.
  agents.json   a file in `~/.toka` mapping a display name to one or
                more paths. Persistent, and shareable across a team.
  --scan PATH   ad hoc, for one command.

Anything found that way is parsed by the same adapters as everything
else, and the `generic` adapter reads any JSON with token counts in it at
any depth — so a tool nobody has ever written an adapter for still
produces numbers as long as it logs its usage somewhere.

Locations are templates so the same entry works on every platform:

  {home}          the user's home directory
  {app_support}   where desktop apps keep per-user state
                  (%APPDATA%, ~/Library/Application Support, ~/.config)
  {toka_home}     where `toka.log` writes, TOKA_HOME or ~/.toka
  {cwd}           the current directory

Only tools that record token usage are listed. GitHub Copilot is
deliberately absent: it stores chat but no token accounting at all, so
discovering it would produce an agent with no numbers.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CONFIG_NAME = "agents.json"
SCAN_ENV = "TOKA_SCAN"

# Editors that host agent extensions. VS Code forks all keep the same
# layout, so a fork costs one entry here rather than a code change.
EDITORS = (
    "Code",
    "Code - Insiders",
    "VSCodium",
    "Cursor",
    "Windsurf",
    "Trae",
)

# Agent extensions, by the globalStorage folder their publisher uses.
EXTENSIONS: tuple[tuple[str, str], ...] = (
    ("Cline", "saoudrizwan.claude-dev/tasks"),
    ("Roo Code", "rooveterinaryinc.roo-cline/tasks"),
    ("Kilo Code", "kilocode.kilo-code/tasks"),
)

# Everything else, as name -> path template.
BUILTIN: tuple[tuple[str, str], ...] = (
    ("Direct API", "{toka_home}"),
    ("Claude Code", "{home}/.claude/projects"),
    ("Claude Desktop", "{app_support}/Claude/local-agent-mode-sessions"),
    ("Continue", "{home}/.continue/dev_data"),
    ("Gemini CLI", "{home}/.gemini/tmp"),
    ("Codex CLI", "{home}/.codex/sessions"),
    # Aider writes into whatever project it ran in, so only the current
    # tree is discoverable without scanning the whole disk.
    ("Aider", "{cwd}/.aider.chat.history.md"),
    ("Aider", "{cwd}/aider.chat.history.md"),
)

NOTES: dict[str, str] = {
    "Continue": "telemetry is opt-in and stops silently; coverage is often partial",
    "Aider": "adapter built from docs, not verified against real traffic",
    "Roo Code": "shares Cline's format",
    "Kilo Code": "shares Cline's format",
    "Direct API": "recorded by toka.log at your own call sites",
    "Claude Desktop": "agent-mode sessions; same format as Claude Code",
}


def _home() -> Path:
    return Path.home()


def toka_home() -> Path:
    return Path(os.environ.get("TOKA_HOME") or (_home() / ".toka"))


def _app_support() -> Path:
    """Where desktop apps keep per-user state, per platform."""
    home = _home()
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA") or home / "AppData/Roaming")
    if sys.platform == "darwin":
        return home / "Library" / "Application Support"
    return Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config")


def expand(template: str) -> Path:
    """Resolve a location template against this machine."""
    return Path(
        template.format(
            home=_home(),
            app_support=_app_support(),
            toka_home=toka_home(),
            cwd=Path.cwd(),
        )
    )


def builtin_locations() -> list[tuple[str, str]]:
    """Every known location, editors expanded, as (agent, template)."""
    out = list(BUILTIN)
    for editor in EDITORS:
        for agent, suffix in EXTENSIONS:
            out.append((agent, f"{{app_support}}/{editor}/User/globalStorage/{suffix}"))
    return out


def configured_locations(config: Path | None = None) -> list[tuple[str, str]]:
    """User-declared locations from `~/.toka/agents.json`.

    Accepts a string or a list of strings per agent, and the same
    templates the built-ins use. A malformed file is reported and skipped
    rather than crashing a run — but it is never silently ignored, or the
    user is left wondering why their agent never appears.
    """
    path = config or (toka_home() / CONFIG_NAME)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: ignoring {path}: {exc}", file=sys.stderr)
        return []

    if not isinstance(raw, dict):
        print(f"warning: {path} should map an agent name to paths", file=sys.stderr)
        return []

    out: list[tuple[str, str]] = []
    for agent, value in raw.items():
        values = [value] if isinstance(value, str) else value
        if not isinstance(values, list):
            print(f"warning: {path}: {agent!r} is neither a path nor a list", file=sys.stderr)
            continue
        out.extend((str(agent), str(v)) for v in values)
    return out


def env_locations() -> list[tuple[str, str]]:
    """Paths from TOKA_SCAN, split the way this platform splits PATH."""
    raw = os.environ.get(SCAN_ENV)
    if not raw:
        return []
    return [("Scanned", part) for part in raw.split(os.pathsep) if part.strip()]


def candidates(extra: dict[str, list[Path]] | None = None) -> dict[str, list[Path]]:
    """Known log locations per agent, existing ones only.

    Later sources win the right to add paths, never to remove them, so a
    user's configuration extends the built-ins rather than replacing them
    — someone adding their own tool does not lose Claude Code.
    """
    found: dict[str, list[Path]] = {}

    def add(agent: str, path: Path) -> None:
        try:
            if not path.exists():
                return
        except OSError:  # unreadable or malformed path: not a candidate
            return
        paths = found.setdefault(agent, [])
        if path not in paths:
            paths.append(path)

    for agent, template in (
        builtin_locations() + configured_locations() + env_locations()
    ):
        try:
            add(agent, expand(template))
        except (KeyError, IndexError, ValueError):
            print(
                f"warning: skipping malformed location for {agent!r}: {template!r}",
                file=sys.stderr,
            )

    for agent, paths in (extra or {}).items():
        for path in paths:
            add(agent, Path(path))

    return found
