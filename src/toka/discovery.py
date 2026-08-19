"""Where agents keep their logs.

`toka compare` needs to find every agent on a machine without being told,
so this maps each known tool to the places it writes. Paths differ by OS;
missing ones are simply skipped.

Only tools that actually record token usage appear here. GitHub Copilot
is deliberately absent — it stores chat sessions but no token accounting
at all, so discovering it would produce an agent with no numbers.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Known:
    agent: str
    # Human-readable note shown when the agent is found but thin.
    note: str = ""


def _home() -> Path:
    return Path.home()


def _vscode_global_storage() -> list[Path]:
    """VS Code and its forks keep extension state per install."""
    home = _home()
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", home / "AppData/Roaming"))
        roots = [base / "Code", base / "Code - Insiders", base / "VSCodium"]
    elif sys.platform == "darwin":
        base = home / "Library/Application Support"
        roots = [base / "Code", base / "Code - Insiders", base / "VSCodium"]
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config"))
        roots = [base / "Code", base / "Code - Insiders", base / "VSCodium"]
    return [r / "User" / "globalStorage" for r in roots]


def candidates() -> dict[str, list[Path]]:
    """Known log locations per agent, existing ones only."""
    home = _home()
    found: dict[str, list[Path]] = {}

    def add(agent: str, path: Path) -> None:
        if path.exists():
            found.setdefault(agent, []).append(path)

    add("Direct API", Path(os.environ.get("TOKA_HOME") or home / ".toka"))
    add("Claude Code", home / ".claude" / "projects")
    add("Continue", home / ".continue" / "dev_data")

    for gs in _vscode_global_storage():
        add("Cline", gs / "saoudrizwan.claude-dev" / "tasks")
        add("Roo Code", gs / "rooveterinaryinc.roo-cline" / "tasks")

    # Aider writes into whatever project it ran in, so only the current
    # tree is discoverable without scanning the whole disk.
    for name in (".aider.chat.history.md", "aider.chat.history.md"):
        add("Aider", Path.cwd() / name)

    return found


NOTES: dict[str, str] = {
    "Continue": "telemetry is opt-in and stops silently; coverage is often partial",
    "Aider": "adapter built from docs, not verified against real traffic",
    "Roo Code": "shares Cline's format",
    "Direct API": "recorded by toka.log at your own call sites",
}
