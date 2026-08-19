"""Transcript discovery.

Parsing lives in `toka.adapters`; this module only finds candidate files
and re-exports the pieces callers expect.
"""

from __future__ import annotations

from pathlib import Path

from .adapters import parse_all
from .record import Request

# Formats worth sniffing. Everything else is skipped without reading.
SUFFIXES = (".jsonl", ".ndjson", ".json", ".md")

# Directories that never contain transcripts but are expensive to walk.
SKIP_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "venv", "dist", "build"}


def find_transcripts(root: Path) -> list[Path]:
    """Candidate transcript files under `root`, newest first."""
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUFFIXES:
            continue
        if SKIP_DIRS & set(path.parts):
            continue
        files.append(path)
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_all(paths: list[Path]) -> list[Request]:
    records, _, _ = parse_all(paths)
    return records


__all__ = ["Request", "find_transcripts", "read_all", "parse_all"]
