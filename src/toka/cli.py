"""toka analyze — read agent transcripts, report where the money went."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .adapters import ADAPTERS, parse_all
from .analyze import analyze
from .ingest import find_transcripts
from .report import render


def default_transcript_root() -> Path:
    return Path.home() / ".claude" / "projects"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="toka",
        description="Measure where an agent's token spend actually goes.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="transcript file or directory (default: ~/.claude/projects)",
    )
    parser.add_argument(
        "--project",
        help="only sessions whose transcript directory contains this substring",
    )
    parser.add_argument("--top", type=int, default=10, help="worst-N sessions to list")
    parser.add_argument("--out", type=Path, help="also write the report here")
    args = parser.parse_args(argv)

    root = args.path or default_transcript_root()
    if not root.exists():
        print(f"error: {root} does not exist", file=sys.stderr)
        return 1

    if root.is_file():
        paths = [root]
    else:
        paths = find_transcripts(root)
        if args.project:
            needle = args.project.lower()
            paths = [p for p in paths if needle in str(p).lower()]

    if not paths:
        print(f"error: no .jsonl transcripts found under {root}", file=sys.stderr)
        return 1

    print(f"scanning {len(paths)} file(s)...", file=sys.stderr)
    requests, used, skipped = parse_all(paths)
    if not requests:
        print(
            "error: no billed model requests found. Supported formats: "
            + ", ".join(a.name for a in ADAPTERS),
            file=sys.stderr,
        )
        return 1

    for name, count in used.most_common():
        print(f"  {name}: {count} file(s)", file=sys.stderr)
    if skipped:
        print(f"  unrecognised: {len(skipped)} file(s)", file=sys.stderr)

    text = render(analyze(requests), top=args.top)
    print(text)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"\nwritten to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
