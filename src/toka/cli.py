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


def build_parser() -> argparse.ArgumentParser:
    """The command line, as one inspectable object.

    Separated from `main` so the docs test can ask the real parser what
    flags exist instead of trusting the README to have kept up.
    """
    parser = argparse.ArgumentParser(
        prog="toka",
        description="Measure where an agent's token spend actually goes.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="analyse every agent found on this machine, side by side",
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
    parser.add_argument(
        "--html",
        type=Path,
        help="also write a dashboard here, for people who don't read terminals",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.compare:
        from .compare import collect, render as render_compare

        extra = {"(supplied)": args.path} if args.path else None
        print("discovering agents...", file=sys.stderr)
        rows = collect(extra)
        print(render_compare(rows))
        if args.html:
            _write_html(args.html, rows)
        return 0

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

    report = analyze(requests)
    text = render(report, top=args.top)
    print(text)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"\nwritten to {args.out}", file=sys.stderr)

    if args.html:
        # One agent is still a comparison of one — the dashboard reads a
        # list of rows either way, so there is no second rendering path to
        # keep in step with this one.
        from .compare import Row

        agent = used.most_common(1)[0][0] if used else "this agent"
        _write_html(args.html, [Row(agent=agent, report=report, files=len(paths))])
    return 0


def _write_html(path: Path, rows) -> None:
    from .dashboard import render as render_html

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(rows), encoding="utf-8")
    print(f"dashboard written to {path}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
