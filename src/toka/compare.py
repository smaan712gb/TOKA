"""Cross-agent comparison — the efficiency leaderboard.

Runs the same analysis over every agent found on the machine and puts the
results side by side. The interesting number is not what any one agent
costs; it is how differently two agents building prompts for the same
kind of work handle the cache.

What this can and cannot compare
--------------------------------
**Cache hit rate is comparable.** It is a property of how an agent
constructs its prompts — whether the prefix stays stable across turns —
not of what task it was doing. An agent with a 60% hit rate is leaving
money on the table regardless of workload.

**Absolute cost is not comparable.** Different agents ran different work
for different amounts of time. A larger bill means more usage, not worse
engineering.

**Recoverable %% is comparable with care.** It is scale-free, which helps,
but it depends on what each source is able to report. An agent that
cannot see cache writes will show less recoverable waste than one that
can — not because it wastes less, but because less is visible. The
`confidence` column says which is which, and it is not decoration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .adapters import parse_all
from .analyze import Report, analyze
from .discovery import NOTES, candidates
from .ingest import find_transcripts


@dataclass
class Row:
    agent: str
    report: Report
    files: int

    @property
    def sessions(self) -> int:
        return len(self.report.sessions)

    @property
    def requests(self) -> int:
        return self.report.total_requests

    @property
    def prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self.report.sessions.values())

    @property
    def cache_blind(self) -> bool:
        """True when the source logs no cache fields at all."""
        return bool(self.report.blind_sessions) and not any(
            s.cache_visible for s in self.report.sessions.values()
        )

    @property
    def hit_rate(self) -> float | None:
        """None when unmeasured — never 0.0.

        A blind source computes to 0% because its cache_read is always
        zero, which reads as a catastrophic score rather than an absent
        one. Ranking that against a real measurement would put the
        best-behaved agent bottom of the table on no evidence.
        """
        if self.cache_blind:
            return None
        return 100.0 * self.report.overall_hit_rate

    @property
    def confidence(self) -> str:
        """What the source lets us see. Drives how far to trust the row."""
        if self.cache_blind:
            return "no cache data"
        if self.report.blind_sessions:
            return "partial — some sources blind"
        if not self.report.write_accounting_reliable:
            return "miss only — writes unreported"
        return "full"

    @property
    def recoverable_pct(self) -> float:
        return self.report.recoverable_pct


def collect(extra: dict[str, Path] | None = None) -> list[Row]:
    """Analyse every discoverable agent. Silently skips empty ones."""
    rows: list[Row] = []
    found = candidates({k: [v] for k, v in (extra or {}).items()})

    for agent, paths in sorted(found.items()):
        files: list[Path] = []
        for p in paths:
            files.extend([p] if p.is_file() else find_transcripts(p))
        if not files:
            continue
        records, used, _ = parse_all(files)
        if not records:
            continue
        rows.append(Row(agent=agent, report=analyze(records), files=sum(used.values())))

    rows.sort(key=lambda r: r.prompt_tokens, reverse=True)
    return rows


def render(rows: list[Row]) -> str:
    from .report import _tokens

    if not rows:
        return (
            "No agent logs found.\n\n"
            "Toka looks in the default locations for Claude Code, Cline,\n"
            "Roo Code, Continue and Aider. Point it at a directory instead:\n"
            "  toka --compare /your/logs"
        )

    out: list[str] = []
    add = out.append
    add("=" * 78)
    add("TOKA — agent efficiency comparison")
    add("=" * 78)
    add("")
    add(
        f"  {'agent':<14} {'sessions':>9} {'requests':>9} {'prompt tok':>11} "
        f"{'hit rate':>9} {'recoverable':>12}"
    )
    add("  " + "-" * 74)
    for r in rows:
        hit = "not logged" if r.hit_rate is None else f"{r.hit_rate:.1f}%"
        rec = "—" if r.cache_blind else f"{r.recoverable_pct:.1f}%"
        add(
            f"  {r.agent:<14} {r.sessions:>9,} {r.requests:>9,} "
            f"{_tokens(r.prompt_tokens):>11} {hit:>9} {rec:>12}"
        )
    add("")
    add("  confidence")
    add("  " + "-" * 74)
    for r in rows:
        note = NOTES.get(r.agent, "")
        line = f"  {r.agent:<14} {r.confidence}"
        if note:
            line += f" — {note}"
        add(line)
    add("")

    # Only agents whose caching was actually observed can be ranked.
    measured = [r for r in rows if r.hit_rate is not None]
    if len(measured) < 2:
        return "\n".join(out + _legend())
    best = max(measured, key=lambda r: r.hit_rate)
    worst = min(measured, key=lambda r: r.hit_rate)
    if best.agent != worst.agent and best.hit_rate - worst.hit_rate > 5:
        add(
            f"  Spread: {best.agent} holds {best.hit_rate:.1f}% of its prompt "
            f"tokens in cache;"
        )
        add(
            f"          {worst.agent} holds {worst.hit_rate:.1f}%. Same job, "
            f"{best.hit_rate - worst.hit_rate:.0f} points apart."
        )
        add("")

    return "\n".join(out + _legend())


def _legend() -> list[str]:
    return [
        "  Reading this table",
        "  " + "-" * 74,
        "  Hit rate is comparable — it reflects how an agent builds prompts,",
        "  not what it was asked to do.",
        "",
        "  Cost and token totals are NOT comparable — they reflect how much",
        "  each agent was used, not how well it was engineered.",
        "",
        "  'not logged' is not a zero. An agent that records no cache fields",
        "  is unmeasured, not failing — it is excluded from ranking entirely.",
        "",
        "  Recoverable % depends on what each source reports. An agent that",
        "  cannot see its own cache writes shows less waste than one that can,",
        "  because less is visible. Check the confidence column first.",
    ]
