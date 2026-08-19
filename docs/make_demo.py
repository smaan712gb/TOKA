"""Build a demo dashboard from synthetic data.

Never from real logs: a real dashboard carries actual spend and session
ids that can contain project names. This invents a plausible workload
instead, and deliberately includes the two cases that make Toka
different — a source whose cache writes are under-reported, and one that
logs no cache fields at all — so the screenshot shows the tool refusing
to report numbers it cannot support.
"""

import random
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, "src")

from toka.analyze import analyze  # noqa: E402
from toka.compare import Row  # noqa: E402
from toka.dashboard import render  # noqa: E402
from toka.record import Request  # noqa: E402

rng = random.Random(20260819)  # fixed, so the image is reproducible
START = datetime(2026, 8, 12, 9, 0, tzinfo=timezone.utc)


def session(name, turns, *, model, churn, gap_minutes=0.5, provider="anthropic"):
    """One conversation. `churn` is how often the prefix breaks and the
    whole context has to be written again."""
    out = []
    context = rng.randint(9_000, 22_000)
    t = START + timedelta(hours=rng.randint(0, 90))
    for i in range(turns):
        broke = i > 0 and rng.random() < churn
        if broke:
            write = context           # prefix invalidated: rewrite it all
            read = 0
        else:
            write = rng.randint(400, 2_500) if i else context
            read = context if i else 0
        context += rng.randint(700, 2_600)
        out.append(
            Request(
                source="demo",
                provider=provider,
                session=name,
                seq=i,
                timestamp=t.isoformat(),
                model=model,
                fresh_input=rng.randint(10, 90),
                cache_write_5m=write,
                cache_write_1h=0,
                cache_read=read,
                output=rng.randint(150, 900),
            )
        )
        # Real sessions have breaks in them — lunch, a meeting, overnight.
        # Rewrites after one are expiry, not churn, and Toka discounts them.
        if rng.random() < 0.06:
            t += timedelta(minutes=rng.uniform(20, 240))
        else:
            t += timedelta(minutes=gap_minutes * rng.uniform(0.4, 3.0))
    return out


# A well-behaved agent: long sessions, prefix rarely breaks.
tidy = []
for n in range(14):
    tidy += session(f"repo-work-{n}", rng.randint(60, 140),
                    model="claude-sonnet-4-5", churn=0.02)

# A busier one that rebuilds its prompt often — the case worth finding.
churny = []
for n in range(9):
    churny += session(f"task-{n}", rng.randint(40, 100),
                      model="claude-opus-5", churn=0.11)

# A router that reports reads but not writes: churn must be suppressed.
router = []
for n in range(6):
    reqs = session(f"route-{n}", rng.randint(30, 70),
                   model="claude-sonnet-4-5", churn=0.05)
    for r in reqs:
        r.cache_read += r.cache_write_5m
        r.cache_write_5m = 0          # the write accounting is missing
    router += reqs

# A source with no cache accounting at all: nothing may be called waste.
blind = []
for n in range(4):
    reqs = session(f"assist-{n}", rng.randint(8, 20),
                   model="claude-haiku-4-5", churn=0.1)
    for r in reqs:
        r.fresh_input += r.cache_read + r.cache_write_5m
        r.cache_read = r.cache_write_5m = 0
        r.cache_visible = False
    blind += reqs

rows = [
    Row(agent="Claude Code", report=analyze(tidy), files=14),
    Row(agent="Cline", report=analyze(churny), files=9),
    Row(agent="Roo Code", report=analyze(router), files=6),
    Row(agent="Continue", report=analyze(blind), files=4),
]
rows.sort(key=lambda r: r.prompt_tokens, reverse=True)

out = sys.argv[1] if len(sys.argv) > 1 else "demo.html"
open(out, "w", encoding="utf-8").write(render(rows))

for r in rows:
    hit = "not logged" if r.hit_rate is None else f"{r.hit_rate:.1f}%"
    print(f"{r.agent:<14} {r.sessions:>4} sessions  hit {hit:>10}  {r.confidence}")
print("\nwrote", out)
