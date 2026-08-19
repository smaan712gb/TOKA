"""Turn raw request records into a waste report.

Two measurements carry the analysis:

1. Cache miss.  Tokens billed as fresh input (1x) that a warm cache would
   have served at 0.1x.  The first request in a session is excluded — it
   has no cache to hit yet, so its fresh input is unavoidable.

2. Write amplification.  In a well-structured session the context grows
   monotonically and each token is written to cache once, so cumulative
   cache writes should land near the session's peak context size.  Writing
   several times that means the cached prefix kept breaking and was
   re-written from scratch.  Those excess writes cost 1.25x-2.0x where a
   read would have cost 0.1x.

Both are lower bounds on recoverable spend, not estimates of total waste.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from .pricing import (
    CACHE_READ_MULT,
    CACHE_WRITE_5M_MULT,
    resolve,
)
from .record import Request

# A cache entry is gone once its TTL lapses, so a rewrite after a longer
# idle gap is expiry, not churn. Only churn is recoverable by better
# context construction.
TTL_5M_SECONDS = 5 * 60
TTL_1H_SECONDS = 60 * 60


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass
class SessionStats:
    session: str
    requests: int = 0
    model: str | None = None
    provider: str = "anthropic"
    fresh_input: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0
    cache_read: int = 0
    output: int = 0
    thinking: int = 0
    peak_context: int = 0
    # Fresh input excluding the session's first request.
    missable_input: int = 0
    # Cache writes issued after the prior entry had already expired.
    expiry_writes: int = 0
    # False when the source reports no cache accounting, so a "miss"
    # cannot be distinguished from an unlogged hit.
    cache_visible: bool = True
    cost: float = 0.0

    @property
    def total_writes(self) -> int:
        return self.cache_write_5m + self.cache_write_1h

    @property
    def write_amplification(self) -> float:
        """Cumulative cache writes / peak context. 1.0 is ideal."""
        if self.peak_context == 0:
            return 0.0
        return self.total_writes / self.peak_context

    @property
    def excess_writes(self) -> int:
        """Writes beyond one full pass over peak context, after discounting
        rewrites that followed a TTL lapse. What remains is prefix churn:
        the cache was still live and got invalidated anyway."""
        return max(0, self.total_writes - self.peak_context - self.expiry_writes)

    @property
    def prompt_tokens(self) -> int:
        return self.fresh_input + self.total_writes + self.cache_read

    @property
    def cache_hit_rate(self) -> float:
        if self.prompt_tokens == 0:
            return 0.0
        return self.cache_read / self.prompt_tokens


@dataclass
class Report:
    sessions: dict[str, SessionStats] = field(default_factory=dict)
    # Anthropic ids we did not recognise exactly, priced by prefix match.
    approximated_models: set[str] = field(default_factory=set)
    # Models with no rate card at all — excluded from every dollar figure.
    unpriced_models: set[str] = field(default_factory=set)
    unpriced_requests: int = 0
    unpriced_tokens: int = 0
    # Sessions from sources that log no cache accounting at all.
    blind_sessions: int = 0
    blind_tokens: int = 0

    @property
    def write_accounting_reliable(self) -> bool:
        """Whether cache-write counts can be trusted.

        Reads require a prior write, so reads vastly exceeding writes
        means the source under-reports writes rather than that the
        prefix was stable. Some agents populate `cacheReads` but leave
        `cacheWrites` at zero. Reporting 0% churn off that data would
        tell the user they have no problem when we simply cannot see it.
        """
        reads = sum(s.cache_read for s in self.sessions.values())
        writes = sum(s.total_writes for s in self.sessions.values())
        if reads == 0:
            return True
        return writes > 0 and reads / writes < 100

    # Aggregate cost split by billing category.
    cost_fresh_input: float = 0.0
    cost_cache_write: float = 0.0
    cost_cache_read: float = 0.0
    cost_output: float = 0.0

    # Recoverable, by mechanism.
    recoverable_miss: float = 0.0
    recoverable_excess_writes: float = 0.0

    @property
    def total_cost(self) -> float:
        return (
            self.cost_fresh_input
            + self.cost_cache_write
            + self.cost_cache_read
            + self.cost_output
        )

    @property
    def recoverable(self) -> float:
        return self.recoverable_miss + self.recoverable_excess_writes

    @property
    def recoverable_pct(self) -> float:
        if self.total_cost == 0:
            return 0.0
        return 100.0 * self.recoverable / self.total_cost

    @property
    def total_requests(self) -> int:
        return sum(s.requests for s in self.sessions.values())

    @property
    def overall_hit_rate(self) -> float:
        prompt = sum(s.prompt_tokens for s in self.sessions.values())
        if prompt == 0:
            return 0.0
        return sum(s.cache_read for s in self.sessions.values()) / prompt


def analyze(requests: list[Request]) -> Report:
    report = Report()

    by_session: dict[str, list[Request]] = defaultdict(list)
    for req in requests:
        by_session[req.session].append(req)

    for session, reqs in by_session.items():
        reqs.sort(key=lambda r: r.seq)
        stats = SessionStats(session=session)
        prev_ts: datetime | None = None

        for req in reqs:
            # Classify this request's writes as expiry or churn.
            ts = _parse_ts(req.timestamp)
            if prev_ts is not None and ts is not None:
                gap = (ts - prev_ts).total_seconds()
                if gap > TTL_5M_SECONDS:
                    stats.expiry_writes += req.cache_write_5m
                if gap > TTL_1H_SECONDS:
                    stats.expiry_writes += req.cache_write_1h
            if ts is not None:
                prev_ts = ts

            price, exact = resolve(req.model, req.provider)
            if stats.model is None and req.model:
                stats.model = req.model
            stats.provider = req.provider

            if price is None:
                # No rate card for this provider — count the tokens, keep
                # them out of every dollar figure rather than guessing.
                report.unpriced_requests += 1
                report.unpriced_tokens += req.context_size + req.output
                if req.model:
                    report.unpriced_models.add(req.model)
            else:
                if not exact and req.model:
                    report.approximated_models.add(req.model)
                p_in = price.input_per_mtok / 1_000_000
                p_out = price.output_per_mtok / 1_000_000

                report.cost_fresh_input += req.fresh_input * p_in
                report.cost_cache_write += (
                    req.cache_write_5m * p_in * price.cache_write_5m_mult
                    + req.cache_write_1h * p_in * price.cache_write_1h_mult
                )
                report.cost_cache_read += req.cache_read * p_in * price.cache_read_mult
                report.cost_output += req.output * p_out

            stats.requests += 1
            stats.fresh_input += req.fresh_input
            stats.cache_write_5m += req.cache_write_5m
            stats.cache_write_1h += req.cache_write_1h
            stats.cache_read += req.cache_read
            stats.output += req.output
            stats.thinking += req.thinking
            stats.peak_context = max(stats.peak_context, req.context_size)
            if not req.cache_visible:
                stats.cache_visible = False
            if req.seq > 0:
                stats.missable_input += req.fresh_input

        # Cache blindness is a property of the source, not of pricing —
        # track it before the unpriced early-return, or a source that is
        # both blind and unpriced silently reports neither.
        if not stats.cache_visible:
            report.blind_sessions += 1
            report.blind_tokens += stats.prompt_tokens

        # Price the two recoverable mechanisms at this session's model.
        price, _ = resolve(stats.model, stats.provider)
        if price is None:
            report.sessions[session] = stats
            continue
        p_in = price.input_per_mtok / 1_000_000

        # A missed token paid 1.0x where a hit costs 0.1x. Only claimable
        # when the source actually reports caching — otherwise fresh
        # input is just "everything", and the figure would be fiction.
        miss = (
            stats.missable_input * p_in * (1.0 - price.cache_read_mult)
            if stats.cache_visible
            else 0.0
        )

        # An excess write paid at least 1.25x where a read costs 0.1x.
        excess = stats.excess_writes * p_in * (
            price.cache_write_5m_mult - price.cache_read_mult
        )

        stats.cost = miss + excess
        report.recoverable_miss += miss
        report.recoverable_excess_writes += excess
        report.sessions[session] = stats

    return report


def worst_sessions(report: Report, n: int = 10) -> list[SessionStats]:
    """Sessions ranked by recoverable dollars."""
    ranked = sorted(report.sessions.values(), key=lambda s: s.cost, reverse=True)
    return ranked[:n]
