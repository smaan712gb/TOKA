"""Model pricing and cache-tier multipliers.

Prices are USD per million tokens, first-party Anthropic API list rates.
Cache multipliers are applied to the model's *input* price:

    cache read      0.1x   (served from cache)
    cache write 5m  1.25x  (ephemeral, 5-minute TTL)
    cache write 1h  2.0x   (ephemeral, 1-hour TTL)
    fresh input     1.0x   (full price, no cache involvement)
"""

from dataclasses import dataclass

CACHE_READ_MULT = 0.1
CACHE_WRITE_5M_MULT = 1.25
CACHE_WRITE_1H_MULT = 2.0


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float
    # Minimum cacheable prefix; shorter prefixes silently do not cache.
    min_cacheable_tokens: int


# Keyed by the model id as it appears in transcripts.
PRICES: dict[str, ModelPrice] = {
    "claude-fable-5": ModelPrice(10.0, 50.0, 512),
    "claude-mythos-5": ModelPrice(10.0, 50.0, 512),
    "claude-opus-5": ModelPrice(5.0, 25.0, 512),
    "claude-opus-4-8": ModelPrice(5.0, 25.0, 1024),
    "claude-opus-4-7": ModelPrice(5.0, 25.0, 2048),
    "claude-opus-4-6": ModelPrice(5.0, 25.0, 4096),
    "claude-opus-4-5": ModelPrice(5.0, 25.0, 4096),
    "claude-sonnet-5": ModelPrice(3.0, 15.0, 1024),
    "claude-sonnet-4-6": ModelPrice(3.0, 15.0, 1024),
    "claude-sonnet-4-5": ModelPrice(3.0, 15.0, 1024),
    "claude-haiku-4-5": ModelPrice(1.0, 5.0, 4096),
}

# Anything not in PRICES falls back to this so a run never dies on an
# unrecognised id. Unknown models are surfaced in the report.
FALLBACK = PRICES["claude-opus-5"]


def resolve(model: str | None) -> tuple[ModelPrice, bool]:
    """Return (price, known). Bare ids are matched first, then prefix match
    so dated snapshots like `claude-haiku-4-5-20251001` resolve correctly."""
    if not model:
        return FALLBACK, False
    if model in PRICES:
        return PRICES[model], True
    for known, price in PRICES.items():
        if model.startswith(known):
            return price, True
    return FALLBACK, False


def cost(
    price: ModelPrice,
    *,
    fresh_input: int,
    cache_write_5m: int,
    cache_write_1h: int,
    cache_read: int,
    output: int,
) -> float:
    """Dollar cost of one request."""
    p_in = price.input_per_mtok / 1_000_000
    p_out = price.output_per_mtok / 1_000_000
    return (
        fresh_input * p_in
        + cache_write_5m * p_in * CACHE_WRITE_5M_MULT
        + cache_write_1h * p_in * CACHE_WRITE_1H_MULT
        + cache_read * p_in * CACHE_READ_MULT
        + output * p_out
    )
