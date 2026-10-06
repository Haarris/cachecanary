"""On-demand Amazon Bedrock list prices for Claude input tokens, used to put dollars on cache misses.

Source: the AWS Price List file for AmazonBedrockFoundationModels (checked 2026-10-06), on-demand
"standard" tier. The prices were identical in us-east-1, us-west-2, eu-central-1 and ap-northeast-1.
Global endpoints ('global.' model IDs) are billed at these prices; regional and geo cross-region
endpoints ('us.', 'eu.', ... or a bare 'anthropic.' ID) cost 10% more for these models.

Legacy models (Sonnet 4, Opus 4 and 4.1, 3.5 Haiku) have one price on every endpoint, with no
regional premium. Claude 3.7 Sonnet and 3.5 Sonnet are left out on purpose: their Bedrock prices
depend on extended-access terms, so CacheCanary shows no dollars for them unless you pass --price.
"""

from dataclasses import dataclass

from cachecanary.models import lookup
from cachecanary.usage import Usage

REGIONAL_PREMIUM = 1.1


@dataclass(frozen=True)
class Rates:
    """USD per million tokens."""
    input: float
    cache_read: float
    cache_write_5m: float
    cache_write_1h: float

    def scaled(self, factor: float) -> "Rates":
        return Rates(self.input * factor, self.cache_read * factor,
                     self.cache_write_5m * factor, self.cache_write_1h * factor)


def _rates(base: float, read_multiplier: float = 0.1) -> Rates:
    return Rates(base, base * read_multiplier, base * 1.25, base * 2)


# Global-endpoint prices, keyed like models.CLAUDE_LIMITS. Each row matches the AWS price file.
GLOBAL_PRICES: dict[str, Rates] = {
    "claude-fable-5-1": _rates(10, 0.025),
    "claude-mythos-5-1": _rates(10, 0.025),
    "claude-fable-5": _rates(10),
    "claude-mythos-5": _rates(10),
    "claude-opus-5-5": _rates(4, 0.05),
    "claude-opus-5": _rates(5),
    "claude-opus-4-8": _rates(5),
    "claude-opus-4-7": _rates(5),
    "claude-opus-4-6": _rates(5),
    "claude-opus-4-5": _rates(5),
    "claude-sonnet-5-5": _rates(2),
    "claude-sonnet-5": _rates(2),
    "claude-sonnet-4-6": _rates(3),
    "claude-sonnet-4-5": _rates(3),
    "claude-haiku-4-5": _rates(1),
    "claude-sonnet-4": _rates(3),
    "claude-opus-4-1": _rates(15),
    "claude-opus-4": _rates(15),
    "claude-3-5-haiku": _rates(0.8),
}

# Older models priced the same on global and regional endpoints (AWS price list).
NO_REGIONAL_PREMIUM = {"claude-sonnet-4", "claude-opus-4-1", "claude-opus-4", "claude-3-5-haiku"}

# Cache read price as a share of the input price, for --price on models CacheCanary can't price.
DEFAULT_READ_MULTIPLIER = 0.1


@dataclass(frozen=True)
class Priced:
    rates: Rates
    source: str  # "list" | "custom"


def rates_for(model_id: str | None, custom_input_price: float | None = None) -> Priced | None:
    """List prices for a Bedrock model ID, or rates built from --price (USD per million input tokens).

    With --price the regional premium is not added (it is your own rate), but the model's cache
    read multiplier is kept when CacheCanary knows the model.
    """
    key, _ = lookup(model_id)
    listed = GLOBAL_PRICES.get(key) if key else None
    if custom_input_price is not None:
        read_multiplier = listed.cache_read / listed.input if listed else DEFAULT_READ_MULTIPLIER
        return Priced(_rates(custom_input_price, read_multiplier), "custom")
    if listed is None:
        return None
    is_global = (model_id or "").lower().startswith("global.")
    if is_global or key in NO_REGIONAL_PREMIUM:
        return Priced(listed, "list")
    return Priced(listed.scaled(REGIONAL_PREMIUM), "list")


def input_cost(usage: Usage, rates: Rates) -> float:
    """What the input side of these calls cost, in USD. Writes with no TTL breakdown count as 5-minute."""
    write_1h = min(usage.cache_write_1h, usage.cache_write)
    write_5m = usage.cache_write - write_1h
    return (usage.uncached_input * rates.input + usage.cache_read * rates.cache_read
            + write_5m * rates.cache_write_5m + write_1h * rates.cache_write_1h) / 1_000_000


def lost_to_misses(usage: Usage, rates: Rates, target_hit: float) -> float:
    """Extra input spend compared with the same calls at the target hit rate.

    The tokens that would have been cache reads at the target were instead paid for as
    uncached input or cache writes, in the same mix those were actually paid in.
    """
    total = usage.total_input
    shortfall = target_hit * total - usage.cache_read
    not_read = usage.uncached_input + usage.cache_write
    if total == 0 or shortfall <= 0 or not_read == 0:
        return 0.0
    paid_for_not_read = input_cost(Usage(usage.uncached_input, 0, usage.cache_write, 0,
                                         usage.cache_write_1h), rates) * 1_000_000 / not_read
    return shortfall * (paid_for_not_read - rates.cache_read) / 1_000_000
