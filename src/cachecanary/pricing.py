"""On-demand Amazon Bedrock list prices for Claude input tokens, used to put dollars on cache misses.

Source: the AWS Price List file for AmazonBedrockFoundationModels (checked 2026-10-15; file published 2026-10-12), on-demand
"standard" tier. The prices were identical in us-east-1, us-west-2, eu-central-1 and ap-northeast-1.
Global endpoints ('global.' model IDs) are billed at these prices; regional and geo cross-region
endpoints ('us.', 'eu.', ... or a bare 'anthropic.' ID) cost 10% more for these models.

Legacy models (Sonnet 4, Opus 4 and 4.1, 3.5 Haiku) have one price on every endpoint, with no
regional premium. Claude 3.7 Sonnet and 3.5 Sonnet are left out on purpose: their Bedrock prices
depend on extended-access terms, so CacheCanary shows no dollars for them unless you pass --price.
"""

import re
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
    "claude-sonnet-5-5": _rates(2, 0.05),
    "claude-sonnet-5": _rates(2),
    "claude-sonnet-4-6": _rates(3),
    "claude-sonnet-4-5": _rates(3),
    "claude-haiku-5-5": _rates(0.1),
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


# --- Anthropic API prices (used by `claude-code`) ------------------------------------------------
#
# Source: https://platform.claude.com/docs/en/about-claude/pricing (checked 2026-11-18), "Model
# pricing", "Fast mode pricing", "Data residency pricing" and "Long context pricing". USD per million
# tokens. Cache hits cost 0.025x the input price on Fable/Mythos 5.1 and 5.2, 0.05x on Opus 5.5 and
# Sonnet 5.5, and 0.1x on every other model. Claude 3.x models are retired and not listed.

@dataclass(frozen=True)
class ApiRates:
    """USD per million tokens, every part of one request."""
    input: float
    cache_write_5m: float
    cache_write_1h: float
    cache_read: float
    output: float

    def scaled(self, factor: float) -> "ApiRates":
        return ApiRates(self.input * factor, self.cache_write_5m * factor, self.cache_write_1h * factor,
                        self.cache_read * factor, self.output * factor)


def _api(base: float, read_multiplier: float = 0.1, output: float | None = None) -> ApiRates:
    return ApiRates(base, base * 1.25, base * 2, base * read_multiplier, base * 5 if output is None else output)


API_PRICES: dict[str, ApiRates] = {
    "claude-fable-5-2": _api(8, 0.025),
    "claude-mythos-5-2": _api(8, 0.025),
    "claude-fable-5-1": _api(10, 0.025),
    "claude-mythos-5-1": _api(10, 0.025),
    "claude-fable-5": _api(10),
    "claude-mythos-5": _api(10),
    "claude-opus-5-5": _api(4, 0.05),
    "claude-opus-5": _api(5),
    "claude-opus-4-8": _api(5),
    "claude-opus-4-7": _api(5),
    "claude-opus-4-6": _api(5),
    "claude-opus-4-5": _api(5),
    "claude-opus-4-1": _api(15),
    "claude-opus-4": _api(15),
    "claude-sonnet-5-5": _api(2, 0.05),
    "claude-sonnet-5": _api(2),
    "claude-sonnet-4-6": _api(3),
    "claude-sonnet-4-5": _api(3),
    "claude-sonnet-4": _api(3),
    "claude-haiku-5-5": _api(0.1),  # prompts up to 100,000 tokens
    "claude-haiku-4-5": _api(1),
}

# Claude Haiku 5.5 is priced by prompt length: a request whose prompt (uncached input + cache reads +
# cache writes) is over 100,000 tokens pays these prices for the whole request.
HAIKU_55_LONG_PROMPT = 100_000
HAIKU_55_LONG = _api(0.5)

# Fast mode (usage.speed == "fast"): its own input and output prices; the cache multipliers apply on top.
FAST_PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-2": (16, 80),
    "claude-mythos-5-2": (16, 80),
    "claude-opus-5-5": (8, 40),
    "claude-opus-5": (10, 50),
    "claude-opus-4-8": (10, 50),
}

# US-only inference (usage.inference_geo == "us") costs 1.1x on every part, for Claude 4.6 and later.
DATA_RESIDENCY_MULTIPLIER = 1.1
DATA_RESIDENCY_MODELS = {
    "claude-fable-5-2", "claude-mythos-5-2", "claude-fable-5-1", "claude-mythos-5-1", "claude-fable-5", "claude-mythos-5",
    "claude-opus-5-5", "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
    "claude-sonnet-5-5", "claude-sonnet-5", "claude-sonnet-4-6", "claude-haiku-5-5",
}

# Default shares of the input price for --price on a model CacheCanary can't price.
DEFAULT_API_SHARES = ApiRates(1, 1.25, 2, DEFAULT_READ_MULTIPLIER, 5)


def api_model_key(model_id: str | None) -> str | None:
    """The API price key a model ID names: a bare API ID (claude-opus-5-5, claude-haiku-4-5-20251001),
    a Vertex ID (claude-opus-4-5@20251101) or a Bedrock ID (us.anthropic.claude-opus-5-5)."""
    if not model_id:
        return None
    lowered = model_id.lower()
    for key in sorted(API_PRICES, key=len, reverse=True):
        if re.search(re.escape(key) + r"(?=$|[@:/\[]|-v\d|-\d{8}|-0(?:$|[@:/\[]|-\d{8}))", lowered):
            return key
    return None


def is_bedrock_id(model_id: str | None) -> bool:
    lowered = (model_id or "").lower()
    return lowered.startswith("arn:") or lowered.startswith("anthropic.") or ".anthropic." in lowered


@dataclass(frozen=True)
class ApiPriced:
    rates: ApiRates
    source: str  # "list" | "custom"
    note: str = ""  # "fast", "us", "long prompt", "bedrock regional" ... for reports


def api_rates_for(model_id: str | None, prompt_tokens: int = 0, speed: str | None = None,
                  inference_geo: str | None = None, custom_input_price: float | None = None) -> ApiPriced | None:
    """Prices for one Claude Code request, or None when the model has no known price.

    Bedrock IDs ('anthropic.', 'us.anthropic.', 'global.anthropic.' and ARNs) are priced with the Bedrock
    rule this package uses for `logs`: global IDs at list, other endpoints 10% more for 4.5 and later
    models. The AWS price list (AmazonBedrockFoundationModels, us-east-1, published 2026-11-09) has the
    same global prices as the Anthropic API table above, output and Haiku 5.5 long prompts included.

    With --price (custom_input_price) the input price is yours and the other parts keep the model's
    shares of it (5m write 1.25x, 1h write 2x, its cache read share, output 5x for every listed model);
    fast mode, data residency, the Bedrock premium and the long prompt tier are then not applied.
    """
    key = api_model_key(model_id)
    listed = API_PRICES.get(key) if key else None
    if custom_input_price is not None:
        shares = listed.scaled(1 / listed.input) if listed else DEFAULT_API_SHARES
        return ApiPriced(shares.scaled(custom_input_price), "custom")
    if listed is None:
        return None
    notes = []
    if key == "claude-haiku-5-5" and prompt_tokens > HAIKU_55_LONG_PROMPT:
        listed = HAIKU_55_LONG
        notes.append("long prompt")
    if (speed or "").lower() == "fast":
        if key not in FAST_PRICES:
            return None  # fast mode on a model with no published fast price: don't guess
        fast_in, fast_out = FAST_PRICES[key]
        read_share = listed.cache_read / listed.input
        listed = ApiRates(fast_in, fast_in * 1.25, fast_in * 2, fast_in * read_share, fast_out)
        notes.append("fast")
    if is_bedrock_id(model_id):
        lowered = (model_id or "").lower()
        is_global = lowered.startswith("global.") or "/global." in lowered
        if not is_global and key not in NO_REGIONAL_PREMIUM:
            listed = listed.scaled(REGIONAL_PREMIUM)
            notes.append("bedrock regional")
    elif (inference_geo or "").lower() == "us" and key in DATA_RESIDENCY_MODELS:
        listed = listed.scaled(DATA_RESIDENCY_MULTIPLIER)
        notes.append("us only")
    return ApiPriced(listed, "list", ", ".join(notes))
