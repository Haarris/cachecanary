"""cachecanary command line.

  cachecanary lint request.json [--model ID]
  cachecanary diff previous.json next.json [--model ID]
  cachecanary probe request.json --model ID [--region R] [--stream]
  cachecanary logs FILE_OR_FOLDER ... [--by model|principal|model+principal] [--min-hit 0.5] [--price 3]

Exit codes: 0 ok, 1 caching problem found (use to gate CI), 2 could not run (bad input, AWS error).
Add --github (before the command) for GitHub Actions annotations and a job summary.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from cachecanary import __version__, diff, lint, logs, pricing, probe
from cachecanary import github as gh
from cachecanary.request import RequestError, normalize

EXIT_OK, EXIT_PROBLEM, EXIT_ERROR = 0, 1, 2
# Hit rate the 'lost to cache misses' figure compares against when --min-hit is not set.
DEFAULT_TARGET_HIT = 0.9


class InputError(Exception):
    pass


def _load(path: str):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InputError(f"file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"{path} is not valid JSON: {exc}") from exc


def _emit(data, as_json: bool, lines: list[str]) -> None:
    print(json.dumps(data, indent=2) if as_json else "\n".join(lines))


def cmd_lint(args) -> int:
    findings = lint.lint(normalize(_load(args.request), args.model))
    lines = [f"[{f.severity}] {f.rule}: {f.message}" + (f" ({f.location})" if f.location else "") for f in findings]
    _emit([asdict(f) for f in findings], args.json, lines or ["No caching problems found."])
    if args.github:
        for f in findings:
            level = "error" if f.severity == "error" else "warning"
            where = f" ({f.location})" if f.location else ""
            print(gh.annotation(level, f"{f.message}{where}", file=args.request, title=f"CacheCanary: {f.rule}"))
        rows = [[f.severity, f.rule, f.location or "", f.message] for f in findings]
        gh.append_summary(f"### CacheCanary lint: `{args.request}`\n\n" + (
            gh.table(["severity", "rule", "location", "message"], rows) if rows else "No caching problems found. ✅"))
    return EXIT_PROBLEM if any(f.severity == "error" for f in findings) else EXIT_OK


def cmd_diff(args) -> int:
    reasons = diff.explain(normalize(_load(args.previous), args.model), normalize(_load(args.next), args.model))
    lines = [f"{r.code}: {r.message}" + (f" ({r.location})" if r.location else "") for r in reasons]
    _emit([asdict(r) for r in reasons], args.json, lines)
    ok = bool(reasons) and reasons[0].code == "prefix-identical"
    if args.github:
        for r in reasons:
            where = f" ({r.location})" if r.location else ""
            print(gh.annotation("notice" if ok else "error", f"{r.message}{where}", file=args.next, title=f"CacheCanary: {r.code}"))
        gh.append_summary(f"### CacheCanary diff: `{args.previous}` → `{args.next}`\n\n" +
                          gh.table(["reason", "location", "explanation"], [[r.code, r.location or "", r.message] for r in reasons]))
    return EXIT_OK if ok else EXIT_PROBLEM


def _probe_explanation(result: probe.ProbeResult, model: str) -> list[str]:
    first, second = result.first, result.second
    if second is None:
        return ["Could not read token usage from the response, so caching could not be verified."]
    if result.passed:
        return []
    if first and first.cache_write == 0 and first.cache_read == 0:
        return ["The first call wrote nothing to the cache: no checkpoint was applied (prefix below the "
                "model minimum, no cache markers, or the library/model ID skipped caching)."]
    if first and first.cache_write > 0:
        hint = ["The first call wrote to the cache but the second did not read it."]
        if model.split(".")[0] in ("us", "eu", "apac", "global", "jp", "au", "ca"):
            hint.append("Cross-region profiles can route the two calls to different Regions; retry or test with "
                        "an in-Region model ID to rule this out.")
        return hint
    return []


def cmd_probe(args) -> int:
    result = probe.probe(_load(args.request), args.model, region=args.region, stream=args.stream)
    data = {"passed": result.passed, "first": asdict(result.first) if result.first else None,
            "second": asdict(result.second) if result.second else None}
    status = "PASS: second call read from cache." if result.passed else "FAIL: second call did not read from cache."
    lines = [status, f"first:  {data['first']}", f"second: {data['second']}"]
    lines += _probe_explanation(result, args.model)
    if not result.passed:
        lines.append("Run `cachecanary lint` on the same request to find the likely cause.")
    _emit(data, args.json, lines)
    if args.github:
        if not result.passed:
            print(gh.annotation("error", " ".join(lines[3:]) or status, file=args.request, title="CacheCanary: cache not read on 2nd call"))
        rows = [[name, u["uncached_input"], u["cache_read"], u["cache_write"]] if u else [name, "?", "?", "?"]
                for name, u in (("1st call", data["first"]), ("2nd call", data["second"]))]
        gh.append_summary(f"### CacheCanary probe: `{args.request}` on `{args.model}` — {'PASS ✅' if result.passed else 'FAIL ❌'}\n\n"
                          + gh.table(["call", "uncached", "cache read", "cache write"], rows))
    return EXIT_OK if result.passed else EXIT_PROBLEM


LOG_SUFFIXES = (".json", ".jsonl", ".gz", ".log", ".txt")


def _log_files(arg: str) -> list[Path]:
    """A file as given, or every log-like file under a folder (S3 syncs nest them by date)."""
    path = Path(arg)
    if not path.exists():
        raise InputError(f"file not found: {arg}")
    if not path.is_dir():
        return [path]
    found = sorted(f for f in path.rglob("*") if f.is_file() and f.name.endswith(LOG_SUFFIXES)
                   and not any(part.startswith(".") for part in f.relative_to(path).parts))
    if not found:
        raise InputError(f"no log files (.json, .jsonl, .gz, .log, .txt) under {arg}")
    return found


def _money(usd: float) -> str:
    if usd == 0:
        return "$0.00"
    if usd < 0.01:
        return "under $0.01"
    return f"${usd:,.2f}"


def _group_cost(g, target: float, custom_price: float | None) -> dict | None:
    """Input cost and spend lost to misses for one group, priced per model. None if no model is priced."""
    cost = lost = 0.0
    priced_any, unpriced, sources = False, [], set()
    for model, usage in g.by_model.items():
        priced = pricing.rates_for(model, custom_price)
        if priced is None:
            if usage.total_input:
                unpriced.append(model)
            continue
        priced_any = True
        sources.add(priced.source)
        cost += pricing.input_cost(usage, priced.rates)
        lost += pricing.lost_to_misses(usage, priced.rates, target)
    if not priced_any:
        return None
    return {"input_cost_usd": round(cost, 6), "lost_usd": round(lost, 6), "target_hit": target,
            "price": "custom" if "custom" in sources else "list", "unpriced_models": sorted(unpriced)}


def cmd_logs(args) -> int:
    stats = logs.ReadStats()
    records = []
    for arg in args.files:
        for path in _log_files(arg):
            try:
                records.extend(logs.iter_records(path, stats))
            except (OSError, UnicodeDecodeError, EOFError, json.JSONDecodeError) as exc:
                raise InputError(f"could not read {path}: {exc}") from exc
    groups = logs.aggregate(records, by=args.by)
    target = args.min_hit if args.min_hit is not None else DEFAULT_TARGET_HIT
    data, lines, failing = {}, [], False
    total_cost = total_lost = 0.0
    any_priced = any_unpriced = False
    for key, g in sorted(groups.items(), key=lambda kv: -kv[1].usage.total_input):
        ratio = g.usage.hit_ratio
        cost = _group_cost(g, target, args.price)
        data[key] = {"calls": g.calls, "unparsed": g.unparsed, "hit_ratio": round(ratio, 3), **asdict(g.usage),
                     **(cost or {"input_cost_usd": None, "lost_usd": None, "target_hit": target})}
        flag = ""
        if args.min_hit is not None and g.calls - g.unparsed > 0 and ratio < args.min_hit:
            flag, failing = "  <-- below threshold", True
        lines.append(f"{key}: hit {ratio:.0%} over {g.calls} calls "
                     f"(read {g.usage.cache_read}, write {g.usage.cache_write}, uncached {g.usage.uncached_input}, "
                     f"unparsed {g.unparsed}){flag}")
        if g.usage.total_input == 0:
            continue
        if cost is None:
            any_unpriced = True
            lines.append("  no price for this model; add --price <USD per million input tokens> to see dollars")
            continue
        any_priced = True
        total_cost += cost["input_cost_usd"]
        total_lost += cost["lost_usd"]
        label = "at your price" if cost["price"] == "custom" else "at list price"
        if cost["lost_usd"] > 0:
            lines.append(f"  input cost {_money(cost['input_cost_usd'])} {label}; about {_money(cost['lost_usd'])} "
                         f"of it lost to cache misses (target: {target:.0%} hit rate)")
        else:
            lines.append(f"  input cost {_money(cost['input_cost_usd'])} {label}; at or above the {target:.0%} target hit rate, "
                         f"nothing lost to cache misses")
        if cost["unpriced_models"]:
            lines.append(f"  not priced: {', '.join(cost['unpriced_models'])} (add --price to include)")
    if any_priced and len(data) > 1:
        lines.append(f"Total: input cost {_money(total_cost)}; about {_money(total_lost)} lost to cache misses "
                     f"(target: {target:.0%} hit rate)")
    if any_priced and not args.price:
        lines.append("List prices: Amazon Bedrock on-demand, Oct 2026 (global. IDs at list, others +10%). "
                     "Use --price for your own rate.")
    if stats.bad_lines:
        lines.append(f"Skipped {stats.bad_lines} unreadable line(s).")
    _emit(data, args.json, lines or ["No invocation log records found."])
    if args.github:
        rows = []
        for key, d in data.items():
            below = args.min_hit is not None and d["calls"] - d["unparsed"] > 0 and d["hit_ratio"] < args.min_hit
            if below:
                print(gh.annotation("error", f"{key}: cache hit rate {d['hit_ratio']:.0%} is below {args.min_hit:.0%}",
                                    title="CacheCanary: low cache hit rate"))
            priced = d["input_cost_usd"] is not None
            rows.append([key, d["calls"], f"{d['hit_ratio']:.0%}", d["cache_read"], d["cache_write"], d["uncached_input"],
                         d["unparsed"], _money(d["input_cost_usd"]) if priced else "",
                         _money(d["lost_usd"]) if priced else "", "❌" if below else ""])
        gh.append_summary("### CacheCanary logs\n\n" + (gh.table(
            ["group", "calls", "hit rate", "read", "write", "uncached", "unparsed", "input cost",
             f"lost vs {target:.0%} hit", "below threshold"], rows)
            if rows else "No invocation log records found."))
    return EXIT_PROBLEM if failing else EXIT_OK


def _unit(value: str) -> float:
    f = float(value)
    if not 0 <= f <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return f


def _price(value: str) -> float:
    try:
        f = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number, e.g. 3 for $3 per million input tokens") from None
    if not 0 < f < 1000:  # also rejects nan and inf
        raise argparse.ArgumentTypeError("must be a price in USD per million input tokens, e.g. 3")
    return f


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cachecanary", description="CacheCanary: catch silent prompt-cache breakage for Claude on Amazon Bedrock.")
    parser.add_argument("--version", action="version", version=f"cachecanary {__version__}")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--github", action="store_true",
                        help="also emit GitHub Actions annotations and a job summary (used by the GitHub Action)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("lint", help="static checks on one request")
    p.add_argument("request")
    p.add_argument("--model")
    p.set_defaults(func=cmd_lint)

    p = sub.add_parser("diff", help="explain why the next request missed the previous one's cache")
    p.add_argument("previous")
    p.add_argument("next")
    p.add_argument("--model")
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("probe", help="send the request twice and check the second reads from cache")
    p.add_argument("request")
    p.add_argument("--model", required=True)
    p.add_argument("--region")
    p.add_argument("--stream", action="store_true", help="use ConverseStream / InvokeModelWithResponseStream")
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("logs", help="cache hit rates from Bedrock invocation logs")
    p.add_argument("files", nargs="+")
    p.add_argument("--by", choices=["model", "principal", "model+principal"], default="model")
    p.add_argument("--min-hit", type=_unit, help="fail if any group's hit ratio is below this (0-1); "
                   "also the target for the dollars lost figure (default 0.9)")
    p.add_argument("--price", type=_price, help="your input price in USD per million tokens, instead of list prices")
    p.set_defaults(func=cmd_logs)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (InputError, RequestError, probe.ProbeError) as exc:
        print(f"cachecanary: error: {exc}", file=sys.stderr)
        if getattr(args, "github", False):
            print(gh.annotation("error", str(exc), title="CacheCanary could not run"))
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
