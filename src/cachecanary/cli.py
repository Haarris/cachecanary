"""cachecanary command line.

  cachecanary lint request.json [--model ID]
  cachecanary diff previous.json next.json [--model ID]
  cachecanary probe request.json --model ID [--region R] [--stream]
  cachecanary logs file.json[.gz] ... [--by model|principal|model+principal] [--min-hit 0.5]

Exit codes: 0 ok, 1 caching problem found (use to gate CI), 2 could not run (bad input, AWS error).
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from cachecanary import diff, lint, logs, probe
from cachecanary.request import RequestError, normalize

EXIT_OK, EXIT_PROBLEM, EXIT_ERROR = 0, 1, 2


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
    return EXIT_PROBLEM if any(f.severity == "error" for f in findings) else EXIT_OK


def cmd_diff(args) -> int:
    reasons = diff.explain(normalize(_load(args.previous), args.model), normalize(_load(args.next), args.model))
    lines = [f"{r.code}: {r.message}" + (f" ({r.location})" if r.location else "") for r in reasons]
    _emit([asdict(r) for r in reasons], args.json, lines)
    return EXIT_OK if reasons and reasons[0].code == "prefix-identical" else EXIT_PROBLEM


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
    return EXIT_OK if result.passed else EXIT_PROBLEM


def cmd_logs(args) -> int:
    stats = logs.ReadStats()
    records = []
    for p in args.files:
        path = Path(p)
        if not path.exists():
            raise InputError(f"file not found: {p}")
        try:
            records.extend(logs.iter_records(path, stats))
        except (OSError, json.JSONDecodeError) as exc:
            raise InputError(f"could not read {p}: {exc}") from exc
    groups = logs.aggregate(records, by=args.by)
    data, lines, failing = {}, [], False
    for key, g in sorted(groups.items(), key=lambda kv: -kv[1].usage.total_input):
        ratio = g.usage.hit_ratio
        data[key] = {"calls": g.calls, "unparsed": g.unparsed, "hit_ratio": round(ratio, 3), **asdict(g.usage)}
        flag = ""
        if args.min_hit is not None and g.calls - g.unparsed > 0 and ratio < args.min_hit:
            flag, failing = "  <-- below threshold", True
        lines.append(f"{key}: hit {ratio:.0%} over {g.calls} calls "
                     f"(read {g.usage.cache_read}, write {g.usage.cache_write}, uncached {g.usage.uncached_input}, "
                     f"unparsed {g.unparsed}){flag}")
    if stats.bad_lines:
        lines.append(f"Skipped {stats.bad_lines} unreadable line(s).")
    _emit(data, args.json, lines or ["No invocation log records found."])
    return EXIT_PROBLEM if failing else EXIT_OK


def _unit(value: str) -> float:
    f = float(value)
    if not 0 <= f <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return f


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cachecanary", description="CacheCanary: catch silent prompt-cache breakage for Claude on Amazon Bedrock.")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
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
    p.add_argument("--min-hit", type=_unit, help="fail if any group's hit ratio is below this (0-1)")
    p.set_defaults(func=cmd_logs)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (InputError, RequestError, probe.ProbeError) as exc:
        print(f"cachecanary: error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
