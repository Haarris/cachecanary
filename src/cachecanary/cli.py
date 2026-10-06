"""cachecanary command line.

  cachecanary lint request.json [--model ID]
  cachecanary diff previous.json next.json [--model ID]
  cachecanary probe request.json --model ID [--region R] [--stream]
  cachecanary logs file.json[.gz] ... [--by model|principal|model+principal] [--min-hit 0.5]

Exit codes: 0 ok, 1 caching problem found (use to gate CI), 2 could not run (bad input, AWS error).
Add --github (before the command) for GitHub Actions annotations and a job summary.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from cachecanary import __version__, diff, lint, logs, probe
from cachecanary import github as gh
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
    if args.github:
        rows = []
        for key, d in data.items():
            below = args.min_hit is not None and d["calls"] - d["unparsed"] > 0 and d["hit_ratio"] < args.min_hit
            if below:
                print(gh.annotation("error", f"{key}: cache hit rate {d['hit_ratio']:.0%} is below {args.min_hit:.0%}",
                                    title="CacheCanary: low cache hit rate"))
            rows.append([key, d["calls"], f"{d['hit_ratio']:.0%}", d["cache_read"], d["cache_write"], d["uncached_input"],
                         d["unparsed"], "❌" if below else ""])
        gh.append_summary("### CacheCanary logs\n\n" + (gh.table(
            ["group", "calls", "hit rate", "read", "write", "uncached", "unparsed", "below threshold"], rows)
            if rows else "No invocation log records found."))
    return EXIT_PROBLEM if failing else EXIT_OK


def _unit(value: str) -> float:
    f = float(value)
    if not 0 <= f <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
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
    p.add_argument("--min-hit", type=_unit, help="fail if any group's hit ratio is below this (0-1)")
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
