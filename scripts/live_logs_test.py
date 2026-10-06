"""Live verification that `cachecanary logs` reads real Bedrock model-invocation logs.

Invocation logging is REGION-WIDE: while it is on, every Bedrock call in that Region is
recorded, including request and response bodies. So this script refuses to run unless the
Region is provably idle and has no existing logging setup, and it removes everything it
created when done.

Safety checks (all must pass, otherwise nothing is created):
  1. Region is not in --protected-regions (default: us-east-2, where production runs).
  2. CloudWatch shows 0 Bedrock invocations in this Region over the last 7 days.
  3. No model-invocation logging configuration exists in this Region (nothing to overwrite).

Then: create a private S3 bucket -> enable text logging to it -> make 4 synthetic calls
(Converse, ConverseStream, InvokeModel, InvokeModelWithResponseStream, each twice) -> wait for
log delivery -> run cachecanary's log parser on the delivered files -> compare with the usage the
API returned -> disable logging and delete the bucket (unless --keep).

    python scripts/live_logs_test.py --region us-west-2
"""

import argparse
import datetime as dt
import json
import sys
import time
import uuid

from cachecanary import logs as cc_logs
from cachecanary import probe
from cachecanary.usage import Usage

SONNET = "us.anthropic.claude-sonnet-4-6"


def check_safety(region: str, protected: set[str], cloudwatch, bedrock, allowed_invocations: int = 0) -> list[str]:
    """Return reasons to refuse; empty list means safe to proceed."""
    problems = []
    if region in protected:
        problems.append(f"{region} is protected (production). Choose an idle Region.")
        return problems
    end = dt.datetime.now(dt.timezone.utc)
    stats = cloudwatch.get_metric_statistics(
        Namespace="AWS/Bedrock", MetricName="Invocations",
        StartTime=end - dt.timedelta(days=7), EndTime=end, Period=7 * 24 * 3600, Statistics=["Sum"],
    )
    total = sum(p.get("Sum", 0) for p in stats.get("Datapoints", []))
    if total > allowed_invocations:
        problems.append(
            f"{region} had {int(total)} Bedrock invocations in the last 7 days (allowed: {allowed_invocations}); "
            "logging would capture other callers. If these are all your own test calls (check with CloudWatch "
            "by ModelId and hour), pass --allow-invocations with that exact number.")
    config = bedrock.get_model_invocation_logging_configuration().get("loggingConfig")
    if config:
        problems.append(f"{region} already has invocation logging configured; refusing to overwrite it: {json.dumps(config)}")
    return problems


def bucket_policy(bucket: str, account: str, region: str) -> str:
    return json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Sid": "AmazonBedrockLogsWrite",
            "Effect": "Allow",
            "Principal": {"Service": "bedrock.amazonaws.com"},
            "Action": ["s3:PutObject"],
            "Resource": [f"arn:aws:s3:::{bucket}/AWSLogs/{account}/BedrockModelInvocationLogs/*"],
            "Condition": {
                "StringEquals": {"aws:SourceAccount": account},
                "ArnLike": {"aws:SourceArn": f"arn:aws:bedrock:{region}:{account}:*"},
            },
        }],
    })


def synthetic_payloads(salt: str) -> list[tuple[str, dict, bool]]:
    text = f"[logtest {salt}] " + " ".join(f"Rule {i}: answer politely and cite section {i}." for i in range(220))
    conv = {"system": [{"text": text}, {"cachePoint": {"type": "default"}}],
            "messages": [{"role": "user", "content": [{"text": "Reply with OK."}]}],
            "inferenceConfig": {"maxTokens": 5}}
    inv = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 5,
           "system": [{"type": "text", "text": text + " (invoke)", "cache_control": {"type": "ephemeral"}}],
           "messages": [{"role": "user", "content": "Reply with OK."}]}
    conv_s = json.loads(json.dumps(conv))
    conv_s["system"][0]["text"] += " (stream)"
    inv_s = json.loads(json.dumps(inv))
    inv_s["system"][0]["text"] += " (stream)"
    return [("converse", conv, False), ("converse-stream", conv_s, True),
            ("invoke", inv, False), ("invoke-stream", inv_s, True)]


def read_delivered(s3, bucket: str, account: str, workdir) -> tuple[list[dict], int, list[str]]:
    """Download every delivered object and parse it with cachecanary's own reader."""
    import pathlib

    prefix = f"AWSLogs/{account}/BedrockModelInvocationLogs/"
    records, keys = [], []
    stats = cc_logs.ReadStats()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
            local = pathlib.Path(workdir) / obj["Key"].replace("/", "__")
            local.write_bytes(s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read())
            records.extend(cc_logs.iter_records(local, stats))
    return records, stats.bad_lines, keys


def run(args) -> int:
    import boto3

    session = boto3.Session(region_name=args.region)
    account = session.client("sts").get_caller_identity()["Account"]
    bedrock = session.client("bedrock")
    problems = check_safety(args.region, set(args.protected_regions), session.client("cloudwatch"), bedrock,
                            args.allow_invocations)
    if problems:
        print("REFUSING to run:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        return 2

    salt = uuid.uuid4().hex[:8]
    bucket = f"cachecanary-logtest-{account}-{args.region}-{salt}"
    s3 = session.client("s3")
    created_bucket = enabled_logging = False
    print(f"Account {account}, Region {args.region}, run {salt}")
    try:
        kwargs = {} if args.region == "us-east-1" else {"CreateBucketConfiguration": {"LocationConstraint": args.region}}
        s3.create_bucket(Bucket=bucket, **kwargs)
        created_bucket = True
        s3.put_public_access_block(Bucket=bucket, PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
        s3.put_bucket_tagging(Bucket=bucket, Tagging={"TagSet": [
            {"Key": "purpose", "Value": "cachecanary-log-test"}, {"Key": "delete-after", "Value": "same-day"}]})
        s3.put_bucket_policy(Bucket=bucket, Policy=bucket_policy(bucket, account, args.region))
        print(f"Created private bucket {bucket}")

        bedrock.put_model_invocation_logging_configuration(loggingConfig={
            "s3Config": {"bucketName": bucket},
            "textDataDeliveryEnabled": True, "imageDataDeliveryEnabled": False,
            "embeddingDataDeliveryEnabled": False,
        })
        enabled_logging = True
        print("Enabled invocation logging in this Region (text only)")
        time.sleep(10)  # let the config take effect before calling

        client = session.client("bedrock-runtime")
        expected = Usage()
        per_call = []
        for name, payload, stream in synthetic_payloads(salt):
            for attempt in (1, 2):
                usage = probe._call(client, payload, SONNET, stream)
                per_call.append((name, attempt, usage))
                if usage:
                    expected = expected + usage
                time.sleep(1)
        for name, attempt, usage in per_call:
            print(f"  {name} #{attempt}: {usage}")

        print(f"Waiting up to {args.wait_minutes} min for log delivery...")
        deadline = time.time() + args.wait_minutes * 60
        import tempfile

        records, bad, keys = [], 0, []
        with tempfile.TemporaryDirectory() as workdir:
            while time.time() < deadline:
                records, bad, keys = read_delivered(s3, bucket, account, workdir)
                if sum(1 for r in records if cc_logs._is_invocation_record(r)) >= len(per_call):
                    break
                time.sleep(30)
        invocation = [r for r in records if cc_logs._is_invocation_record(r)]
        print(f"Objects delivered: {len(keys)} (e.g. {keys[:2]})")
        print(f"Invocation records: {len(invocation)} (expected {len(per_call)}), other objects: {len(records) - len(invocation)}, unreadable fragments: {bad}")
        if args.save:
            with open(args.save, "w") as fh:
                json.dump(records, fh, indent=1, default=str)
            print(f"Saved raw records to {args.save} (synthetic prompts only)")

        groups = cc_logs.aggregate(records)
        got = Usage()
        unparsed = 0
        for g in groups.values():
            got = got + g.usage
            unparsed += g.unparsed
        ops = sorted({r.get("operation") for r in records})
        print(f"Operations seen in logs: {ops}")
        print(f"API usage   : read {expected.cache_read}, write {expected.cache_write}, uncached {expected.uncached_input}")
        print(f"From logs   : read {got.cache_read}, write {got.cache_write}, uncached {got.uncached_input}, unparsed {unparsed}")
        ok = (len(invocation) == len(per_call) and bad == 0 and unparsed == 0 and got.cache_read == expected.cache_read
              and got.cache_write == expected.cache_write)
        print("PASS: logs parser matches the API exactly." if ok else
              "FAIL: mismatch. Inspect the saved records (--save) to see the real record shape.")
        return 0 if ok else 1
    finally:
        cleanup(args, bedrock, s3, bucket, enabled_logging, created_bucket)


def _empty_bucket(s3, bucket: str) -> None:
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if keys:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": keys})


def cleanup(args, bedrock, s3, bucket, enabled_logging, created_bucket) -> None:
    """Each step runs even if another fails; anything left over is printed for manual removal."""
    if args.keep:
        if enabled_logging or created_bucket:
            print(f"--keep: bucket {bucket} and the logging config in {args.region} were LEFT IN PLACE.")
        return
    leftovers = []
    if enabled_logging:
        try:
            bedrock.delete_model_invocation_logging_configuration()
            print("Disabled invocation logging in this Region")
        except Exception as exc:
            leftovers.append(f"logging config in {args.region} ({exc}) -> aws bedrock delete-model-invocation-logging-configuration --region {args.region}")
    if created_bucket:
        # Log files can still land for a short while after logging is disabled; retry emptying.
        for attempt in range(4):
            try:
                _empty_bucket(s3, bucket)
                s3.delete_bucket(Bucket=bucket)
                print(f"Deleted bucket {bucket}")
                break
            except Exception as exc:
                if attempt == 3:
                    leftovers.append(f"bucket {bucket} ({exc}) -> aws s3 rb s3://{bucket} --force")
                else:
                    time.sleep(15)
    if leftovers:
        print("CLEANUP INCOMPLETE. Remove manually:\n  - " + "\n  - ".join(leftovers), file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True)
    parser.add_argument("--protected-regions", nargs="*", default=["us-east-2"])
    parser.add_argument("--allow-invocations", type=int, default=0,
                        help="exact count of known (your own) Bedrock calls in the Region over 7 days to tolerate")
    parser.add_argument("--wait-minutes", type=float, default=12)
    parser.add_argument("--save", default="live_log_records.json", help="where to save raw delivered records ('' to skip)")
    parser.add_argument("--keep", action="store_true", help="do not remove the bucket/logging config afterwards")
    sys.exit(run(parser.parse_args()))
