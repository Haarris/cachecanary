"""Live check: send the same request twice and assert the second one reads from cache.

Meant for CI. Uses the caller's own AWS credentials via boto3 (optional dependency).
"""

import json
import time
from dataclasses import dataclass

from cachecanary.usage import Usage, from_response, from_stream_events


class ProbeError(RuntimeError):
    """The probe could not run (credentials, access, model, throttling). Not a cache verdict."""


# Bedrock error codes mapped to an actionable hint.
_HINTS = {
    "AccessDeniedException": "Your AWS identity lacks bedrock:InvokeModel, or model access isn't enabled for this model in this Region.",
    "ResourceNotFoundException": "Model ID or inference profile not found in this Region.",
    "ValidationException": "Bedrock rejected the request shape (check the model ID, TTL support and request fields).",
    "ThrottlingException": "Throttled by Bedrock quotas; retry later or use another Region.",
    "ServiceQuotaExceededException": "A Bedrock service quota was exceeded.",
    "ModelNotReadyException": "The model is not ready yet; retry shortly.",
    "ModelTimeoutException": "The model timed out; retry.",
    "ServiceUnavailableException": "Bedrock is temporarily unavailable; retry.",
}


@dataclass
class ProbeResult:
    first: Usage | None
    second: Usage | None

    @property
    def passed(self) -> bool:
        return bool(self.second and self.second.cache_read > 0)


def _read_body(body):
    if hasattr(body, "read"):
        body = body.read()
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8")
    return json.loads(body) if isinstance(body, str) else body


def _call(client, payload: dict, model_id: str, stream: bool) -> Usage | None:
    if "anthropic_version" in payload:
        body = json.dumps(payload)
        if stream:
            resp = client.invoke_model_with_response_stream(modelId=model_id, body=body)
            return from_stream_events(resp.get("body") or [])
        resp = client.invoke_model(modelId=model_id, body=body)
        return from_response(_read_body(resp.get("body")))
    request = {k: v for k, v in payload.items() if k != "modelId"}
    if stream:
        resp = client.converse_stream(modelId=model_id, **request)
        return from_stream_events(resp.get("stream") or [])
    return from_response(client.converse(modelId=model_id, **request))


def _translate(exc: Exception) -> ProbeError:
    code = None
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = (response.get("Error") or {}).get("Code")
    name = type(exc).__name__
    if code:
        hint = _HINTS.get(code, "")
        return ProbeError(f"{code}: {exc}. {hint}".strip())
    if name in ("NoCredentialsError", "PartialCredentialsError", "ProfileNotFound"):
        return ProbeError(f"{name}: no usable AWS credentials. Set AWS_PROFILE or configure credentials.")
    if name == "MissingDependencyException":
        # e.g. credentials from `aws login` need the CRT extra.
        return ProbeError(f"{exc} Fix: pip install 'boto3[crt]' (or 'cachecanary[aws]').")
    if name in ("NoRegionError",):
        return ProbeError("No AWS Region configured; pass --region or set AWS_REGION.")
    if name in ("EndpointConnectionError", "ConnectTimeoutError", "ReadTimeoutError"):
        return ProbeError(f"{name}: could not reach Bedrock. Check network/Region.")
    return ProbeError(f"{name}: {exc}")


def make_client(region: str | None):
    """Separate so tests (and callers with custom sessions) can replace it."""
    try:
        import boto3
    except ImportError as exc:
        raise ProbeError("boto3 is not installed; run: pip install 'cachecanary[aws]'") from exc
    try:
        return boto3.client("bedrock-runtime", region_name=region)
    except Exception as exc:  # credential/region problems surface here, before any call
        raise _translate(exc) from exc


def probe(payload: dict, model_id: str, client=None, region: str | None = None,
          pause_s: float = 1.0, stream: bool = False) -> ProbeResult:
    try:
        if client is None:
            client = make_client(region)
        first = _call(client, payload, model_id, stream)
        time.sleep(pause_s)
        second = _call(client, payload, model_id, stream)
    except ProbeError:
        raise
    except Exception as exc:  # botocore errors are imported lazily; classify by shape
        raise _translate(exc) from exc
    return ProbeResult(first, second)
