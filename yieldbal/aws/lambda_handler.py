"""Small read API over exported signals; no numerical dependencies are required in Lambda."""
from __future__ import annotations

import json
import os
import re

_CACHE = {}


def _load_signals() -> list[dict]:
    bucket = os.environ["RESULTS_BUCKET"]
    key = os.getenv("RESULTS_KEY", "yieldbal/signals.jsonl")
    identity = (bucket, key)
    if _CACHE.get("identity") != identity:
        import boto3
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        _CACHE.update(identity=identity, signals=[json.loads(line) for line in body.splitlines() if line.strip()])
    return _CACHE["signals"]


def _response(status: int, body) -> dict:
    return {"statusCode": status, "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body, allow_nan=False)}


def handler(event: dict, context) -> dict:
    path = event.get("rawPath") or event.get("path", "")
    query = event.get("queryStringParameters") or {}
    if path != "/top" and not path.startswith("/case/"):
        return _response(404, {"error": "Unknown route"})
    try:
        if path == "/top":
            n = int(query.get("n", "10"))
            if not 1 <= n <= 100:
                return _response(400, {"error": "n must be between 1 and 100"})
            signals = _load_signals()
            if query.get("stream"):
                signals = [s for s in signals if s.get("stream") == query["stream"]]
            if query.get("state"):
                signals = [s for s in signals if s.get("state") == query["state"]]
            return _response(200, {"synthetic": True, "cases": sorted(signals, key=lambda s: s["score"], reverse=True)[:n]})
        entity = path.removeprefix("/case/")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", entity):
            return _response(400, {"error": "Invalid entity id"})
        signal = next((s for s in _load_signals() if s["entity_id"] == entity), None)
        return _response(200, {"synthetic": True, **signal}) if signal else _response(404, {"error": "Recycler not in inspection queue"})
    except (ValueError, TypeError):
        return _response(400, {"error": "Invalid request"})
    except Exception:
        return _response(503, {"error": "Audit results unavailable"})
