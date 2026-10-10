"""Case summaries: Amazon Bedrock when credentials exist, a deterministic template otherwise.

The model is told to use ONLY the numbers in the case file, so the text cannot add facts the detectors did not find.
"""
from __future__ import annotations

import json
import os

# Set PHANTOM_BEDROCK_MODEL to a Bedrock model id / inference profile enabled in your account and region.
# Without it, summaries fall back to the deterministic template.
MODEL_ID = os.environ.get("PHANTOM_BEDROCK_MODEL", "")

SYSTEM = ("You write one-paragraph briefings for a State Pollution Control Board inspector. Use ONLY the facts in the JSON case file. "
          "Do not infer motives and do not accuse; say what the data shows and what to verify first. Max 120 words. Plain language.")


def template_summary(case: dict) -> str:
    p = case["profile"]
    rs = [r["text"] for r in case["reasons"][:3]] or ["a statistically unusual combination of signals"]
    steps = case["inspection_checklist"][:2]
    text = f"{p['name']} ({p['state']}) ranks #{case.get('rank', '?')} of the plants screened. The data shows: " + "; ".join(rs) + "."
    if steps:
        text += f" Suggested first step: {steps[0]}"
        if len(steps) > 1:
            text += f" Then: {steps[1]}"
    return text


def _has_credentials() -> bool:
    return bool(os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE") or os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
                or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))


def summarise(case: dict, use_bedrock: bool | None = None, region: str | None = None) -> dict:
    """Return {"text": ..., "source": ...}. Falls back to the template on any AWS error."""
    if use_bedrock is None:
        use_bedrock = _has_credentials() and bool(MODEL_ID)
    if not use_bedrock:
        return {"text": template_summary(case), "source": "template"}
    try:
        import boto3
        br = boto3.client("bedrock-runtime", region_name=region or os.environ.get("AWS_REGION", "us-east-1"))
        slim = {k: case[k] for k in ("profile", "reasons", "evidence", "brands_exposed", "inspection_checklist") if k in case}
        r = br.converse(modelId=MODEL_ID, system=[{"text": SYSTEM}],
                        messages=[{"role": "user", "content": [{"text": json.dumps(slim, default=str)}]}],
                        inferenceConfig={"maxTokens": 300, "temperature": 0.2})
        return {"text": r["output"]["message"]["content"][0]["text"], "source": "bedrock"}
    except Exception as e:  # no credentials, model not enabled, throttled ...
        return {"text": template_summary(case), "source": f"template (bedrock unavailable: {type(e).__name__})"}
