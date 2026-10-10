"""GET /top?n=20 and GET /case/{entity_id} over precomputed Phantom Plastic results (DynamoDB signals, S3 case files)."""
import json
import os
from decimal import Decimal

import boto3

ddb = boto3.resource("dynamodb")
s3 = boto3.client("s3")


def _json(o):
    if isinstance(o, Decimal):
        return float(o)
    raise TypeError


def _resp(code, body):
    return {"statusCode": code, "headers": {"content-type": "application/json"}, "body": json.dumps(body, default=_json)}


def handler(event, _ctx):
    path = event.get("rawPath", "")
    if path.endswith("/top"):
        n = int((event.get("queryStringParameters") or {}).get("n", 20))
        items = ddb.Table(os.environ["TABLE"]).scan()["Items"]
        items.sort(key=lambda x: -float(x["score"]))
        return _resp(200, items[:n])
    eid = (event.get("pathParameters") or {}).get("entity_id")
    if eid:
        try:
            case = json.loads(s3.get_object(Bucket=os.environ["BUCKET"], Key=f"cases/{eid}.json")["Body"].read())
        except s3.exceptions.NoSuchKey:
            return _resp(404, {"error": "no case file"})
        from narrative import summarise
        case["summary"] = summarise(case)
        return _resp(200, case)
    return _resp(404, {"error": "not found"})
