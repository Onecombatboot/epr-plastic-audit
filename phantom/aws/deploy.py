"""Publish pipeline outputs to AWS: case files to S3, signals to DynamoDB.

    python -m phantom.aws.deploy --run phantom/outputs/seed42 --bucket BUCKET --table TABLE
"""
import argparse
import json
from decimal import Decimal
from pathlib import Path


def main():
    import boto3
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--table", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    s3 = boto3.client("s3")
    cases = list((run / "cases").glob("*.json"))
    for f in cases:
        s3.upload_file(str(f), a.bucket, f"cases/{f.name}")
    table = boto3.resource("dynamodb").Table(a.table)
    with table.batch_writer() as bw:
        for line in (run / "signals.jsonl").read_text(encoding="utf-8").splitlines():
            bw.put_item(Item=json.loads(line, parse_float=Decimal))
    print("published", len(cases), "case files")


if __name__ == "__main__":
    main()
