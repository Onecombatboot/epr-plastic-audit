"""Explicit deployment command; without --deploy, prints a reviewable plan only."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def plan(stack: str, region: str, run: Path) -> list[list[str]]:
    if not (Path(run)/"signals.jsonl").exists():
        raise ValueError("Run the local pipeline before deployment")
    template = Path(__file__).with_name("template.yaml")
    build_dir = Path(__file__).parent/".aws-sam"/"build"
    return [["sam", "build", "--template-file", str(template), "--build-dir", str(build_dir)],
            ["sam", "deploy", "--template-file", str(build_dir/"template.yaml"), "--stack-name", stack,
             "--region", region, "--resolve-s3", "--capabilities", "CAPABILITY_IAM", "--no-fail-on-empty-changeset"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", default="yieldbal-audit")
    parser.add_argument("--region", default="ap-south-1")
    parser.add_argument("--run", type=Path, default=Path("yieldbal/outputs/seed42"))
    parser.add_argument("--deploy", action="store_true")
    args = parser.parse_args()
    commands = plan(args.stack, args.region, args.run)
    if not args.deploy:
        print(json.dumps({"commands": commands, "after_deploy": "Upload signals.jsonl to the stack's Bucket output at yieldbal/signals.jsonl."}, indent=2))
        return
    for command in commands:
        subprocess.run(command, check=True)
    import boto3
    outputs = boto3.client("cloudformation", region_name=args.region).describe_stacks(StackName=args.stack)["Stacks"][0]["Outputs"]
    bucket = next(output["OutputValue"] for output in outputs if output["OutputKey"] == "Bucket")
    boto3.client("s3", region_name=args.region).upload_file(str(args.run/"signals.jsonl"), bucket, "yieldbal/signals.jsonl")
    print("Inspection signals uploaded.")


if __name__ == "__main__":
    main()
