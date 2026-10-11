"""End-to-end test: run the pipeline, score it against ground truth, print the metrics table and the acceptance checks.

    python -m biomed.run_eval --data data/synthetic --out biomed/outputs/seed42

Exit code 1 if any acceptance check fails (the failures are real and are listed, not hidden).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .evaluate import evaluate, print_table
from .pipeline import run
from .prep import DEFAULT_DATA


def checks(res: dict, run_info: dict) -> list[tuple[str, bool, str]]:
    f, g, c, s = res["facility"], res["ghost_pickups"], res["cbwtf"], res["signals"]
    oc = c["by_pattern"]["overcapacity"]
    rt = f["recall_by_tier_at_emit"]
    return [
        ("Facility under-handover precision@25 >= 0.85", f["precision"][25] >= 0.85, f"{f['precision'][25]:.2f}"),
        ("Facility recall, obvious >= 0.80 (at flagging threshold)", rt["obvious"] >= 0.80, f"{rt['obvious']:.2f}"),
        ("Facility recall, moderate >= 0.80 (at flagging threshold)", rt["moderate"] >= 0.80, f"{rt['moderate']:.2f}"),
        ("Ghost-pickup recall >= 0.90", g["recall"] >= 0.90, f"{g['recall']:.3f}"),
        ("Ghost-pickup false-alarm rate on clean pickups <= 0.05", g["false_alarm_rate_on_clean"] <= 0.05, f"{g['false_alarm_rate_on_clean']:.4%}"),
        ("Overcapacity: 100% of obvious plants", oc["recall_by_tier"]["obvious"] == 1.0, f"{oc['recall_by_tier']['obvious']}"),
        ("Overcapacity: >= 0.80 of moderate plants", (oc["recall_by_tier"]["moderate"] or 0) >= 0.80, f"{oc['recall_by_tier']['moderate']}"),
        ("100% of signals valid (score in [0,1], >= 2 reasons, evidence refs)", s["invalid"] == 0, f"{s['count'] - s['invalid']}/{s['count']}"),
        ("Load-to-scores processing < 60 s", run_info["processing_s_load_to_scores"] < 60, f"{run_info['processing_s_load_to_scores']} s"),
    ]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default="biomed/outputs/seed42")
    a = ap.parse_args(argv)
    t0 = time.time()
    info = run(a.data, a.out, verbose=False)
    res = evaluate(a.data, a.out)
    (Path(a.out) / "metrics.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print_table(res)
    print("\nACCEPTANCE CHECKS\n" + "=" * 86)
    ok = True
    for name, passed, val in checks(res, info):
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:<72s} {val}")
    print(f"\nTotal wall time {time.time() - t0:.0f}s (pipeline {info['total_s']}s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
