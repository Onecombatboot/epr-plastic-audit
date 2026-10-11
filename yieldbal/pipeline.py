"""Run the independent yield audit; results are additive files under the chosen run folder."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .checks import compute
from .prep import DEFAULT_DATA, DEFAULT_RUN, load, load_reference
from .scoring import score, signals


def run(data_dir: Path = DEFAULT_DATA, out_dir: Path = DEFAULT_RUN) -> dict:
    start = time.perf_counter()
    ref = load_reference()
    tables = load(Path(data_dir))
    monthly, features = compute(tables, ref)
    scores, contributions = score(features, ref)
    emitted = signals(scores, monthly, contributions)
    elapsed = time.perf_counter()-start
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out/"scores.parquet")
    features.to_parquet(out/"features.parquet")
    monthly.to_parquet(out/"monthly_checks.parquet", index=False)
    contributions.to_parquet(out/"model_contributions.parquet")
    with (out/"signals.jsonl").open("w") as file:
        for signal in emitted:
            file.write(json.dumps(signal, allow_nan=False)+"\n")
    meta = {"entities": len(scores), "entity_months": len(monthly), "streams": sorted(scores.stream.unique()),
            "runtime_s": round(elapsed, 3), "priority": int(scores.priority.sum()),
            "screening_including_priority": int(scores.screening.sum()), "signals": len(emitted),
            "suspected_hollow_t": float(scores.suspected_hollow_t.sum()), "synthetic": True,
            "reference": ref, "label_access": "none"}
    (out/"run.json").write_text(json.dumps(meta, indent=2, allow_nan=False)+"\n")
    return {"scores": scores, "monthly": monthly, "features": features, "signals": emitted, "meta": meta}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    result = run(args.data, args.out)
    print(json.dumps(result["meta"], indent=2))
    print(result["scores"][["stream", "rank", "score", "tier", "suspected_hollow_t"]].head(10).to_string())


if __name__ == "__main__":
    main()
