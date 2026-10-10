"""Generate every synthetic dataset.

    python -m synthdata.generate_all --scale large --seed 42 --out data/synthetic
    python -m synthdata.generate_all --scale small --only registry,plastic

Registry is always built first: plastic and yield depend on it (shared entities).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import __version__, bmw, ocems, plastic, registry, yield_streams
from .common import load_calibration, write_json

ALL = ["registry", "plastic", "yield", "ocems", "bmw"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="large", choices=["small", "medium", "large"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--only", default=",".join(ALL))
    ap.add_argument("--side-trace-scale", type=float, default=1.0, help="scenario: scale the by-products of fraud (1.0 = baseline)")
    ap.add_argument("--evasion-boost", type=float, default=0.0, help="scenario: push covering behaviours (fabricated outputs, doctored bills, ghost feedstock) towards 1")
    a = ap.parse_args(argv)
    only = set(a.only.split(","))
    out = Path(a.out)
    cal = load_calibration()
    cal["fraud"]["side_trace_scale"] = a.side_trace_scale
    cal["fraud"]["evasion_boost"] = a.evasion_boost
    manifest: dict = {}
    timings = {}
    t0 = time.time()
    reg = None
    if only & {"registry", "plastic", "yield"}:
        t = time.time(); reg = registry.generate(cal, a.scale, a.seed, out, manifest); timings["registry"] = round(time.time() - t, 1)
        print(f"registry  {timings['registry']:>7}s", flush=True)
    if "plastic" in only:
        t = time.time(); plastic.generate(cal, a.scale, a.seed, out, manifest, reg); timings["plastic"] = round(time.time() - t, 1)
        print(f"plastic   {timings['plastic']:>7}s", flush=True)
    if "yield" in only:
        t = time.time(); yield_streams.generate(cal, a.scale, a.seed, out, manifest, reg); timings["yield"] = round(time.time() - t, 1)
        print(f"yield     {timings['yield']:>7}s", flush=True)
    if "ocems" in only:
        t = time.time(); ocems.generate(cal, a.scale, a.seed, out, manifest); timings["ocems"] = round(time.time() - t, 1)
        print(f"ocems     {timings['ocems']:>7}s", flush=True)
    if "bmw" in only:
        t = time.time(); bmw.generate(cal, a.scale, a.seed, out, manifest); timings["bmw"] = round(time.time() - t, 1)
        print(f"bmw       {timings['bmw']:>7}s", flush=True)
    for k in manifest:
        manifest[k]["bytes"] = Path(k).stat().st_size
    tables = {str(Path(k).relative_to(out)).replace("\\", "/"): v for k, v in manifest.items()}
    prev = out / "manifest.json"
    if prev.exists() and only != set(ALL):          # partial re-run: merge into the existing manifest
        old = json.loads(prev.read_text(encoding="utf-8"))
        tables = {**old.get("tables", {}), **tables}
        timings = {**old.get("timings_s", {}), **timings}
    meta = {"generator_version": __version__, "scale": a.scale, "seed": a.seed, "side_trace_scale": a.side_trace_scale, "evasion_boost": a.evasion_boost, "timings_s": timings,
            "total_s": round(sum(timings.values()), 1), "tables": tables}
    write_json(meta, out / "manifest.json")
    tot_rows = sum(v["rows"] for v in tables.values())
    tot_mb = sum(v["bytes"] for v in tables.values()) / 1e6
    print(f"done: {len(tables)} tables, {tot_rows:,} rows, {tot_mb:,.1f} MB in {meta['total_s']}s -> {out}")


if __name__ == "__main__":
    main()
