"""End-to-end Biomedical Waste Gap pipeline.

    python -m biomed.pipeline --data data/synthetic --out biomed/outputs/seed42
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .cbwtf import cbwtf_reasons, cbwtf_scores
from .facility import facility_scores, onset_date, prepare, reasons as fac_reasons
from .ghosts import pickup_flags
from .prep import DEFAULT_DATA, attach_facility, load, load_reference

EMIT_THRESHOLD = 0.20


def run(data_dir=DEFAULT_DATA, out_dir="biomed/outputs/seed42", top_cases=300, verbose=True) -> dict:
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    T: dict[str, float] = {}
    t0 = time.time()
    ref = load_reference()
    t = load(data_dir)
    f, c, p, d, ch = t["facilities"], t["cbwtfs"], t["pickups"], t["cbwtf_daily_treatment"], t["facility_changes"]
    T["load"] = time.time() - t0
    log(f"[{time.time() - t0:5.1f}s] loaded {len(p):,} pickups, {len(f):,} facilities, {len(c)} CBWTFs")

    p = attach_facility(p, f)
    flags = pickup_flags(p, f, ch, ref)
    T["ghost_flags"] = time.time() - t0
    log(f"[{time.time() - t0:5.1f}s] ghost tests: {int(flags['ghost'].sum()):,} pickups fail a physical test (GPS tolerance {flags.attrs['gps_tolerance_km'] * 1000:.0f} m)")

    clean = p[~flags["ghost"].to_numpy()]
    pr, er = prepare(clean, f, ch, ref)
    T["accumulation"] = time.time() - t0
    log(f"[{time.time() - t0:5.1f}s] accumulation windows for {len(pr):,} pickups; calendar factors learned: {pr.attrs['calendar']}")

    fac = facility_scores(pr, f, er, ch, ref)
    cb = cbwtf_scores(p, flags, fac, c, d, ref)
    T["scoring"] = time.time() - t0
    log(f"[{time.time() - t0:5.1f}s] facility and CBWTF scores")

    fac.to_parquet(out / "facility_scores.parquet")
    cb.to_parquet(out / "cbwtf_scores.parquet")
    keep = flags[flags["ghost"] | flags["gps_missing"]].join(p[["pickup_id", "facility_id", "cbwtf_id", "date"]])
    keep.to_parquet(out / "flagged_pickups.parquet")

    # signals (shared agent contract)
    ftype = f.set_index("facility_id")["facility_type"]
    changes_by_fac = ch.groupby("facility_id")
    n_sig = 0
    low_ids = pr.sort_values("r").groupby("fcode").head(3)[["fcode", "pickup_id"]].groupby("fcode")["pickup_id"].apply(list)
    with open(out / "signals.jsonl", "w", encoding="utf-8") as fh:
        for fid, row in fac[fac["tier"] != "none"].sort_values("score", ascending=False).iterrows():
            fcode = f.index[f["facility_id"] == fid][0]
            spec = ref["types"][row["facility_type"]]
            norm = (spec["kg_per_unit_day"][0] * (ref["occupancy"][0] if spec["basis"] == "beds" else 1.0), spec["kg_per_unit_day"][1] * (ref["occupancy"][1] if spec["basis"] == "beds" else 1.0))
            on = onset_date(pr, fcode) if row["s_cp"] > 0.05 else None
            rs = fac_reasons(row, f.loc[fcode], "bed" if spec["basis"] == "beds" else "patient", norm, row["n_changes"], on)
            refs = [f"bmw/facilities:{fid}"] + [f"bmw/pickups:{x}" for x in low_ids.get(fcode, [])]
            if fid in changes_by_fac.groups:
                refs += [f"bmw/facility_changes:{fid}:{r.effective_date:%Y-%m-%d}:{r.change_type}" for r in changes_by_fac.get_group(fid).itertuples()]
            fh.write(json.dumps({"entity_id": fid, "entity_kind": "facility", "agent": "biomedical_gap", "score": round(float(row["score"]), 4), "reasons": rs, "evidence_refs": refs}) + "\n")
            n_sig += 1
        gp = keep[keep["ghost"]].sort_values("hard_flags", ascending=False)
        gp_by = gp.groupby("cbwtf_id")["pickup_id"].apply(lambda s: list(s.head(5)))
        dd = d.merge(c[["cbwtf_id", "incinerator_kg_per_hr", "permitted_kg_per_day"]], on="cbwtf_id")
        bad_days = dd[(dd["incinerated_kg_reported"] > dd["permitted_kg_per_day"]) | (dd["incinerated_kg_reported"] > dd["incinerator_kg_per_hr"] * 24)].groupby("cbwtf_id")["date"].apply(lambda s: [f"{x:%Y-%m-%d}" for x in s.head(3)])
        for cid, row in cb[cb["score"] >= EMIT_THRESHOLD].sort_values("score", ascending=False).iterrows():
            refs = [f"bmw/cbwtfs:{cid}"] + [f"bmw/pickups:{x}" for x in gp_by.get(cid, [])] + [f"bmw/cbwtf_daily_treatment:{cid}:{x}" for x in bad_days.get(cid, [])]
            fh.write(json.dumps({"entity_id": cid, "entity_kind": "cbwtf", "agent": "biomedical_gap", "score": round(float(row["score"]), 4), "reasons": cbwtf_reasons(row), "evidence_refs": refs}) + "\n")
            n_sig += 1
    T["signals"] = time.time() - t0

    # pickups of the top facilities, for the dashboard's generation-vs-handover curves
    topf = fac.sort_values("score", ascending=False).head(top_cases).index
    pr[pr["facility_id"].isin(topf)][["pickup_id", "facility_id", "date", "kg", "acc_days", "exp_mid", "r"]].to_parquet(out / "top_facility_pickups.parquet")

    res = {"version": __version__, "n_pickups": int(len(p)), "n_signals": n_sig, "calendar_factors": pr.attrs["calendar"],
           "gps_tolerance_m": round(flags.attrs["gps_tolerance_km"] * 1000, 1),
           "timings_s": {k: round(v - (T.get(prev, 0) if prev else 0), 1) for prev, (k, v) in zip([None] + list(T), T.items())},
           "total_s": round(time.time() - t0, 1)}
    # pure compute time excluding the 'top facility pickups' export and file writes
    res["processing_s_load_to_scores"] = round(T["scoring"], 1)
    (out / "run.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"[{time.time() - t0:5.1f}s] done: {n_sig} signals; load-to-scores {T['scoring']:.1f}s")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default="biomed/outputs/seed42")
    a = ap.parse_args(argv)
    run(a.data, a.out)


if __name__ == "__main__":
    main()
