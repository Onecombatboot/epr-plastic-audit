"""Pickup-level ghost detection from first principles: a collection that cannot have happened.

Five independent physical impossibilities, each checkable from the barcode record alone:

  closure     a pickup is recorded while the facility has declared itself closed
  cadence     a pickup is recorded on a day outside the facility's own collection rhythm
              (alternate-day facilities collect on one parity; twice-weekly facilities on two fixed weekdays)
  displaced   the vehicle's GPS fix is more than 1 km from the facility (GPS noise is metres to tens of metres)
  dwell       the vehicle stayed less than the minimum time needed to load waste
  copy        the record repeats another pickup's four category weights exactly (and has no GPS fix to contradict it)

A missing GPS fix is NOT evidence on its own: about 3% of stops lose GPS for innocent device reasons. It is used
only at plant level (excess over the baseline) and as supporting evidence for a pickup that already fails another test.
No labels are read here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .prep import closures_table, haversine_km


def _cadence_violation(p: pd.DataFrame, f: pd.DataFrame) -> np.ndarray:
    sched = f["pickup_schedule"].to_numpy()[p["fcode"].to_numpy()]
    d = p["date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    dow = (d + 3) % 7                                  # 1970-01-01 was a Thursday; Monday = 0
    par = d % 2
    fc = p["fcode"].to_numpy()
    n = len(f)
    viol = np.zeros(len(p), dtype=bool)

    alt = sched == "alternate"
    if alt.any():
        cnt = np.zeros((n, 2), dtype=np.int64)
        np.add.at(cnt, (fc[alt], par[alt]), 1)
        mode = cnt.argmax(axis=1)
        viol[alt] = par[alt] != mode[fc[alt]]
    tw = sched == "twice_weekly"
    if tw.any():
        cnt = np.zeros((n, 7), dtype=np.int64)
        np.add.at(cnt, (fc[tw], dow[tw]), 1)
        top2 = np.argsort(-cnt, axis=1)[:, :2]
        keep = (top2[fc[tw]] == dow[tw][:, None]).any(axis=1)
        viol[tw] = ~keep
    return viol


def pickup_flags(p: pd.DataFrame, f: pd.DataFrame, ch: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """Return a frame aligned to `p` with the four hard flags, GPS facts and an overall ghost flag."""
    g = ref["ghost"]
    out = pd.DataFrame(index=p.index)
    has_gps = p["gps_lat"].notna().to_numpy()
    dist = np.full(len(p), np.nan)
    dist[has_gps] = haversine_km(p["flat"].to_numpy()[has_gps], p["flon"].to_numpy()[has_gps],
                                 p["gps_lat"].to_numpy()[has_gps], p["gps_lon"].to_numpy()[has_gps])
    out["dist_km"] = dist
    # A fix beyond ~1 km cannot be GPS noise. The observed noise scale (99.5th percentile of stop-to-facility distance) is kept
    # for reporting only; using it as the threshold would flag legitimate noise (about 0.2% of clean stops).
    noise = float(np.nanpercentile(dist, 99.5)) if has_gps.any() else float("nan")
    tol = float(g["displacement_km"])
    out["gps_missing"] = ~has_gps
    out["displaced"] = np.nan_to_num(dist, nan=0.0) > tol
    out["short_dwell"] = (p["dwell_min"].to_numpy() < g["min_dwell_min"]) & has_gps

    closed = np.zeros(len(p), dtype=bool)
    fidx = p.groupby("facility_id", observed=True).indices
    d = p["date"].to_numpy()
    for r in closures_table(ch).itertuples(index=False):
        m = fidx.get(r.facility_id)
        if m is not None:
            closed[m] = (d[m] >= np.datetime64(r.start)) & (d[m] < np.datetime64(r.end))
    out["in_closure"] = closed
    out["off_cadence"] = _cadence_violation(p, f)
    # copied record: the same four category weights to the last digit on two different dates of one facility. Real collections of
    # several kilograms never repeat to 0.01 kg in every category, so a repeat is a copy; the copy is the one with no GPS fix.
    big = np.flatnonzero((p["kg"].to_numpy() >= g["duplicate_min_kg"]))
    key = p.iloc[big][["fcode", "yellow_kg", "red_kg", "white_kg", "blue_kg"]]
    isdup = key.duplicated(keep=False).to_numpy()
    cand = big[isdup]
    copy = np.zeros(len(p), dtype=bool)
    if len(cand):
        q = p.iloc[cand][["fcode", "yellow_kg", "red_kg", "white_kg", "blue_kg", "date"]].copy()
        q["gps"] = has_gps[cand]
        gid = q.groupby(["fcode", "yellow_kg", "red_kg", "white_kg", "blue_kg"], sort=False).ngroup()
        q["any_gps"] = q.groupby(gid)["gps"].transform("any")
        q["first"] = q.groupby(gid)["date"].transform("min") == q["date"]
        # the copy lacks GPS; if every member lacks GPS, the earliest is taken as the original
        flag = (~q["gps"]) & (q["any_gps"] | ~q["first"])
        copy[cand] = flag.to_numpy()
    out["duplicate_copy"] = copy
    out["hard_flags"] = out[["in_closure", "off_cadence", "displaced", "short_dwell", "duplicate_copy"]].sum(axis=1)
    out["ghost"] = out["hard_flags"] > 0
    out.attrs["gps_tolerance_km"] = tol
    out.attrs["gps_noise_p995_km"] = noise
    return out
