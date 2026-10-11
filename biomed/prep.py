"""Load the BMW tables and put pickups on a physical footing: waste accumulates between collections.

A pickup record is not "kg per day". It is the waste produced since the previous collection, minus the days the
facility was closed. Comparing it with expected generation therefore needs the length of that window
(`acc_days`), the beds in service during it (declared expansions), and the calendar (season, weekday) effects.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "synthetic"
CATS = ["yellow_kg", "red_kg", "white_kg", "blue_kg"]


def load_reference() -> dict:
    return yaml.safe_load((Path(__file__).parent / "reference.yaml").read_text(encoding="utf-8"))


def load(data_dir=DEFAULT_DATA) -> dict[str, pd.DataFrame]:
    d = Path(data_dir) / "bmw"
    t = {k: pd.read_parquet(d / f"{k}.parquet") for k in ["facilities", "cbwtfs", "pickups", "cbwtf_daily_treatment", "facility_changes"]}
    p = t["pickups"]
    p["date"] = pd.to_datetime(p["pickup_date"])
    p["kg"] = p[CATS].sum(axis=1)
    p.drop(columns=["pickup_date"], inplace=True)
    t["cbwtf_daily_treatment"]["date"] = pd.to_datetime(t["cbwtf_daily_treatment"]["date"])
    ch = t["facility_changes"]
    ch["effective_date"] = pd.to_datetime(ch["effective_date"])
    return t


def closures_table(ch: pd.DataFrame) -> pd.DataFrame:
    c = ch[ch.change_type == "temporary_closure"].copy()
    c["start"] = c["effective_date"]
    c["end"] = c["start"] + pd.to_timedelta(c["old_value"], unit="D")      # old_value = declared duration in days
    return c[["facility_id", "start", "end"]]


def attach_facility(p: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    """Add integer facility code, coordinates and type to the pickup frame (positional lookups: fast on 4.6M rows)."""
    codes = pd.Categorical(p["facility_id"], categories=f["facility_id"]).codes
    p["fcode"] = codes
    for col in ["lat", "lon"]:
        p["f" + col] = f[col].to_numpy()[codes]
    return p


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    a = np.sin(np.radians(lat2 - lat1) / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def expected_rates(f: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """Expected kg/day per facility: low, mid and high of the published norm range (occupancy x kg per bed or per patient)."""
    occ_lo, occ_hi = ref["occupancy"]
    rows = []
    for ft, g in f.groupby("facility_type"):
        spec = ref["types"][ft]
        lo, hi = spec["kg_per_unit_day"]
        if spec["basis"] == "beds":
            units = g["beds"].to_numpy(dtype=float)
            e_lo, e_hi = units * occ_lo * lo, units * occ_hi * hi
        else:
            units = g["avg_daily_patients"].to_numpy(dtype=float)
            e_lo, e_hi = units * lo, units * hi
        rows.append(pd.DataFrame({"facility_id": g["facility_id"].to_numpy(), "units": units, "e_lo": e_lo, "e_hi": e_hi,
                                   "e_mid": np.sqrt(e_lo * e_hi)}))
    return pd.concat(rows).set_index("facility_id").reindex(f["facility_id"])


def accumulation(p: pd.DataFrame, f: pd.DataFrame, ch: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """For each (non-suspect) pickup: days of waste accumulated since the previous pickup, net of declared closures.

    `p` must already be restricted to pickups believed to have happened and sorted by (fcode, date).
    Adds: prev_date, acc_days, bed_mult.  The first pickup of each facility has no known window and is dropped later.
    """
    p = p.sort_values(["fcode", "date"], kind="mergesort").reset_index(drop=True)
    same = p["fcode"].to_numpy()[1:] == p["fcode"].to_numpy()[:-1]
    prev = p["date"].shift(1)
    prev[np.concatenate([[True], ~same])] = pd.NaT
    p["prev_date"] = prev
    gap = (p["date"] - prev).dt.days.to_numpy(dtype=float)

    # closed days inside the window: days prev+1 .. date, i.e. the half-open interval [prev+1, date+1)
    closed = np.zeros(len(p))
    fidx = p.groupby("facility_id", observed=True).indices
    d_all = p["date"].to_numpy()
    pv_all = p["prev_date"].to_numpy()
    one = np.timedelta64(1, "D")
    for r in closures_table(ch).itertuples(index=False):
        m = fidx.get(r.facility_id)
        if m is None:
            continue
        lo = np.maximum(pv_all[m] + one, np.datetime64(r.start))
        hi = np.minimum(d_all[m] + one, np.datetime64(r.end))
        ov = (hi - lo).astype("timedelta64[D]").astype(float)
        closed[m] = np.where(np.isnan(ov), 0.0, np.clip(ov, 0, None))
    p["acc_days"] = np.clip(gap - closed, 0.0, None)

    # declared bed expansions: time-weighted beds over the window
    bm = np.ones(len(p))
    for r in ch[ch.change_type == "bed_expansion"].itertuples(index=False):
        m = fidx.get(r.facility_id)
        if m is None:
            continue
        eff = np.datetime64(r.effective_date)
        d, pv = d_all[m], pv_all[m]
        span = (d - pv).astype("timedelta64[D]").astype(float)
        after = np.clip((d + one - np.maximum(pv + one, eff)).astype("timedelta64[D]").astype(float), 0, None)
        w = np.where(span > 0, np.clip(after / np.where(span > 0, span, 1), 0, 1), 1.0)
        bm[m] = 1.0 + (float(r.new_value) / float(r.old_value) - 1.0) * np.where(d >= eff, w, 0.0)
    p["bed_mult"] = bm
    return p
