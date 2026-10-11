"""Treatment-facility (CBWTF) audit: do its pickups, weights, treatment hours and diesel add up physically?

  ghosts    pickup-level impossibilities (see ghosts.py) aggregated per plant
  padding   a plant that inflates recorded weights lifts ALL the facilities it serves above their peers
  capacity  incinerating more than the unit can burn (rated kg/hr x 24 h), or more than the permit allows
  diesel    burning waste takes fuel; far too little diesel per kg means the burn did not happen
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .facility import ramp


def daily_features(c: pd.DataFrame, d: pd.DataFrame, ref: dict) -> pd.DataFrame:
    d = d.merge(c[["cbwtf_id", "incinerator_kg_per_hr", "permitted_kg_per_day"]], on="cbwtf_id")
    rated_day = d["incinerator_kg_per_hr"] * ref["cbwtf"]["max_incinerator_hours"]
    d["impossible"] = d["incinerated_kg_reported"] > rated_day * 1.001
    d["over_permit_burn"] = d["incinerated_kg_reported"] > d["permitted_kg_per_day"] * 1.001
    d["intake_over_permit"] = d["intake_kg"] > d["permitted_kg_per_day"] * 1.001
    d["rate_over_rating"] = (d["incinerator_hours_reported"] > 0) & (d["incinerated_kg_reported"] / d["incinerator_hours_reported"].clip(lower=1e-6) > d["incinerator_kg_per_hr"] * 1.02)
    d["diesel_per_kg"] = d["diesel_litres_purchased"] / d["incinerated_kg_reported"].replace(0, np.nan)
    lo = ref["cbwtf"]["diesel_l_per_kg"][0]
    d["diesel_low"] = d["diesel_per_kg"] < lo
    g = d.groupby("cbwtf_id")
    out = pd.DataFrame({
        "days": g.size(), "impossible_days": g["impossible"].sum(), "over_permit_burn_days": g["over_permit_burn"].sum(),
        "intake_over_permit_days": g["intake_over_permit"].sum(), "rate_over_rating_days": g["rate_over_rating"].sum(),
        "diesel_med": g["diesel_per_kg"].median(), "diesel_low_share": g["diesel_low"].mean(),
        "incin_share": g["incinerated_kg_reported"].sum() / g["intake_kg"].sum(),
    })
    return out


def cbwtf_scores(p: pd.DataFrame, flags: pd.DataFrame, fac: pd.DataFrame, c: pd.DataFrame, d: pd.DataFrame, ref: dict) -> pd.DataFrame:
    C = c.set_index("cbwtf_id")[["name", "state", "district", "lat", "lon", "incinerator_kg_per_hr", "permitted_kg_per_day"]].copy()
    cb = p["cbwtf_id"].to_numpy()
    fl = flags.assign(cbwtf_id=cb)
    g = fl.groupby("cbwtf_id")
    C["pickups"] = g.size()
    C["ghost_pickups"] = g["ghost"].sum()
    for k in ["in_closure", "off_cadence", "displaced", "short_dwell", "duplicate_copy"]:
        C["n_" + k] = g[k].sum()
    C["gps_missing_rate"] = g["gps_missing"].mean()
    C["ghost_rate"] = C["ghost_pickups"] / C["pickups"]
    C["s_ghost"] = 1 - np.exp(-C["ghost_pickups"] / 8.0)

    # padding: recorded weights inflated on some facilities push them above anything a legitimate site can produce.
    # Count the facilities a plant serves whose median handover exceeds the published maximum; the legitimate rate of
    # that event is estimated from the whole population (median over plants), so the test needs no assumed distribution.
    fz = fac[fac["n"] >= 20]
    n = fz.groupby("cbwtf_id").size()
    C["n_facilities"] = n
    fz = fz.assign(above=fz["above_norm"] > 0.08, upper=fz["z_level"] > 1.5)
    C["n_above_max"] = fz.groupby("cbwtf_id")["above"].sum()
    C["n_upper"] = fz.groupby("cbwtf_id")["upper"].sum()
    p_above = max(float(fz["above"].mean()), 0.002)
    p_upper = max(float(fz.groupby("cbwtf_id")["upper"].mean().median()), 0.005)
    ex_a, ex_u = n * p_above, n * p_upper
    C["z_above"] = (C["n_above_max"] - ex_a) / np.sqrt(ex_a + 1.0)
    C["z_upper"] = (C["n_upper"] - ex_u) / np.sqrt(ex_u + 1.0)
    C["upper_share"] = C["n_upper"] / n
    C["z_median_level"] = fz.groupby("cbwtf_id")["z_level"].median()
    C["s_pad"] = np.maximum(ramp(C["z_above"], 2.0, 6.0), ramp(C["z_upper"], 2.5, 7.0))

    D = daily_features(c, d, ref)
    C = C.join(D)
    C["s_cap"] = 1 - np.exp(-(C["impossible_days"].fillna(0) + C["over_permit_burn_days"].fillna(0)) / 4.0)
    C["s_diesel"] = ramp(C["diesel_low_share"].fillna(0), 0.10, 0.60)
    C["s_intake"] = ramp(C["intake_over_permit_days"].fillna(0) / C["days"].clip(lower=1), 0.0, 0.05)
    parts = {"ghost": 0.95 * C["s_ghost"], "padding": 0.80 * C["s_pad"], "capacity": 0.95 * C["s_cap"], "diesel": 0.30 * C["s_diesel"], "intake": 0.15 * C["s_intake"]}
    C["score"] = 1 - (1 - parts["ghost"]) * (1 - parts["padding"]) * (1 - parts["capacity"]) * (1 - parts["diesel"]) * (1 - parts["intake"])
    for k, v in parts.items():
        C["w_" + k] = v
    return C.fillna({"score": 0.0})


def cbwtf_reasons(r: pd.Series) -> list[str]:
    out = []
    if r["w_ghost"] > 0.2:
        bits = [f"{int(r['n_' + k])} {lab}" for k, lab in [("in_closure", "during a declared closure"), ("off_cadence", "off the facility's collection rhythm"),
                                                         ("displaced", "with GPS displaced from the facility"), ("short_dwell", "with under 3 minutes at the stop"), ("duplicate_copy", "copying another pickup's weights with no GPS fix")] if r["n_" + k] > 0]
        out.append(f"{int(r['ghost_pickups'])} of {int(r['pickups']):,} recorded pickups ({r['ghost_rate']:.2%}) cannot have happened: " + ", ".join(bits))
    if r["w_padding"] > 0.2:
        out.append(f"{int(r['n_above_max'])} of the {int(r['n_facilities'])} facilities it serves record more waste than any legitimate site of their type can produce "
                   f"(excess z = {r['z_above']:.1f}), and {int(r['n_upper'])} sit in the upper tail ({r['upper_share']:.0%}, z = {r['z_upper']:.1f}); consistent with inflated recorded weights")
    if r["w_capacity"] > 0.2:
        out.append(f"Reported burning above physical capacity on {int(r['impossible_days'])} day(s) (more than {r['incinerator_kg_per_hr']:.0f} kg/hr x 24 h) "
                   f"and above its {r['permitted_kg_per_day']:,.0f} kg/day permit on {int(r['over_permit_burn_days'])} day(s)")
    if r["w_diesel"] > 0.2:
        out.append(f"Diesel bought is below the physical minimum on {r['diesel_low_share']:.0%} of days (median {r['diesel_med']:.3f} L per kg burned; plausible 0.08-0.20)")
    if r["w_intake"] > 0.1:
        out.append(f"Daily intake exceeded the permitted {r['permitted_kg_per_day']:,.0f} kg on {int(r['intake_over_permit_days'])} day(s)")
    out.append(f"Based on {int(r['pickups']):,} pickups and {int(r['days'])} daily treatment records")
    return out
