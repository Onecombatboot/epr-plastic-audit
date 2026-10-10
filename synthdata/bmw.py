"""Biomedical waste (BMW) world: healthcare facilities -> barcode-tracked pickups -> CBWTFs.

Truth: daily waste generated per facility (beds x occupancy x kg/bed/day, or patients x
kg/patient), seasonal vector-borne disease uplift, weekday effects. Pickups collect the
waste accumulated since the previous pickup; CBWTF vehicles log GPS stops.

Fraud
  facility  under_handover   : hands over only part of its waste (rest dumped/burnt)
  CBWTF     ghost_pickup     : records pickups that never happened (GPS absent/far/zero dwell)
  CBWTF     weight_padding   : inflates recorded kg at some facilities (billing by weight)
  CBWTF     overcapacity     : intake beyond incinerator capacity, reported as fully treated
                               (diesel bought is too low for the kg claimed incinerated)
Benign: bed expansions, temporary closures (both declared), GPS device failures,
occupancy swings, dengue-season peaks, small clinics with genuinely tiny volumes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import CLUSTERS, haversine_km, jitter_latlon, make_ids, PREFIX, rng_for, sample_clusters, write_table

CATS = ["yellow", "red", "white", "blue"]
BAG_KG = {"yellow": 6.0, "red": 5.0, "white": 3.0, "blue": 8.0}
TYPE_WORD = {"hospital_large": "Hospital", "hospital_medium": "Hospital", "nursing_home": "Nursing Home", "clinic": "Clinic",
             "lab": "Diagnostics", "dental": "Dental Care", "veterinary": "Veterinary Hospital"}


def generate(cal: dict, scale: str, seed: int, out: Path, manifest: dict) -> dict:
    rng = rng_for(seed, "bmw")
    bc = cal["bmw"]
    sc = cal["scales"][scale]
    N, D, NC = int(sc["bmw_facilities"]), int(sc["bmw_days"]), int(sc["bmw_cbwtf"])
    days = pd.date_range(bc["start"], periods=D, freq="D")
    dow = days.dayofweek.to_numpy()
    month = days.month.to_numpy()
    dmix = cal["fraud"]["difficulty_mix"]
    dnames, dp = list(dmix), np.array(list(dmix.values()))

    # ------------------------------------------------------------------ CBWTFs
    c_cl = sample_clusters(rng, NC)
    c_lat, c_lon = jitter_latlon(rng, [CLUSTERS[c].lat for c in c_cl], [CLUSTERS[c].lon for c in c_cl], km=15)
    c_ids = make_ids(rng, "CBWTF", NC)
    c_inc = rng.uniform(*bc["incinerator_kg_per_hr"], NC).round(-1)
    c_ghost = rng.random(NC) < bc["ghost_pickup_cbwtf_rate"]
    c_pad = rng.random(NC) < bc["weight_padding_cbwtf_rate"]
    c_over = rng.random(NC) < bc["overcapacity_cbwtf_rate"]
    c_diff = rng.choice(dnames, NC, p=dp / dp.sum())
    c_state = np.array([CLUSTERS[c].state for c in c_cl])

    # ------------------------------------------------------------------ facilities
    ftypes = list(bc["facility_types"])
    ft = rng.choice(ftypes, N, p=np.array([bc["facility_types"][t]["share"] for t in ftypes]))
    f_cl = sample_clusters(rng, N)
    f_lat, f_lon = jitter_latlon(rng, [CLUSTERS[c].lat for c in f_cl], [CLUSTERS[c].lon for c in f_cl], km=10)
    f_state = np.array([CLUSTERS[c].state for c in f_cl])
    # assign each facility to the nearest CBWTF in its state (else nearest overall)
    cb_of = np.zeros(N, int)
    for s in np.unique(f_state):
        fi = np.flatnonzero(f_state == s)
        ci = np.flatnonzero(c_state == s)
        if len(ci) == 0:
            ci = np.arange(NC)
        dist = haversine_km(f_lat[fi, None], f_lon[fi, None], c_lat[None, ci], c_lon[None, ci])
        cb_of[fi] = ci[np.argmin(dist, axis=1)]
    beds = np.zeros(N, int)
    pats = np.zeros(N)
    rate = np.zeros(N)
    pickup = np.empty(N, dtype=object)
    for t in ftypes:
        idx = ft == t
        spec = bc["facility_types"][t]
        pickup[idx] = spec["pickup"]
        if "beds" in spec:
            beds[idx] = np.round(np.exp(rng.uniform(np.log(spec["beds"][0]), np.log(spec["beds"][1]), idx.sum())))
            rate[idx] = rng.uniform(*spec["kg_per_bed_day"], idx.sum())
        else:
            pats[idx] = np.round(np.exp(rng.uniform(np.log(spec["patients_day"][0]), np.log(spec["patients_day"][1]), idx.sum())))
            rate[idx] = rng.uniform(*spec["kg_per_patient"], idx.sum())
    occ = rng.uniform(*bc["occupancy"], N)
    split = rng.dirichlet(np.array(list(bc["category_split_mean"].values())) * 60, N)
    f_ids = make_ids(rng, "HCF", N)
    names = [f"{a} {TYPE_WORD[t]}" if rng.random() < 0.6 else f"{a} {b} {TYPE_WORD[t]}"
             for a, b, t in zip(rng.choice(PREFIX, N), rng.choice(["Care", "Life", "City", "Health", "Medicare", "Multispeciality", "Sanjivani"], N), ft)]
    under = rng.random(N) < bc["under_handover_rate"]
    f_diff = np.where(under, rng.choice(dnames, N, p=dp / dp.sum()), None)
    hand_frac = np.ones(N)
    hand_frac[under] = [rng.uniform(*{"obvious": (0.25, 0.5), "moderate": (0.5, 0.7), "subtle": (0.7, 0.85)}[d]) for d in f_diff[under]]
    intermittent = under & (rng.random(N) < 0.4)

    # declared changes (benign)
    changes = []
    expand = rng.random(N) < 0.03
    expand &= beds > 0
    exp_day = rng.integers(30, D - 30, N)
    exp_factor = rng.uniform(1.2, 1.6, N)
    closed = rng.random(N) < 0.015
    close_start = rng.integers(10, D - 50, N)
    close_len = rng.integers(7, 45, N)
    for i in np.flatnonzero(expand):
        changes.append((f_ids[i], days[exp_day[i]].date(), "bed_expansion", int(beds[i]), int(round(beds[i] * exp_factor[i]))))
    for i in np.flatnonzero(closed):
        changes.append((f_ids[i], days[close_start[i]].date(), "temporary_closure", int(close_len[i]), None))

    # vehicles: each CBWTF runs ceil(n_facilities / 60) vehicles
    veh_of = np.empty(N, dtype=object)
    veh_ids = []
    for c in range(NC):
        fi = np.flatnonzero(cb_of == c)
        nv = max(1, int(np.ceil(len(fi) / 60)))
        vids = [f"{c_ids[c]}-V{k + 1:02d}" for k in range(nv)]
        veh_ids += vids
        if len(fi):
            veh_of[fi] = np.array(vids, dtype=object)[rng.integers(0, nv, len(fi))]

    season = np.where(np.isin(month, bc["vector_borne_season"]["months"]), 1.0, 0.0)
    st_uplift = {s: rng.uniform(*bc["vector_borne_season"]["uplift"]) for s in np.unique(f_state)}

    rows = []
    lab_rows = []
    cb_true_intake = np.zeros((NC, D))
    for i in range(N):
        up = st_uplift[f_state[i]]
        s_mult = 1 + season * (up - 1)
        if beds[i] > 0:
            b = np.where(np.arange(D) >= exp_day[i], beds[i] * exp_factor[i], beds[i]) if expand[i] else np.full(D, beds[i])
            o = np.clip(occ[i] + pd.Series(rng.normal(0, 1, D)).ewm(alpha=0.05, adjust=False).mean().to_numpy() * 0.25, 0.3, 1.0)
            gen = b * o * rate[i] * s_mult * np.where(dow == 6, 0.9, 1.0) * rng.lognormal(0, 0.12, D)
        else:
            closed_day = {"clinic": 0.7, "lab": 0.5, "dental": 0.9, "veterinary": 0.6}[ft[i]]
            p = pats[i] * np.where(dow == 6, 1 - closed_day, 1.0) * s_mult * rng.lognormal(0, 0.2, D)
            gen = p * rate[i]
        if closed[i]:
            gen[close_start[i]:close_start[i] + close_len[i]] = 0.0
        # pickup calendar
        if pickup[i] == "daily":
            pdays = np.flatnonzero(~((dow == 6) & (rng.random(D) < 0.2)))
        elif pickup[i] == "alternate":
            pdays = np.arange(int(rng.integers(0, 2)), D, 2)
        else:
            pdays = np.flatnonzero(np.isin(dow, [0, 3]))
        if closed[i]:
            pdays = pdays[(pdays < close_start[i]) | (pdays >= close_start[i] + close_len[i])]
        if len(pdays) == 0:
            continue
        cum = np.concatenate([[0.0], np.cumsum(gen)])
        prev = np.concatenate([[0], pdays[:-1] + 1])
        acc = cum[pdays + 1] - cum[prev]
        hf = np.full(len(pdays), hand_frac[i])
        if intermittent[i]:
            hf = np.where(rng.random(len(pdays)) < 0.35, rng.uniform(0.15, 0.5, len(pdays)), 1.0)
        hf = np.minimum(hf * rng.uniform(0.98, 1.0, len(pdays)), 1.0)
        handed = acc * hf
        c = cb_of[i]
        np.add.at(cb_true_intake[c], pdays, handed)
        pad = np.ones(len(pdays))
        if c_pad[c] and rng.random() < 0.3:
            pad = np.full(len(pdays), {"obvious": rng.uniform(1.4, 1.7), "moderate": rng.uniform(1.25, 1.4), "subtle": rng.uniform(1.12, 1.25)}[c_diff[c]])
        rec = handed * pad * (1 + rng.normal(0, 0.02, len(pdays)))
        catw = rec[:, None] * split[i][None, :] * rng.lognormal(0, 0.08, (len(pdays), 4))
        catw = catw / catw.sum(1, keepdims=True) * rec[:, None]
        catw = np.round(np.maximum(catw, 0), 2)
        nb = np.maximum(1, np.ceil(catw / np.array([BAG_KG[k] for k in CATS])[None, :]).sum(1)).astype(int)
        has_gps = rng.random(len(pdays)) > 0.03
        g_lat, g_lon = jitter_latlon(rng, np.full(len(pdays), f_lat[i]), np.full(len(pdays), f_lon[i]), km=0.05)
        arr = days[pdays].to_numpy() + (rng.uniform(7, 18, len(pdays)) * 3600).astype("timedelta64[s]")
        dwell = np.round(rng.uniform(3, 15, len(pdays)), 1)
        rows.append(pd.DataFrame({
            "facility_id": f_ids[i], "cbwtf_id": c_ids[c], "vehicle_id": veh_of[i], "pickup_date": days[pdays].date,
            **{f"{k}_kg": catw[:, j] for j, k in enumerate(CATS)}, "n_bags": nb,
            "gps_lat": np.where(has_gps, g_lat.round(6), np.nan), "gps_lon": np.where(has_gps, g_lon.round(6), np.nan),
            "gps_arrival": np.where(has_gps, arr, np.datetime64("NaT", "s")), "dwell_min": np.where(has_gps, dwell, np.nan)}))
        lab_rows.append(pd.DataFrame({"facility_id": f_ids[i], "pickup_date": days[pdays].date, "true_accumulated_kg": acc.round(3),
                                      "handed_kg": handed.round(3), "padding_factor": pad.round(3), "is_ghost": False}))

        # ghost pickups recorded by a fraudulent CBWTF on days nobody came
        if c_ghost[c] and rng.random() < 0.18 and len(pdays) > 5:
            d = c_diff[c]
            cand = np.setdiff1d(np.arange(D), pdays)
            if closed[i]:
                cand = np.union1d(cand[:0], np.arange(close_start[i], min(D, close_start[i] + close_len[i])))
            k = min(len(cand), int(rng.integers(5, 40)))
            if k == 0:
                continue
            gd = np.sort(rng.choice(cand, k, replace=False))
            src = rng.integers(0, len(pdays), k)
            gw = catw[src] * (1.0 if d == "obvious" else rng.uniform(0.9, 1.1, (k, 1)))
            gw = np.round(gw, 2)
            if d == "obvious":
                glat = np.full(k, np.nan); glon = np.full(k, np.nan); gdw = np.full(k, np.nan); garr = np.full(k, np.datetime64("NaT", "s"))
            else:
                off_km = rng.uniform(2, 10, k) if d == "moderate" else rng.uniform(0.1, 0.3, k)
                ang = rng.uniform(0, 2 * np.pi, k)
                glat = f_lat[i] + off_km / 111 * np.sin(ang)
                glon = f_lon[i] + off_km / 111 * np.cos(ang) / np.cos(np.radians(f_lat[i]))
                gdw = np.round(rng.uniform(0, 1.2, k), 1) if d == "subtle" else np.round(rng.uniform(0, 6, k), 1)
                garr = days[gd].to_numpy() + (rng.uniform(7, 18, k) * 3600).astype("timedelta64[s]")
            rows.append(pd.DataFrame({
                "facility_id": f_ids[i], "cbwtf_id": c_ids[c], "vehicle_id": veh_of[i], "pickup_date": days[gd].date,
                **{f"{kk}_kg": gw[:, j] for j, kk in enumerate(CATS)}, "n_bags": nb[src],
                "gps_lat": np.round(glat, 6), "gps_lon": np.round(glon, 6), "gps_arrival": garr, "dwell_min": gdw}))
            lab_rows.append(pd.DataFrame({"facility_id": f_ids[i], "pickup_date": days[gd].date, "true_accumulated_kg": 0.0,
                                          "handed_kg": 0.0, "padding_factor": 1.0, "is_ghost": True}))

    pickups = pd.concat(rows, ignore_index=True)
    plabels = pd.concat(lab_rows, ignore_index=True)
    order = np.lexsort((pickups["facility_id"].to_numpy(), pd.to_datetime(pickups["pickup_date"]).to_numpy()))
    pickups = pickups.iloc[order].reset_index(drop=True)
    plabels = plabels.iloc[order].reset_index(drop=True)
    pid = make_ids(rng, "PKP", len(pickups), 9)
    pickups.insert(0, "pickup_id", pid)
    plabels.insert(0, "pickup_id", pid)

    # ------------------------------------------------------------------ CBWTF daily treatment log
    rec_by_c = pickups.assign(tot=pickups[[f"{k}_kg" for k in CATS]].sum(1), di=(pd.to_datetime(pickups["pickup_date"]) - days[0]).dt.days) \
        .groupby(["cbwtf_id", "di"])[["yellow_kg", "red_kg", "white_kg", "blue_kg", "tot"]].sum()
    c_pos = {c: k for k, c in enumerate(c_ids)}
    diesel_rate = rng.uniform(*bc["diesel_l_per_kg_incinerated"], NC)
    t_rows, truth_rows = [], []
    for c in range(NC):
        try:
            g = rec_by_c.loc[c_ids[c]]
        except KeyError:
            continue
        y = np.zeros(D); r = np.zeros(D); tot = np.zeros(D)
        y[g.index.to_numpy()] = g["yellow_kg"].to_numpy()
        r[g.index.to_numpy()] = (g["red_kg"] + g["white_kg"] + g["blue_kg"]).to_numpy()
        tot[g.index.to_numpy()] = g["tot"].to_numpy()
        if (y > 0).any():
            if c_over[c]:
                # undersized incinerator: rated capacity below the yellow intake it reports burning
                f_ = {"obvious": rng.uniform(0.45, 0.6), "moderate": rng.uniform(0.6, 0.8), "subtle": rng.uniform(0.8, 0.95)}[c_diff[c]]
                c_inc[c] = max(10.0, round(np.percentile(y[y > 0], 50) / 20.0 * f_, -1))
            else:
                c_inc[c] = max(c_inc[c], np.ceil((y.max() / 18.0 + 5) / 10) * 10)   # sized above the peak day
        cap_day = c_inc[c] * rng.uniform(19, 21.5)
        true_inc = np.minimum(y, cap_day)
        untreated = y - true_inc
        hours_true = true_inc / c_inc[c]
        hours_rep = hours_true if not c_over[c] else np.minimum(y / c_inc[c], 24.0)
        diesel = true_inc * diesel_rate[c] * rng.uniform(0.95, 1.05, D)
        t_rows.append(pd.DataFrame({"cbwtf_id": c_ids[c], "date": days.date, "intake_kg": tot.round(2),
                                    "incinerated_kg_reported": y.round(2), "autoclaved_kg_reported": r.round(2),
                                    "incinerator_hours_reported": hours_rep.round(2), "diesel_litres_purchased": diesel.round(1)}))
        truth_rows.append(pd.DataFrame({"cbwtf_id": c_ids[c], "date": days.date, "true_incinerated_kg": true_inc.round(2),
                                        "untreated_kg": untreated.round(2)}))
    treatment = pd.concat(t_rows, ignore_index=True)
    treat_truth = pd.concat(truth_rows, ignore_index=True)

    facilities = pd.DataFrame({"facility_id": f_ids, "name": names, "facility_type": ft, "beds": beds,
                               "avg_daily_patients": pats.astype(int), "state": f_state,
                               "district": [CLUSTERS[c].district for c in f_cl], "lat": f_lat.round(6), "lon": f_lon.round(6),
                               "cbwtf_id": c_ids[cb_of], "pickup_schedule": pickup,
                               "authorisation_no": [f"BMW/{CLUSTERS[c].gst_code}/{rng.integers(10000, 99999)}" for c in f_cl]})
    cbwtfs = pd.DataFrame({"cbwtf_id": c_ids, "name": [f"{a} Biomedical Waste Treatment Facility" for a in rng.choice(PREFIX, NC)],
                           "state": c_state, "district": [CLUSTERS[c].district for c in c_cl], "lat": c_lat.round(6), "lon": c_lon.round(6),
                           "incinerator_kg_per_hr": c_inc, "permitted_kg_per_day": (c_inc * 20).round(-1),
                           "n_vehicles": [sum(v.startswith(cid + "-") for v in veh_ids) for cid in c_ids]})
    fac_labels = pd.DataFrame({"facility_id": f_ids, "under_handover": under, "handover_fraction": hand_frac.round(3),
                               "intermittent": intermittent, "difficulty": f_diff, "true_kg_rate": rate.round(5),
                               "true_occupancy_base": occ.round(3)})
    cb_labels = pd.DataFrame({"cbwtf_id": c_ids, "ghost_pickups": c_ghost, "weight_padding": c_pad, "overcapacity": c_over,
                              "difficulty": np.where(c_ghost | c_pad | c_over, c_diff, None)})
    d = out / "bmw"
    L = out / "labels" / "bmw"
    write_table(facilities, d / "facilities.parquet", manifest)
    write_table(cbwtfs, d / "cbwtfs.parquet", manifest)
    write_table(pd.DataFrame(changes, columns=["facility_id", "effective_date", "change_type", "old_value", "new_value"]),
                d / "facility_changes.parquet", manifest)
    write_table(pickups, d / "pickups.parquet", manifest)
    write_table(treatment, d / "cbwtf_daily_treatment.parquet", manifest)
    write_table(fac_labels, L / "facility_labels.parquet", manifest)
    write_table(cb_labels, L / "cbwtf_labels.parquet", manifest)
    write_table(plabels, L / "pickup_labels.parquet", manifest)
    write_table(treat_truth, L / "cbwtf_treatment_truth.parquet", manifest)
    return {"pickups": pickups}
