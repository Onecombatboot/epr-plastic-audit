"""OCEMS (online continuous effluent monitoring) readings with planted tampering.

15-minute series of pH, COD, BOD, TSS and flow for industrial units and CETPs. Signals
are driven by a production schedule, smoothed by ETP retention, with AR(1) process noise,
genuine ETP upsets (real exceedances), sensor measurement noise and quantisation.

Benign data-quality problems (must NOT be confused with tampering): communication gaps,
power outages, calibration events (flagged or not), slow sensor drift, maintenance, and
genuine stuck sensors.

Tampering types (by difficulty): flatline, over-smoothing, replay (copy-paste of an
earlier compliant window), dilution, night bypass, clipping below the standard,
chemically implausible BOD/COD, and scaling. Tampering is preferentially placed over
genuine upsets (that is when operators have a motive).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import CLUSTERS, company_names, jitter_latlon, make_ids, rng_for, sample_clusters, write_table

TAMPER_TYPES = ["flatline", "smoothing", "replay", "dilution", "night_bypass", "clipping", "implausible_ratio", "scaling"]
PARAMS = ["ph", "cod_mgl", "bod_mgl", "tss_mgl", "flow_m3h"]


def ar1(rng, n, phi, sigma):
    e = rng.normal(0, 1, n)
    y = pd.Series(e).ewm(alpha=1 - phi, adjust=False).mean().to_numpy()
    s = y.std()
    return y / (s if s > 0 else 1) * sigma


def smooth(x, steps):
    return pd.Series(x).ewm(alpha=1.0 / max(steps, 1), adjust=False).mean().to_numpy()


def production(rng, schedule, T, hour, dow):
    if schedule == "continuous":
        p = 0.88 + ar1(rng, T, 0.995, 0.05)
    elif schedule == "two_shift":
        p = np.where((hour >= 6) & (hour < 22), 1.0, 0.15) * np.where(dow == 6, 0.2, 1.0)
    elif schedule == "three_shift":
        p = 0.9 - 0.25 * np.isin(hour, [6, 14, 22]) - np.where(dow == 6, 0.2, 0.0)
    else:  # batch
        p = np.full(T, 0.3)
        t = int(rng.integers(0, 40))
        while t < T:
            dur = int(rng.integers(16, 48))
            p[t:t + dur] = 1.0
            t += dur + int(rng.integers(16, 120))
    down = rng.random(T // 96 + 1) < 0.02          # random shutdown days
    p = p * np.where(np.repeat(down, 96)[:T], 0.05, 1.0)
    return np.clip(p, 0.02, 1.1)


def generate(cal: dict, scale: str, seed: int, out: Path, manifest: dict) -> dict:
    rng = rng_for(seed, "ocems")
    oc = cal["ocems"]
    sc = cal["scales"][scale]
    N, D = int(sc["ocems_units"]), int(sc["ocems_days"])
    step = oc["freq_minutes"]
    T = D * 24 * 60 // step
    ts = pd.date_range(oc["start"], periods=T, freq=f"{step}min")
    hour = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()
    std = oc["standards"]
    inds = list(oc["industries"])
    ind = rng.choice(inds, N, p=np.array([oc["industries"][k]["share"] for k in inds]))
    cl = sample_clusters(rng, N)
    lat, lon = jitter_latlon(rng, [CLUSTERS[c].lat for c in cl], [CLUSTERS[c].lon for c in cl], km=3)
    unit_ids = make_ids(rng, "UNIT", N)
    names, _ = company_names(rng, N, "used_oil")
    names = np.array([nm.replace("Petrochem", "Chemicals").replace("Lubricants", "Industries").replace("Oil Refinery", "Mills")
                      .replace("Re-Refiners", "Processors").replace("Hydrocarbons", "Organics").replace("Petro Products", "Textiles")
                      for nm in names], dtype=object)
    tamper_unit = rng.random(N) < oc["tamper_unit_rate"]
    dmix = cal["fraud"]["difficulty_mix"]
    unit_diff = np.where(tamper_unit, rng.choice(list(dmix), N, p=np.array(list(dmix.values()))), None)

    readings, row_labels, episodes, faults = [], [], [], []
    units_meta = []
    for u in range(N):
        spec = oc["industries"][ind[u]]
        p = production(rng, spec["schedule"], T, hour, dow)
        design = rng.uniform(*spec["flow_m3h"])
        flow = design * (0.25 + 0.75 * smooth(p, 4)) * np.exp(ar1(rng, T, 0.95, 0.04))
        c_mean = rng.uniform(*spec["cod_mgl"])
        retention = int(rng.integers(24, 96))
        load = smooth(p, retention)
        cod = c_mean * (0.75 + 0.5 * load) * np.exp(ar1(rng, T, 0.97, 0.10))
        # genuine ETP upsets -> real exceedances
        upset = np.ones(T)
        upset_windows = []
        for _ in range(rng.poisson(D / 20)):
            s0 = int(rng.integers(0, T - 8))
            dur = int(rng.integers(24, 192))
            peak = rng.uniform(1.3, 2.6)
            shape = np.sin(np.linspace(0, np.pi, min(dur, T - s0))) * (peak - 1) + 1
            upset[s0:s0 + len(shape)] *= shape
            upset_windows.append((s0, s0 + len(shape)))
        cod *= upset
        ratio = rng.uniform(*spec["cod_bod_ratio"]) * np.exp(ar1(rng, T, 0.999, 0.03) + ar1(rng, T, 0.5, 0.05))
        bod = cod / ratio
        tss = rng.uniform(30, 75) * (0.8 + 0.4 * load) * np.exp(ar1(rng, T, 0.97, 0.12)) * upset ** 0.7
        ph = rng.uniform(6.8, 7.8) + ar1(rng, T, 0.98, 0.08)
        for _ in range(rng.poisson(D / 10)):              # dosing excursions
            s0 = int(rng.integers(0, T - 12))
            ph[s0:s0 + int(rng.integers(4, 12))] += rng.choice([-1, 1]) * rng.uniform(0.4, 1.4)
        true = {"ph": ph.copy(), "cod_mgl": cod.copy(), "bod_mgl": bod.copy(), "tss_mgl": tss.copy(), "flow_m3h": flow.copy()}

        # measurement layer
        meas = {
            "ph": ph + rng.normal(0, 0.02, T),
            "cod_mgl": cod * (1 + rng.normal(0, 0.02, T)) + rng.normal(0, 1.0, T),
            "bod_mgl": bod * (1 + rng.normal(0, 0.03, T)),
            "tss_mgl": tss * (1 + rng.normal(0, 0.03, T)),
            "flow_m3h": flow * (1 + rng.normal(0, 0.015, T)),
        }
        status = np.array(["OK"] * T, dtype=object)
        tamper = np.array([""] * T, dtype=object)
        fault = np.array([""] * T, dtype=object)

        # ---------------- tampering (applied to measured values)
        if tamper_unit[u]:
            d = unit_diff[u]
            k = int(rng.integers(1, 5))
            types = rng.choice(TAMPER_TYPES, k, replace=k > len(TAMPER_TYPES))
            for t_type in types:
                if upset_windows and rng.random() < 0.6:
                    a, b = upset_windows[int(rng.integers(len(upset_windows)))]
                    s0 = max(0, a - int(rng.integers(0, 24)))
                else:
                    s0 = int(rng.integers(96, T - 96))
                dur_days = {"flatline": (0.25, 5), "smoothing": (2, 15), "replay": (1, 7), "dilution": (1, 10),
                            "night_bypass": (5, 30), "clipping": (3, 20), "implausible_ratio": (1, 5), "scaling": (5, 30)}[t_type]
                e0 = min(T, s0 + int(rng.uniform(*dur_days) * 96))
                if e0 - s0 < 8:
                    continue
                sl = slice(s0, e0)
                L = e0 - s0
                params = []
                if t_type == "flatline":
                    for prm in ["cod_mgl", "bod_mgl", "tss_mgl"]:
                        v = std[prm] * rng.uniform(0.35, 0.75)
                        jitter = 0.0 if d == "obvious" else (0.002 if d == "moderate" else 0.01)
                        meas[prm][sl] = v * (1 + rng.normal(0, jitter, L))
                        params.append(prm)
                elif t_type == "smoothing":
                    for prm in ["cod_mgl", "bod_mgl", "tss_mgl"]:
                        target = std[prm] * rng.uniform(0.5, 0.8)
                        keep = {"obvious": 0.05, "moderate": 0.15, "subtle": 0.35}[d]
                        x = meas[prm][sl]
                        sm = smooth(x, 96)
                        sm = sm * min(1.0, target / max(sm.mean(), 1e-6))     # heavy filter, pulled under the standard
                        meas[prm][sl] = sm + (x - smooth(x, 16)) * keep
                        params.append(prm)
                elif t_type == "replay":
                    src = max(0, s0 - L)
                    for _try in range(20):                    # find an earlier, fully compliant window to paste
                        cand = int(rng.integers(0, max(1, s0 - L)))
                        if cand + L <= s0 and (true["cod_mgl"][cand:cand + L] < std["cod_mgl"]).all():
                            src = cand
                            break
                    noise = 0.0 if d == "obvious" else (0.005 if d == "moderate" else 0.02)
                    for prm in ["cod_mgl", "bod_mgl", "tss_mgl", "ph"]:
                        meas[prm][sl] = meas[prm][src:src + L] * (1 + rng.normal(0, noise, L))
                        params.append(prm)
                elif t_type == "dilution":
                    f = {"obvious": rng.uniform(2.0, 3.0), "moderate": rng.uniform(1.5, 2.0), "subtle": rng.uniform(1.25, 1.5)}[d]
                    meas["flow_m3h"][sl] *= f
                    for prm in ["cod_mgl", "bod_mgl", "tss_mgl"]:
                        meas[prm][sl] /= f
                    params = ["flow_m3h", "cod_mgl", "bod_mgl", "tss_mgl"]
                elif t_type == "night_bypass":
                    night = ((hour[sl] >= 23) | (hour[sl] < 5))
                    f = {"obvious": 0.1, "moderate": 0.2, "subtle": 0.35}[d]
                    for prm in ["flow_m3h"]:
                        x = meas[prm][sl]
                        x[night] *= f
                        meas[prm][sl] = x
                    for prm in ["cod_mgl", "bod_mgl", "tss_mgl"]:
                        x = meas[prm][sl]
                        x[night] = np.minimum(x[night], std[prm] * rng.uniform(0.4, 0.7))
                        meas[prm][sl] = x
                    params = ["flow_m3h", "cod_mgl", "bod_mgl", "tss_mgl"]
                    tamper[s0:e0][night] = "night_bypass"
                elif t_type == "clipping":
                    cap = rng.uniform(0.88, 0.98)
                    for prm in ["cod_mgl", "bod_mgl", "tss_mgl"]:
                        x = meas[prm][sl]
                        over = x > std[prm] * cap
                        x[over] = std[prm] * cap * (1 - np.abs(rng.normal(0, 0.01 if d != "obvious" else 0.0, over.sum())))
                        meas[prm][sl] = x
                    params = ["cod_mgl", "bod_mgl", "tss_mgl"]
                elif t_type == "implausible_ratio":
                    if d == "obvious":
                        meas["bod_mgl"][sl] = meas["cod_mgl"][sl] * rng.uniform(1.05, 1.5)
                    else:
                        meas["bod_mgl"][sl] = meas["cod_mgl"][sl] / rng.uniform(12, 25)
                    params = ["bod_mgl"]
                elif t_type == "scaling":
                    f = rng.uniform(0.45, 0.7)
                    meas["cod_mgl"][sl] *= f
                    if d == "subtle":
                        meas["bod_mgl"][sl] *= f          # keeps the ratio plausible
                        params = ["cod_mgl", "bod_mgl"]
                    else:
                        params = ["cod_mgl"]
                if t_type != "night_bypass":
                    tamper[sl] = np.where(tamper[sl] == "", t_type, tamper[sl])
                episodes.append((unit_ids[u], ts[s0], ts[e0 - 1], t_type, d, ",".join(dict.fromkeys(params))))

        # ---------------- benign faults (applied after tampering; they also hit tamper units)
        for _ in range(rng.poisson(D / 7)):                     # communication gaps
            s0 = int(rng.integers(0, T))
            L = int(min(T - s0, rng.geometric(0.08)))
            for prm in PARAMS:
                meas[prm][s0:s0 + L] = np.nan
            status[s0:s0 + L] = "NODATA"
            fault[s0:s0 + L] = "comm_gap"
        for _ in range(rng.poisson(D / 45)):                    # power outage
            s0 = int(rng.integers(0, T))
            L = int(min(T - s0, rng.integers(16, 96)))
            for prm in PARAMS:
                meas[prm][s0:s0 + L] = np.nan
            status[s0:s0 + L] = "NODATA"
            fault[s0:s0 + L] = "power_outage"
        day0 = int(rng.integers(0, 7))
        for dd in range(day0, D, int(rng.integers(6, 9))):     # calibration
            s0 = dd * 96 + int(rng.integers(40, 64))
            if s0 + 4 >= T:
                break
            L = int(rng.integers(2, 5))
            meas["cod_mgl"][s0:s0 + L] = rng.choice([0.0, 100.0, 200.0]) + rng.normal(0, 0.5, L)
            meas["bod_mgl"][s0:s0 + L] = rng.choice([0.0, 20.0]) + rng.normal(0, 0.2, L)
            flagged = rng.random() < 0.7
            status[s0:s0 + L] = "CAL" if flagged else status[s0:s0 + L]
            fault[s0:s0 + L] = "calibration_flagged" if flagged else "calibration_unflagged"
        for _ in range(rng.poisson(D / 30)):                    # slow drift, reset later
            s0 = int(rng.integers(0, T - 96))
            L = int(min(T - s0, rng.integers(3 * 96, 10 * 96)))
            drift = np.linspace(0, rng.uniform(-0.15, 0.15), L)
            meas["cod_mgl"][s0:s0 + L] *= (1 + drift)
            fault[s0:s0 + L] = np.where(fault[s0:s0 + L] == "", "sensor_drift", fault[s0:s0 + L])
        for _ in range(rng.poisson(D / 40)):                    # maintenance (flagged flatline)
            s0 = int(rng.integers(0, T - 8))
            L = int(rng.integers(4, 24))
            for prm in ["cod_mgl", "bod_mgl", "tss_mgl"]:
                v = meas[prm][s0 - 1] if s0 > 0 else meas[prm][0]
                meas[prm][s0:s0 + L] = v
            status[s0:s0 + L] = "MAINT"
            fault[s0:s0 + L] = "maintenance"
        if rng.random() < 0.06:                                  # genuine stuck sensor (unflagged!)
            s0 = int(rng.integers(0, T - 300))
            L = int(rng.integers(24, 288))
            prm = rng.choice(["cod_mgl", "ph", "tss_mgl"])
            meas[prm][s0:s0 + L] = meas[prm][s0 - 1] if s0 else meas[prm][0]
            fault[s0:s0 + L] = "stuck_sensor"
            faults.append((unit_ids[u], ts[s0], ts[s0 + L - 1], "stuck_sensor", prm))

        q = {"ph": 0.01, "cod_mgl": 0.1, "bod_mgl": 0.1, "tss_mgl": 0.1, "flow_m3h": 0.1}
        for prm in PARAMS:
            x = np.round(meas[prm] / q[prm]) * q[prm]
            meas[prm] = np.where(np.isnan(x), np.nan, np.maximum(x, 0 if prm != "ph" else 0))
        exceed_true = (true["cod_mgl"] > std["cod_mgl"]) | (true["bod_mgl"] > std["bod_mgl"]) | (true["tss_mgl"] > std["tss_mgl"])
        rep_ok = (np.nan_to_num(meas["cod_mgl"], nan=0) <= std["cod_mgl"]) & (np.nan_to_num(meas["bod_mgl"], nan=0) <= std["bod_mgl"]) & \
                 (np.nan_to_num(meas["tss_mgl"], nan=0) <= std["tss_mgl"])
        readings.append(pd.DataFrame({"unit_id": unit_ids[u], "ts": ts, **{k: meas[k].astype("float32") for k in PARAMS}, "status": status}))
        row_labels.append(pd.DataFrame({"unit_id": unit_ids[u], "ts": ts, "tamper_type": tamper, "fault_type": fault,
                                        "true_exceedance": exceed_true, "hidden_exceedance": exceed_true & rep_ok & (tamper != ""),
                                        "true_cod_mgl": true["cod_mgl"].astype("float32"), "true_bod_mgl": true["bod_mgl"].astype("float32"),
                                        "true_flow_m3h": true["flow_m3h"].astype("float32")}))
        units_meta.append((unit_ids[u], names[u], ind[u], CLUSTERS[cl[u]].state, CLUSTERS[cl[u]].district, CLUSTERS[cl[u]].estate,
                           round(float(lat[u]), 6), round(float(lon[u]), 6), round(design, 1), spec["schedule"],
                           std["cod_mgl"], std["bod_mgl"], std["tss_mgl"], std["ph_min"], std["ph_max"],
                           rng.choice(["Analyzer-A", "Analyzer-B", "Analyzer-C", "Analyzer-D"])))

    units = pd.DataFrame(units_meta, columns=["unit_id", "name", "industry", "state", "district", "estate", "lat", "lon",
                                              "design_flow_m3h", "operating_schedule", "std_cod_mgl", "std_bod_mgl", "std_tss_mgl",
                                              "std_ph_min", "std_ph_max", "analyzer_model"])
    unit_labels = pd.DataFrame({"unit_id": unit_ids, "is_tamper_unit": tamper_unit, "difficulty": unit_diff})
    ep = pd.DataFrame(episodes, columns=["unit_id", "start_ts", "end_ts", "tamper_type", "difficulty", "params_affected"])
    fl = pd.DataFrame(faults, columns=["unit_id", "start_ts", "end_ts", "fault_type", "param"])
    d = out / "ocems"
    L = out / "labels" / "ocems"
    write_table(units, d / "units.parquet", manifest)
    write_table(pd.concat(readings, ignore_index=True), d / "readings.parquet", manifest)
    write_table(pd.concat(row_labels, ignore_index=True), L / "row_labels.parquet", manifest)
    write_table(ep, L / "tamper_episodes.parquet", manifest)
    write_table(fl, L / "benign_fault_episodes.parquet", manifest)
    write_table(unit_labels, L / "unit_labels.parquet", manifest)
    return {"units": units}
