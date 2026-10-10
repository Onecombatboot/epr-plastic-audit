"""Mass-Balance & Yield agent data for e-waste, battery, tyre and used-oil recyclers.

Per entity-month: reported input / certificates, reported output sales by product,
and DISCOM electricity bills. Same truth-then-fabrication principle as plastic:
  * lazy fraud      -> inputs/certificates inflated, outputs stay true  (output-balance gap)
  * careful fraud   -> outputs fabricated to match (only energy/capacity expose it)
  * doctored bills  -> energy inflated too, but the bill's rupee maths is inconsistent
Benign confounders: campaign processing with inventory lag, atypical-but-legit yields,
captive solar, ramp-up, fast growers.

Also writes per-stream agent configs (stream_configs/*.yaml): "add a stream = add a config".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .bills import make_bills
from .common import fy_months, make_ids, rng_for, write_table

STREAMS = ["ewaste", "battery", "tyre", "used_oil"]
PRODUCT_PRICE = {  # INR per tonne, assumption (order of magnitude only)
    "ferrous": (22000, 32000), "plastics": (15000, 30000), "pcb": (150000, 450000), "copper_cables": (250000, 420000),
    "aluminium": (110000, 160000), "glass": (500, 3000), "residue": (0, 0), "lead": (160000, 195000),
    "polypropylene": (40000, 60000), "electrolyte_neutralised": (0, 0), "slag_residue": (0, 0), "black_mass": (180000, 420000),
    "copper": (600000, 780000), "steel_casing": (25000, 35000), "pyrolysis_oil": (32000, 45000), "carbon_black": (12000, 25000),
    "steel_wire": (22000, 30000), "pyro_gas": (0, 0), "rubber_granules": (14000, 24000), "textile_fibre": (0, 2000),
    "base_oil": (60000, 85000), "light_fuel": (35000, 50000), "asphalt_residue": (8000, 18000), "water_loss": (0, 0),
}
NON_SALEABLE = {"residue", "electrolyte_neutralised", "slag_residue", "pyro_gas", "water_loss"}


def stream_config(cal, stream) -> dict:
    sc = cal["streams"][stream]
    procs = {}
    for name, p in sc["processes"].items():
        tot = sum(p["outputs"].values())
        procs[name] = {
            "kwh_per_t": p["kwh_per_t"],
            "outputs": {k: {"expected": round(float(v / tot), 3),
                            "tolerance": round(float(max(0.03, 2.5 * np.sqrt((v / tot) * (1 - v / tot) / (p["conc"] + 1)))), 3),
                            "saleable": k not in NON_SALEABLE} for k, v in p["outputs"].items()},
        }
    return {"stream": stream, "certificate_basis": "input_tonnes_processed", "assumption": True,
            "note": "Yield/energy ranges are assumptions; cite published figures before quoting.", "processes": procs}


def generate(cal: dict, scale: str, seed: int, out: Path, manifest: dict, reg: dict) -> dict:
    rng = rng_for(seed, "yield")
    ents, phys = reg["entities"], reg["physics"]
    E = ents[ents["stream"].isin(STREAMS)].reset_index(drop=True)
    P = phys.set_index("entity_id").loc[E["entity_id"]].reset_index()
    n = len(E)
    ids = E["entity_id"].to_numpy()
    months = fy_months(cal)
    M = len(months)
    reg_m = pd.to_datetime(E["epr_registration_date"]).dt.to_period("M").dt.to_timestamp().to_numpy()
    active = months.to_numpy()[None, :] >= reg_m[:, None]
    age = np.cumsum(active, axis=1)
    cap_m = P["true_capacity_tpa"].to_numpy() / 12
    util = P["utilization_base"].to_numpy()
    fraud = P["is_fraud"].to_numpy()
    diff = P["difficulty"].to_numpy()
    hollow = P["hollow_fraction"].to_numpy()
    stream = E["stream"].to_numpy()
    proc = E["process_type"].to_numpy()

    season = np.ones((n, M))
    bat = stream == "battery"
    season[bat] *= (1 + 0.10 * np.isin(months.month, [11, 12, 1, 4, 5])[None, :])       # replacement peaks
    ramp = np.clip(age / 4.0, 0.3, 1.0)
    grow = np.ones((n, M))
    fast = P["fast_grower"].to_numpy()
    grow[fast] = np.linspace(0.45, 1.25, M)[None, :]
    true_in = cap_m[:, None] * util[:, None] * season * ramp * grow * rng.lognormal(0, 0.09, (n, M))
    # campaign processors: run hard some months, idle others (legit, but monthly-noisy)
    campaign = (~fraud) & (rng.random(n) < 0.12)
    for i in np.flatnonzero(campaign):
        on = rng.random(M) < 0.55
        true_in[i] = np.where(on, true_in[i] / 0.55, true_in[i] * 0.05)
    true_in = np.minimum(true_in, cap_m[:, None] * 0.98) * active
    pure = fraud & (hollow > 0.9)
    true_in[pure] *= rng.uniform(0.02, 0.08, (pure.sum(), 1))
    hollow_in = np.where(fraud[:, None], true_in / np.clip(1 - hollow[:, None], 0.03, None) - true_in, 0.0) * active
    for i in np.flatnonzero(pure):
        hollow_in[i] = np.maximum(cap_m[i] * rng.uniform(0.5, 1.2) * rng.lognormal(0, 0.1, M) * active[i] - true_in[i], 0)
    burst = P["deadline_burst"].to_numpy()
    fy_idx = np.repeat(np.arange(M // 12), 12)[:M]
    bw = np.where(np.isin(months.month, [2, 3]), 6.0, np.where(months.month == 1, 2.0, 0.35))
    for i in np.flatnonzero(burst & fraud):
        for f in range(fy_idx.max() + 1):
            mm = (fy_idx == f) & active[i]
            if mm.any():
                tot = hollow_in[i, mm].sum()
                w = bw[mm] * rng.uniform(0.8, 1.2, mm.sum())
                hollow_in[i, mm] = tot * w / w.sum()
    claimed_in = true_in + hollow_in

    # per-entity true yields
    atypical = (~fraud) & (rng.random(n) < 0.05)
    yrows = []
    prod_names = {}
    for i in range(n):
        p = cal["streams"][stream[i]]["processes"][proc[i]]
        names = list(p["outputs"])
        mean = np.array(list(p["outputs"].values()), float)
        conc = p["conc"] * (0.3 if atypical[i] else 1.0)
        yrows.append(rng.dirichlet(mean / mean.sum() * conc))
        prod_names[i] = names

    # outputs: sold with an inventory lag (campaign processors sell with longer lags)
    out_fab = P["output_fabrication"].to_numpy()
    orows = []
    for i in range(n):
        y = yrows[i]
        lag_p = [0.55, 0.30, 0.15] if campaign[i] else [0.85, 0.13, 0.02]
        basis = claimed_in[i] if (fraud[i] and out_fab[i]) else true_in[i]
        for k, pname in enumerate(prod_names[i]):
            if pname in NON_SALEABLE:
                continue
            q = basis * y[k] * rng.lognormal(0, 0.03, M)
            sold = np.zeros(M)
            for lag, pr in enumerate(lag_p):
                sold[lag:] += q[:M - lag] * pr
            lo, hi = PRODUCT_PRICE[pname]
            price = rng.uniform(lo, hi) if hi > 0 else 0.0
            for m in np.flatnonzero(sold > 0.01):
                orows.append((ids[i], months[m].date(), pname, round(sold[m], 3), round(sold[m] * price * rng.uniform(0.95, 1.05), 0)))
    outputs = pd.DataFrame(orows, columns=["entity_id", "month", "product", "tonnes_sold", "value_inr"])

    ops = pd.DataFrame({"entity_id": np.repeat(ids, M), "stream": np.repeat(stream, M), "process_type": np.repeat(proc, M),
                        "month": np.tile(months.date, n), "input_t_reported": claimed_in.ravel().round(3),
                        "_active": active.ravel()})
    ops["certificates_t"] = ops["input_t_reported"]
    ops = ops[ops["_active"]].drop(columns="_active").reset_index(drop=True)

    kwh = P["kwh_per_t"].to_numpy()
    bills = make_bills(rng, ids, E["state"].to_numpy(), E["connected_load_kw"].to_numpy(), kwh, P["solar_fraction"].to_numpy(),
                       fraud, diff, P["doctored_bill"].to_numpy(), true_in, claimed_in, active, months, reg["tariff"],
                       lambda k: make_ids(rng, "BILY", k, 8))

    truth = pd.DataFrame({"entity_id": np.repeat(ids, M), "month": np.tile(months.date, n), "true_input_t": true_in.ravel().round(3),
                          "hollow_t": hollow_in.ravel().round(3), "_a": active.ravel()})
    truth = truth[truth["_a"]].drop(columns="_a").reset_index(drop=True)
    ylab = pd.DataFrame({"entity_id": ids, "campaign_processor": campaign, "atypical_legit_yield": atypical,
                         "true_yields": [dict(zip(prod_names[i], np.round(yrows[i], 4))) for i in range(n)]})
    ylab["true_yields"] = ylab["true_yields"].map(lambda d: yaml.safe_dump({k: float(v) for k, v in d.items()}, default_flow_style=True).strip())

    d = out / "yield"
    L = out / "labels" / "yield"
    write_table(ops, d / "operations_monthly.parquet", manifest)
    write_table(outputs, d / "outputs_monthly.parquet", manifest)
    write_table(bills, d / "electricity_bills.parquet", manifest)
    write_table(truth, L / "monthly_truth.parquet", manifest)
    write_table(ylab, L / "entity_extra_labels.parquet", manifest)
    cfg_dir = d / "stream_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for s in STREAMS:
        with open(cfg_dir / f"{s}.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(stream_config(cal, s), f, sort_keys=False)
    pcfg = {"stream": "plastic", "certificate_basis": "input_tonnes_processed", "assumption": True,
            "processes": {k: {"kwh_per_t": v["kwh_per_t"], "outputs": {"recovered_material": {"expected": round(sum(v["yield"]) / 2, 3),
                                                                                              "tolerance": round((v["yield"][1] - v["yield"][0]) / 2 + 0.03, 3),
                                                                                              "saleable": v["yield"][1] > 0}}}
                          for k, v in cal["plastic"]["eol_types"].items()}}
    with open(cfg_dir / "plastic.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(pcfg, f, sort_keys=False)
    return {"ops": ops}
