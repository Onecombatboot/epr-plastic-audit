"""Monthly DISCOM electricity bills shared by every recycler stream.

Units billed follow the TRUE physical throughput (bills come from the DISCOM). Fraudsters
with `doctored_bill` upload bills whose units are inflated to match their claims:
moderate/obvious leave the rupee amount unedited (units x tariff != amount), subtle ones
edit the amount too but with a stale tariff (small, systematic rate mismatch).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DISCOMS = {"Gujarat": ["DGVCL", "MGVCL", "PGVCL", "UGVCL", "Torrent Power"], "Maharashtra": ["MSEDCL", "Adani Electricity"],
           "Delhi": ["BSES Rajdhani", "BSES Yamuna", "Tata Power-DDL"], "Haryana": ["DHBVN", "UHBVN"],
           "Uttar Pradesh": ["PVVNL", "MVVNL", "KESCo", "NPCL"], "Tamil Nadu": ["TANGEDCO"], "Karnataka": ["BESCOM", "CESC Mysore", "HESCOM"],
           "Telangana": ["TSSPDCL"], "Andhra Pradesh": ["APEPDCL", "APSPDCL"], "West Bengal": ["WBSEDCL", "CESC"],
           "Rajasthan": ["JVVNL", "JdVVNL", "AVVNL"], "Punjab": ["PSPCL"], "Madhya Pradesh": ["MPPKVVCL", "MPMKVVCL"],
           "Uttarakhand": ["UPCL"], "Kerala": ["KSEB"], "Odisha": ["TPCODL"], "Himachal Pradesh": ["HPSEBL"]}
FYS = ["FY2023-24", "FY2024-25", "FY2025-26"]


def fy_of(ts: pd.Timestamp) -> str:
    y = ts.year if ts.month >= 4 else ts.year - 1
    return f"FY{y}-{(y + 1) % 100:02d}"


def make_tariff(rng, pc, states) -> pd.DataFrame:
    rows = []
    for s in sorted(states):
        e0 = rng.uniform(*pc["tariff_by_state_inr_per_kwh"])
        f0 = rng.uniform(*pc["fixed_charge_inr_per_kw_month"])
        du = rng.uniform(*pc["electricity_duty"])
        for fy, mult in zip(FYS, [1.0, 1.035, 1.07]):
            rows.append((s, fy, round(e0 * mult, 2), round(f0 * mult, 0), round(du, 3)))
    return pd.DataFrame(rows, columns=["state", "fy", "energy_rate_inr_per_kwh", "fixed_rate_inr_per_kw_month", "duty_rate"])


def make_bills(rng, ids, states, load, kwh, solar, fraud, diff, doctored, true_t, claimed_t, active, months, tariff,
               id_prefix_rng_ids) -> pd.DataFrame:
    tkey = {(r.state, r.fy): (r.energy_rate_inr_per_kwh, r.fixed_rate_inr_per_kw_month, r.duty_rate) for r in tariff.itertuples()}
    fy_m = [fy_of(m) for m in months]
    n = len(ids)
    base_frac = rng.uniform(0.03, 0.08, n)
    consumer_no = [str(rng.integers(10**9, 10**10)) for _ in range(n)]
    discom = [rng.choice(DISCOMS.get(s, ["DISCOM"])) for s in states]
    rows = []
    for i in range(n):
        for m in np.flatnonzero(active[i]):
            miss_p = 0.30 if (fraud[i] and diff[i] == "obvious") else 0.02
            if rng.random() < miss_p:
                continue
            base = load[i] * 720 * base_frac[i]
            true_units = (true_t[i, m] * kwh[i] + base) * (1 - solar[i]) * rng.uniform(0.96, 1.04)
            er, fr, du = tkey[(states[i], fy_m[m])]
            fixed = load[i] * fr
            units_rep = amount_units = true_units
            if doctored[i]:
                units_rep = (claimed_t[i, m] * kwh[i] + base) * rng.uniform(0.95, 1.05)
                if diff[i] == "subtle":
                    amount_units = units_rep
                    er = er * rng.uniform(0.86, 0.93)
            energy = amount_units * er
            duty = (energy + fixed) * du
            rows.append((ids[i], months[m].date(), discom[i], consumer_no[i], load[i], round(units_rep, 0),
                         round(fixed, 2), round(energy, 2), round(duty, 2), round(fixed + energy + duty, 0)))
    bills = pd.DataFrame(rows, columns=["entity_id", "billing_month", "discom", "consumer_no", "sanctioned_load_kw", "units_kwh",
                                        "fixed_charge_inr", "energy_charge_inr", "duty_inr", "total_amount_inr"])
    bills.insert(0, "bill_id", id_prefix_rng_ids(len(bills)))
    return bills
