"""Plastic EPR world (Phantom Plastic): collectors -> recyclers -> certificate lots -> brands,
plus recycled-output sales, monthly electricity bills, and reported monthly processing.

Design principle: simulate the TRUE physical operations first, then the REPORTED world
= truth + fabrication. Every fraud signal (capacity overrun, energy deficit, ghost
feedstock, below-cost pricing, ...) therefore *emerges* from the simulation instead of
being painted on, and clean entities are physically consistent by construction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .bills import make_bills
from .common import (CLUSTERS, STATE_OF_CLUSTER, addresses, company_names, fy_months, gstins, haversine_km,
                     jitter_latlon, make_ids, pan_numbers, phones, rng_for, sample_clusters, vehicle_regs, write_table)

CATS = ["CAT_I", "CAT_II", "CAT_III", "CAT_IV"]
SECTOR_CAT_MIX = {
    "Beverages": [0.80, 0.15, 0.05, 0.00], "Snacks": [0.10, 0.30, 0.60, 0.00], "Personal Care": [0.55, 0.30, 0.15, 0.00],
    "FMCG": [0.40, 0.35, 0.23, 0.02], "Dairy": [0.45, 0.45, 0.10, 0.00], "Pharma": [0.60, 0.25, 0.15, 0.00],
    "Retail": [0.30, 0.58, 0.10, 0.02], "Home Care": [0.60, 0.30, 0.10, 0.00], "Foods": [0.20, 0.40, 0.38, 0.02],
    "Consumer Products": [0.50, 0.35, 0.13, 0.02],
}
GST_RATE = 0.18


def _month_end(ts):
    return ts + pd.offsets.MonthEnd(0)


def generate(cal: dict, scale: str, seed: int, out: Path, manifest: dict, reg: dict) -> dict:
    rng = rng_for(seed, "plastic")
    pc = cal["plastic"]
    ents = reg["entities"]
    phys = reg["physics"]
    E = ents[ents["stream"] == "plastic"].reset_index(drop=True)
    P = phys.set_index("entity_id").loc[E["entity_id"]].reset_index()
    nR = len(E)
    months = fy_months(cal)
    M = len(months)
    reg_month = pd.to_datetime(E["epr_registration_date"]).dt.to_period("M").dt.to_timestamp().to_numpy()
    active = months.to_numpy()[None, :] >= reg_month[:, None]          # [nR, M]
    age = np.cumsum(active, axis=1)                                      # months since activation

    # ------------------------------------------------------------------ true monthly processing
    cap_m = P["true_capacity_tpa"].to_numpy() / 12.0
    util = P["utilization_base"].to_numpy()
    monsoon = rng.uniform(*pc["monsoon_factor"], nR)
    is_monsoon = np.isin(months.month, [7, 8, 9])
    seasonal = np.where(is_monsoon[None, :], monsoon[:, None], 1.0) * (1 + 0.04 * np.isin(months.month, [10, 11, 12])[None, :])
    ramp = np.clip(age / 4.0, 0.3, 1.0)
    grow = np.ones((nR, M))
    fast = P["fast_grower"].to_numpy()
    grow[fast] = np.linspace(0.45, 1.25, M)[None, :]
    noise = rng.lognormal(0, 0.08, (nR, M))
    true_t = cap_m[:, None] * util[:, None] * seasonal * ramp * grow * noise
    true_t = np.minimum(true_t, cap_m[:, None] * 0.98) * active
    fraud = P["is_fraud"].to_numpy()
    hollow_frac = P["hollow_fraction"].to_numpy()
    # pure paper shells (obvious, very high hollow share) run almost nothing physically
    pure_shell = fraud & (hollow_frac > 0.9)
    true_t[pure_shell] *= rng.uniform(0.02, 0.08, (pure_shell.sum(), 1))
    true_t = true_t.round(3)

    eol = E["process_type"].to_numpy()
    mix = np.zeros((nR, 4))
    for k in pc["category_mix"]:
        idx = eol == k
        mix[idx] = rng.dirichlet(np.clip(np.array(pc["category_mix"][k]), 0.005, None) * 30, idx.sum())
    kwh_eff = P["kwh_per_t"].to_numpy().copy()
    legit_cat3 = (~fraud) & (eol == "recycling") & (rng.random(nR) < 0.03)   # genuine MLP recyclers (confounder)
    mix[legit_cat3] = rng.dirichlet([3, 3, 4, 0.1], legit_cat3.sum())
    kwh_eff[legit_cat3] *= 1.3

    # ------------------------------------------------------------------ claimed (reported) processing
    hollow_t = np.where(fraud[:, None], true_t / np.clip(1 - hollow_frac[:, None], 0.03, None) - true_t, 0.0) * active
    pure_rows = np.flatnonzero(pure_shell)
    if len(pure_rows):   # paper shells claim a plausible-sized plant, not 3% of one
        claim_level = cap_m[pure_rows, None] * rng.uniform(0.5, 1.2, (len(pure_rows), 1)) * rng.lognormal(0, 0.1, (len(pure_rows), M))
        hollow_t[pure_rows] = np.maximum(claim_level * active[pure_rows] - true_t[pure_rows], 0)
    burst = P["deadline_burst"].to_numpy()
    fy_idx = np.repeat(np.arange(M // 12), 12)[:M]
    burst_w = np.where(np.isin(months.month, [2, 3]), 6.0, np.where(np.isin(months.month, [1]), 2.0, 0.35))
    for i in np.flatnonzero(burst):
        for f in range(fy_idx.max() + 1):
            mm = (fy_idx == f) & active[i]
            if mm.sum() == 0:
                continue
            tot = hollow_t[i, mm].sum()
            w = burst_w[mm] * rng.uniform(0.8, 1.2, mm.sum())
            hollow_t[i, mm] = tot * w / w.sum()
    hollow_t = hollow_t.round(3)
    cat3_inf = P["cat3_inflation"].to_numpy()
    hollow_mix = mix.copy()
    hollow_mix[cat3_inf] = rng.dirichlet([1.5, 1.0, 7.0, 0.1], cat3_inf.sum())
    true_tc = true_t[:, :, None] * mix[:, None, :]
    hollow_tc = hollow_t[:, :, None] * hollow_mix[:, None, :]
    claimed_tc = (true_tc + hollow_tc).round(3)

    # ------------------------------------------------------------------ collectors
    ctypes = list(pc["collector_types"])
    nC = int(nR * pc["collector_ratio"])
    c_cl = sample_clusters(rng, nC)
    c_type = rng.choice(ctypes, nC, p=np.array(list(pc["collector_types"].values())))
    c_lat, c_lon = jitter_latlon(rng, [CLUSTERS[c].lat for c in c_cl], [CLUSTERS[c].lon for c in c_cl], km=12)
    c_names, c_cons = company_names(rng, nC, "collector")
    c_size = rng.pareto(1.6, nC) + 1
    c_vehicles = [vehicle_regs(rng, [CLUSTERS[c].state] * int(rng.integers(1, 7))) for c in c_cl]
    c_state = STATE_OF_CLUSTER[c_cl]
    state_pool = {s: np.flatnonzero(c_state == s) for s in np.unique(c_state)}

    r_state = E["state"].to_numpy()
    r_lat, r_lon = E["lat"].to_numpy(), E["lon"].to_numpy()
    truck = pc["truckload_t"]
    t_lo = np.array([truck[t][0] for t in ctypes]); t_hi = np.array([truck[t][1] for t in ctypes])
    ctype_idx = np.array([ctypes.index(t) for t in c_type])

    inv_parts = []          # dict-of-arrays chunks
    legit_supply_cm = np.zeros((nC, M))
    # supplier sets
    suppliers = []
    for i in range(nR):
        pool = state_pool.get(r_state[i], np.arange(nC))
        k = int(np.clip(round(4 + np.log1p(cap_m[i] * 12) * 1.6 + rng.normal(0, 2)), 3, 25))
        local = rng.random(k) < 0.8
        w = c_size[pool] / c_size[pool].sum()
        a = rng.choice(pool, min(k, len(pool)), replace=False, p=w)
        b = rng.choice(nC, k, replace=False)
        s = np.unique(np.where(local[:len(a)], a, b[:len(a)]))
        suppliers.append(s)

    def emit_invoices(i, sup_idx, ton_ms, kinds, cat_mix_row, ghost_style=None, seller_kind="collector",
                      seller_ids=None, vehicle_pool=None):
        """Turn a [M, S] tonnage matrix into individual invoices for buyer recycler i."""
        Mi, S = ton_ms.shape
        flat = ton_ms.ravel()
        nz = np.flatnonzero(flat > 0.05)
        if len(nz) == 0:
            return
        m_of = nz // S
        s_of = nz % S
        if seller_kind == "collector":
            sel = sup_idx[s_of]
            ct = ctype_idx[np.minimum(sel, nC - 1)]
            mean_truck = np.where(sel < nC, (t_lo[ct] + t_hi[ct]) / 2, rng.uniform(6, 14, len(sel)))   # ghosts: big paper loads
        else:
            mean_truck = np.full(len(nz), 12.0)
        counts = np.maximum(1, np.round(flat[nz] / mean_truck * rng.uniform(0.8, 1.2, len(nz)))).astype(int)
        rep = np.repeat(np.arange(len(nz)), counts)
        g = rng.gamma(5.0, 1.0, len(rep))
        gs = np.bincount(rep, g, minlength=len(nz))
        ton = flat[nz][rep] * g / gs[rep]
        mo = m_of[rep]
        days_in = months[mo].days_in_month.to_numpy()
        day = (rng.random(len(rep)) * days_in).astype(int)
        date = months[mo].to_numpy() + day.astype("timedelta64[D]")
        wd = pd.DatetimeIndex(date).dayofweek.to_numpy()
        if ghost_style in (None, "subtle"):
            shift = (wd == 6) & (rng.random(len(rep)) < 0.9)          # legit: very few Sunday deliveries
            date = np.where(shift, date - np.timedelta64(1, "D"), date)
            ton = np.round(ton, 2)
        elif ghost_style == "obvious":
            month_end = rng.random(len(rep)) < 0.35                   # bulk paper invoices on month end
            date = np.where(month_end, (months[mo] + pd.offsets.MonthEnd(0)).to_numpy(), date)
            ton = np.round(ton)                                       # round-number tonnages
        else:
            ton = np.where(rng.random(len(rep)) < 0.5, np.round(ton), np.round(ton, 2))
        ton = np.maximum(ton, 0.2)
        cat = rng.choice(4, len(rep), p=cat_mix_row / cat_mix_row.sum())
        sid = (seller_ids if seller_ids is not None else sup_idx)[s_of[rep]]
        inv_parts.append({"buyer": np.full(len(rep), i), "seller_idx": sid, "seller_kind": np.full(len(rep), seller_kind, dtype=object),
                          "date": date, "tonnes": ton, "cat": cat, "kind": np.full(len(rep), kinds, dtype=object),
                          "ghost_style": np.full(len(rep), ghost_style or "", dtype=object)})

    # legit backing for TRUE tonnes
    for i in range(nR):
        s = suppliers[i]
        base = rng.dirichlet(np.ones(len(s)) * 0.8)
        shares = base[None, :] * rng.lognormal(0, 0.35, (M, len(s)))
        shares /= shares.sum(1, keepdims=True)
        inp = true_t[i] / rng.uniform(0.97, 1.0, M)
        ton = inp[:, None] * shares
        legit_supply_cm[s] += ton.T
        emit_invoices(i, s, ton, "legit", mix[i])

    # ------------------------------------------------------------------ fraud backing for HOLLOW tonnes
    ghost_rows = []         # new collector rows
    ghost_of = []           # (ghost_global_idx, recycler i)
    E_ids = E["entity_id"].to_numpy()
    labels = reg["labels"].set_index("entity_id")
    op = labels.loc[E_ids, "operator_id"].to_numpy()
    diff = P["difficulty"].to_numpy()
    ghost_b = P["ghost_feedstock"].to_numpy()
    rt_b = P["round_trip"].to_numpy()

    # round-trip rings among plastic fraud entities (prefer same operator)
    rt_idx = list(np.flatnonzero(rt_b))
    rt_idx.sort(key=lambda j: (str(op[j]), rng.random()))
    rings = []
    while len(rt_idx) >= 3:
        k = min(len(rt_idx), int(rng.integers(3, 6)))
        rings.append([rt_idx.pop(0) for _ in range(k)])
    if rt_idx and rings:
        rings[-1].extend(rt_idx)
    in_ring = {j: r for r, ring in enumerate(rings) for j in ring}
    ring_rows = []

    next_ghost = nC
    for i in np.flatnonzero(fraud):
        H = hollow_t[i]
        if H.sum() <= 0:
            continue
        d = diff[i]
        backing = H.copy()
        if i in in_ring:   # part of i's hollow input "justified" by material invoiced to it by the previous ring member
            ring = rings[in_ring[i]]
            prv = ring[(ring.index(i) - 1) % len(ring)]
            q = H * rng.uniform(0.3, 0.6)
            ton = (q / rng.uniform(0.97, 1.0, M))[:, None]
            emit_invoices(i, np.array([prv]), ton, "round_trip", np.array([0.9, 0.1, 0, 0]),
                          ghost_style="moderate", seller_kind="recycler", seller_ids=np.array([prv]))
            ring_rows.append((E_ids[prv], E_ids[i], f"RING{in_ring[i]}"))
            backing = H - q
        if ghost_b[i]:
            g = int(rng.integers(2, 9))
            gidx = np.arange(next_ghost, next_ghost + g)
            next_ghost += g
            for gi in gidx:
                ghost_rows.append((gi, i))
                ghost_of.append((gi, i))
            shares = rng.dirichlet(np.ones(g) * 1.5)
            ton = (backing / rng.uniform(0.97, 1.0, M))[:, None] * shares[None, :]    # same buffer as legit purchases
            emit_invoices(i, gidx, ton, "ghost", hollow_mix[i], ghost_style=d, seller_ids=gidx)
        elif d != "obvious":
            # inflate invoices of real suppliers (collector supplies beyond its real capacity)
            s = suppliers[i]
            shares = rng.dirichlet(np.ones(len(s)))
            ton = (backing / rng.uniform(0.97, 1.0, M))[:, None] * shares[None, :]
            emit_invoices(i, s, ton, "inflated_real", hollow_mix[i], ghost_style="subtle")
        # obvious & no ghosts: hollow tonnes simply have NO feedstock invoices (lazy paper claim)

    # ------------------------------------------------------------------ build collector table (real + ghost)
    nG = next_ghost - nC
    allC = nC + nG
    col_ids = make_ids(rng, "COL", allC)
    col = pd.DataFrame({"collector_id": col_ids})
    g_cl = np.zeros(nG, int)
    g_type = np.empty(nG, dtype=object)
    g_reg = []
    g_phone = phones(rng, nG)
    g_plot_from = np.full(nG, -1)
    E_cl_idx = np.array([next(k for k, c in enumerate(CLUSTERS) if c.estate == es) for es in E["estate"]])
    first_inv = {}
    for part in inv_parts:
        if part["kind"][0] == "ghost":
            for s_, d_ in zip(part["seller_idx"], part["date"]):
                if s_ not in first_inv or d_ < first_inv[s_]:
                    first_inv[s_] = d_
    for j, (gi, i) in enumerate(ghost_rows):
        d = diff[i]
        g_cl[j] = E_cl_idx[i] if rng.random() < 0.8 else sample_clusters(rng, 1)[0]
        g_type[j] = rng.choice(["aggregator", "trader", "kabadiwala_coop"], p=[0.5, 0.35, 0.15])
        fi = pd.Timestamp(first_inv.get(gi, months[0]))
        g_reg.append(fi - pd.Timedelta(days=int(rng.uniform(5, 180))))
        share_p = {"obvious": 0.40, "moderate": 0.15, "subtle": 0.05}[d]
        if rng.random() < share_p:
            g_phone[j] = E["contact_phone"].iat[i]
        if rng.random() < share_p:
            g_plot_from[j] = i
    # ghosts of the same recycler also share phones among themselves sometimes
    for i in set(i for _, i in ghost_rows):
        js = [j for j, (_, ii) in enumerate(ghost_rows) if ii == i]
        if len(js) > 1 and rng.random() < 0.4:
            for j in js[1:]:
                if rng.random() < 0.5:
                    g_phone[j] = g_phone[js[0]]

    all_cl = np.concatenate([c_cl, g_cl])
    all_type = np.concatenate([c_type, g_type]).astype(object)
    all_names = np.concatenate([c_names, company_names(rng, nG, "collector")[0]])
    all_cons = np.concatenate([c_cons, rng.choice(["Proprietorship", "Partnership"], nG)])
    pans = pan_numbers(rng, allC, all_cons, all_names)
    col["name"] = all_names
    col["collector_type"] = all_type
    col["state"] = [CLUSTERS[c].state for c in all_cl]
    col["district"] = [CLUSTERS[c].district for c in all_cl]
    other_state = rng.random(allC) < 0.03
    gst_state = np.where(other_state, [CLUSTERS[c].gst_code for c in sample_clusters(rng, allC)], [CLUSTERS[c].gst_code for c in all_cl])
    col["gstin"] = gstins(gst_state, pans)
    no_gst = (all_type == "shg") & (rng.random(allC) < 0.6)
    col.loc[no_gst, "gstin"] = None
    addr = addresses(rng, all_cl)
    for j in range(nG):
        if g_plot_from[j] >= 0:
            addr[nC + j] = E["address"].iat[g_plot_from[j]]
    col["address"] = addr
    gl, gn = jitter_latlon(rng, [CLUSTERS[c].lat for c in g_cl], [CLUSTERS[c].lon for c in g_cl], km=6)
    col["lat"] = np.concatenate([c_lat, gl]).round(6)
    col["lon"] = np.concatenate([c_lon, gn]).round(6)
    col["phone"] = np.concatenate([phones(rng, nC), g_phone])
    legit_reg = [pd.Timestamp("2015-01-01") + pd.Timedelta(days=int(x)) for x in rng.uniform(0, 3200, nC)]
    col["registration_date"] = pd.to_datetime(pd.Series(legit_reg + g_reg)).dt.date
    monthly_max = legit_supply_cm.max(axis=1)
    decl = np.concatenate([
        np.maximum(monthly_max * rng.uniform(1.1, 2.0, nC), rng.lognormal(np.log(pc["collector_monthly_capacity_t_lognormal"]["median"]),
                                                                          pc["collector_monthly_capacity_t_lognormal"]["sigma"], nC) * 0.3),
        rng.uniform(5, 40, nG)])
    col["declared_monthly_capacity_t"] = np.round(decl, 0)
    col["vehicle_count"] = np.concatenate([[len(v) for v in c_vehicles], rng.integers(1, 3, nG)])
    col_labels = pd.DataFrame({"collector_id": col_ids, "is_ghost": np.arange(allC) >= nC,
                               "ghost_for_entity": [None] * nC + [E_ids[i] for _, i in ghost_rows],
                               "operator_id": [None] * nC + [op[i] for _, i in ghost_rows]})

    # ------------------------------------------------------------------ invoices table
    inv = {k: np.concatenate([p[k] for p in inv_parts]) for k in inv_parts[0]}
    nI = len(inv["tonnes"])
    seller_is_rec = inv["seller_kind"] == "recycler"
    seller_ids = np.where(seller_is_rec, E_ids[np.where(seller_is_rec, inv["seller_idx"], 0)],
                          col_ids[np.where(~seller_is_rec, inv["seller_idx"], 0)])
    s_state = np.where(seller_is_rec, r_state[np.where(seller_is_rec, inv["seller_idx"], 0)],
                       col["state"].to_numpy()[np.where(~seller_is_rec, inv["seller_idx"], 0)])
    s_lat = np.where(seller_is_rec, r_lat[np.where(seller_is_rec, inv["seller_idx"], 0)], col["lat"].to_numpy()[np.where(~seller_is_rec, inv["seller_idx"], 0)])
    s_lon = np.where(seller_is_rec, r_lon[np.where(seller_is_rec, inv["seller_idx"], 0)], col["lon"].to_numpy()[np.where(~seller_is_rec, inv["seller_idx"], 0)])
    b = inv["buyer"]
    dist = haversine_km(s_lat, s_lon, r_lat[b], r_lon[b]) * rng.uniform(1.15, 1.45, nI)   # road factor
    cat = inv["cat"]
    price_lo = np.array([pc["feedstock_price_inr_per_t"][c][0] for c in CATS])
    price_hi = np.array([pc["feedstock_price_inr_per_t"][c][1] for c in CATS])
    rate = rng.uniform(price_lo[cat], price_hi[cat])
    rate = np.where(inv["kind"] == "round_trip", rng.uniform(42000, 62000, nI), rate)   # "recycled granules"
    rate = np.round(rate / 50) * 50
    value = np.round(inv["tonnes"] * rate, 2)
    # vehicles: legit collectors use their own small fleet; obvious ghosts reuse 1 truck or none
    veh = np.empty(nI, dtype=object)
    for s_ in np.unique(inv["seller_idx"][~seller_is_rec]):
        m = (~seller_is_rec) & (inv["seller_idx"] == s_)
        pool = c_vehicles[s_] if s_ < nC else vehicle_regs(rng, [col["state"].iat[s_]] * int(rng.integers(1, 3)))
        veh[m] = rng.choice(pool, m.sum())
    for s_ in np.unique(inv["seller_idx"][seller_is_rec]):
        m = seller_is_rec & (inv["seller_idx"] == s_)
        veh[m] = rng.choice(vehicle_regs(rng, [r_state[s_]] * 2), m.sum())
    gs = inv["ghost_style"]
    obvious_g = gs == "obvious"
    reuse_truck = obvious_g & (rng.random(nI) < 0.6)
    if reuse_truck.any():
        veh[reuse_truck] = vehicle_regs(rng, ["Delhi"])[0]
    eway = np.array([str(x) for x in rng.integers(10**11, 10**12 - 1, nI)], dtype=object)
    need_eway = value > 50000
    missing = (~need_eway) | (obvious_g & (rng.random(nI) < 0.7)) | ((gs == "moderate") & (rng.random(nI) < 0.25)) | \
              (need_eway & (rng.random(nI) < 0.015))
    eway[missing] = None
    order = np.argsort(inv["date"], kind="stable")
    invoices = pd.DataFrame({
        "invoice_id": make_ids(rng, "INV", nI, 9),
        "invoice_date": pd.to_datetime(inv["date"]).date,
        "seller_id": seller_ids, "seller_type": inv["seller_kind"],
        "buyer_id": E_ids[b], "category": np.array(CATS)[cat],
        "tonnes": np.round(inv["tonnes"], 2), "rate_inr_per_t": rate, "taxable_value_inr": value,
        "gst_inr": np.round(value * GST_RATE, 2), "vehicle_reg": veh, "eway_bill_no": eway,
        "distance_km": np.round(dist, 1), "seller_state": s_state,
    }).iloc[order].reset_index(drop=True)
    inv_labels = pd.DataFrame({"invoice_id": invoices["invoice_id"], "kind": inv["kind"][order]})

    # ------------------------------------------------------------------ reported monthly processing (returns)
    rows = []
    for c in range(4):
        rows.append(pd.DataFrame({
            "entity_id": np.repeat(E_ids, M), "month": np.tile(months.date, nR), "category": CATS[c],
            "processed_t_reported": claimed_tc[:, :, c].ravel(), "_true": true_tc[:, :, c].ravel(), "_hollow": hollow_tc[:, :, c].ravel(),
            "_active": active.ravel()}))
    mp = pd.concat(rows, ignore_index=True)
    mp = mp[mp["_active"] & (mp["processed_t_reported"] > 0.0005)].reset_index(drop=True)
    mp["eol_type"] = pd.Series(eol, index=E_ids).loc[mp["entity_id"]].to_numpy()
    monthly_truth = mp[["entity_id", "month", "category", "_true", "_hollow"]].rename(columns={"_true": "true_t", "_hollow": "hollow_t"})
    monthly = mp[["entity_id", "month", "category", "eol_type", "processed_t_reported"]].copy()
    monthly["processed_t_reported"] = monthly["processed_t_reported"].round(3)

    # ------------------------------------------------------------------ certificate lots
    lot = monthly.copy()
    lot = lot[lot["processed_t_reported"] >= 0.01].reset_index(drop=True)
    lag = rng.integers(5, 41, len(lot))
    is_burst = pd.Series(burst, index=E_ids).loc[lot["entity_id"]].to_numpy()
    gen = pd.to_datetime(lot["month"]) + pd.offsets.MonthEnd(0) + pd.to_timedelta(lag, "D")
    # burst fraudsters batch-generate late, just before the 30 June annual-return deadline
    m_ = pd.to_datetime(lot["month"])
    fy_end_year = np.where(m_.dt.month >= 4, m_.dt.year + 1, m_.dt.year)
    late = is_burst & (rng.random(len(lot)) < 0.6)
    late_date = pd.to_datetime(pd.Series(fy_end_year).astype(str) + "-06-" + pd.Series(rng.integers(5, 29, len(lot))).astype(str).str.zfill(2))
    gen = pd.Series(np.where(late, np.maximum(late_date.to_numpy(), gen.to_numpy()), gen.to_numpy()))
    lot.insert(0, "lot_id", make_ids(rng, "LOT", len(lot), 9))
    lot["generated_on"] = pd.to_datetime(gen).dt.date
    lot["fy"] = ["FY" + str(y - 1) + "-" + str(y % 100).zfill(2) for y in fy_end_year]
    lot = lot.rename(columns={"processed_t_reported": "tonnes", "entity_id": "recycler_id", "month": "period_month"})
    mt = monthly_truth[monthly_truth["true_t"] + monthly_truth["hollow_t"] > 0]
    key = mt.set_index(["entity_id", "month", "category"])
    hol = key.loc[list(zip(lot["recycler_id"], lot["period_month"], lot["category"])), "hollow_t"].to_numpy()
    lot_labels = pd.DataFrame({"lot_id": lot["lot_id"], "hollow_tonnes": np.round(np.minimum(hol, lot["tonnes"]), 3)})
    lot_labels["hollow_share"] = np.round(lot_labels["hollow_tonnes"] / lot["tonnes"].to_numpy(), 4)

    # ------------------------------------------------------------------ certificate prices (per seller, category, FY)
    floor = np.array([pc["cert_cost_floor_inr_per_t"][c] for c in CATS], float)
    decl_cap = E["declared_capacity_tpa"].to_numpy()
    efficient = (~fraud) & (decl_cap >= np.quantile(decl_cap, 0.85)) & (rng.random(nR) < 0.6)
    markup = rng.uniform(*pc["legit_markup"], nR)
    markup[efficient] = rng.uniform(*pc["efficient_legit_markup"], efficient.sum())
    under = P["underprice"].to_numpy()
    for i in np.flatnonzero(fraud):
        if under[i]:
            markup[i] = rng.uniform(*pc["hollow_markup"][diff[i]])
        else:
            markup[i] = rng.uniform(0.95, 1.3)
    fy_trend = {"FY2023-24": 1.0, "FY2024-25": 0.86, "FY2025-26": 0.76}

    # ------------------------------------------------------------------ brands & obligations
    nB = int(nR * pc["brands_per_recycler"])
    sectors = list(SECTOR_CAT_MIX)
    b_sector = rng.choice(sectors, nB)
    b_cl = sample_clusters(rng, nB)
    b_names, b_cons = company_names(rng, nB, "brand")
    b_size = rng.lognormal(0, 1.6, nB)
    b_type = rng.choice(["brand_owner", "producer", "importer"], nB, p=[0.5, 0.35, 0.15])
    style = rng.choice(["price_sensitive", "balanced", "quality"], nB, p=[0.30, 0.50, 0.20])
    b_mix = np.vstack([rng.dirichlet(np.clip(np.array(SECTOR_CAT_MIX[s]), 0.01, None) * 25) for s in b_sector])
    b_ids = make_ids(rng, "PIBO", nB)
    b_pan = pan_numbers(rng, nB, b_cons, b_names)
    brands = pd.DataFrame({"brand_id": b_ids, "name": b_names, "pibo_type": b_type, "sector": b_sector,
                           "state": [CLUSTERS[c].state for c in b_cl],
                           "gstin": gstins([CLUSTERS[c].gst_code for c in b_cl], b_pan),
                           "registration_date": [(pd.Timestamp("2022-04-01") + pd.Timedelta(days=int(x))).date()
                                                 for x in rng.exponential(220, nB).clip(0, 1300)]})

    fys = sorted(lot["fy"].unique())
    rec_age_y = (pd.Timestamp("2026-03-31") - pd.to_datetime(E["epr_registration_date"])).dt.days.to_numpy() / 365
    size_w = np.log1p(decl_cap)
    sales_rows = []
    obligations = []
    lot_by = {k: g.sort_values("generated_on") for k, g in lot.groupby(["fy", "category"])}
    for fy in fys:
        for c, cname in enumerate(CATS):
            g = lot_by.get((fy, cname))
            if g is None:
                continue
            supply = g.groupby("recycler_id")["tonnes"].sum()
            rec_pos = pd.Series(np.arange(nR), index=E_ids).loc[supply.index].to_numpy()
            remaining = supply.to_numpy().copy()
            price = floor[c] * markup[rec_pos] * fy_trend.get(fy, 0.75)
            genuine = mt[(mt["category"] == cname)]
            genuine = genuine[pd.Series([("FY" + str((m.year if m.month >= 4 else m.year - 1)) + "-" +
                                          str(((m.year if m.month >= 4 else m.year - 1) + 1) % 100).zfill(2)) == fy
                                         for m in pd.to_datetime(genuine["month"])], index=genuine.index)]
            total_demand = genuine["true_t"].sum() * pc["demand_to_legit_supply"]
            share = b_size * b_mix[:, c]
            demand = total_demand * share / share.sum()
            demand = np.where(demand < 0.05, 0, np.round(demand, 3))
            obligations.append(pd.DataFrame({"brand_id": b_ids, "fy": fy, "category": cname, "obligation_t": demand}))
            w_price = {"price_sensitive": (1 / price) ** 4, "balanced": (1 / price) ** 1.5 * size_w[rec_pos] ** 0.3,
                       "quality": (1 / price) ** 0.5 * size_w[rec_pos] ** 0.8 * np.sqrt(np.clip(rec_age_y[rec_pos], 0.1, None))}
            order_b = rng.permutation(np.flatnonzero(demand > 0))
            for bi in order_b:
                need = demand[bi]
                k = int(np.clip(1 + np.log1p(need) * 0.9 + rng.normal(0, 1), 1, 10))
                w = w_price[style[bi]] * (remaining > 0.01)
                for _ in range(k):
                    tot = w.sum()
                    if tot <= 0 or need <= 0.01:
                        break
                    j = rng.choice(len(w), p=w / tot)
                    take = min(remaining[j], need if _ == k - 1 else need * rng.uniform(0.3, 1.0))
                    take = round(float(take), 3)
                    if take <= 0:
                        w[j] = 0
                        continue
                    remaining[j] -= take
                    need -= take
                    w[j] = 0 if remaining[j] <= 0.01 else w[j]
                    sales_rows.append((fy, cname, supply.index[j], bi, take, price[j] * rng.uniform(0.96, 1.04)))
    obligations = pd.concat(obligations, ignore_index=True)
    obligations = obligations[obligations["obligation_t"] > 0].reset_index(drop=True)

    # split seller-level sales across that seller's lots FIFO; sale dates cluster before the deadline
    sales = pd.DataFrame(sales_rows, columns=["fy", "category", "seller_id", "b_idx", "tonnes", "price"])
    sales["transaction_id"] = make_ids(rng, "TXN", len(sales), 9)
    lot_q = {k: list(zip(g["lot_id"], g["tonnes"].astype(float), pd.to_datetime(g["generated_on"])))
             for k, g in lot.sort_values("generated_on").groupby(["fy", "category", "recycler_id"])}
    lot_rem = {k: [x[1] for x in v] for k, v in lot_q.items()}
    lot_ptr = {k: 0 for k in lot_q}
    out_rows = []
    fy_end = {fy: pd.Timestamp(f"20{fy[7:9]}-03-31") for fy in fys}
    for r in sales.itertuples(index=False):
        k = (r.fy, r.category, r.seller_id)
        need = r.tonnes
        fe = fy_end[r.fy]
        if rng.random() < 0.6:
            sampled = fe + pd.Timedelta(days=int(rng.integers(1, 91)))           # Apr-Jun deadline rush
        else:
            sampled = fe - pd.Timedelta(days=int(rng.integers(0, 365)))
        while need > 1e-6 and lot_ptr[k] < len(lot_q[k]):
            p = lot_ptr[k]
            lid, _, gdate = lot_q[k][p]
            take = min(need, lot_rem[k][p])
            lot_rem[k][p] -= take
            need -= take
            if lot_rem[k][p] <= 1e-6:
                lot_ptr[k] += 1
            d = max(sampled, gdate + pd.Timedelta(days=int(rng.integers(0, 10))))
            out_rows.append((r.transaction_id, lid, r.seller_id, b_ids[r.b_idx], r.category, r.fy, d.date(), round(take, 3),
                             round(r.price, 0)))
    cert_sales = pd.DataFrame(out_rows, columns=["transaction_id", "lot_id", "seller_id", "buyer_id", "category", "fy", "sale_date",
                                                 "tonnes", "price_inr_per_t"])
    cert_sales.insert(0, "sale_id", make_ids(rng, "SALE", len(cert_sales), 9))
    cert_sales["amount_inr"] = np.round(cert_sales["tonnes"] * cert_sales["price_inr_per_t"], 2)

    # brand ground truth: exposure to hollow certificates
    hs = lot_labels.set_index("lot_id")["hollow_share"]
    cs = cert_sales.assign(hollow_t=cert_sales["tonnes"].to_numpy() * hs.loc[cert_sales["lot_id"]].to_numpy())
    agg = cs.groupby("buyer_id").agg(bought_t=("tonnes", "sum"), hollow_t=("hollow_t", "sum"))
    obl = obligations.groupby("brand_id")["obligation_t"].sum()
    brand_labels = pd.DataFrame({"brand_id": b_ids, "procurement_style": style})
    brand_labels["obligation_t"] = obl.reindex(b_ids).fillna(0).to_numpy().round(3)
    brand_labels["bought_t"] = agg["bought_t"].reindex(b_ids).fillna(0).to_numpy().round(3)
    brand_labels["hollow_t"] = agg["hollow_t"].reindex(b_ids).fillna(0).to_numpy().round(3)
    brand_labels["true_hollow_exposure"] = np.where(brand_labels["bought_t"] > 0, brand_labels["hollow_t"] / brand_labels["bought_t"], 0).round(4)
    brand_labels["compliance_ratio"] = np.where(brand_labels["obligation_t"] > 0, brand_labels["bought_t"] / brand_labels["obligation_t"], np.nan).round(4)

    # ------------------------------------------------------------------ recycled output sales (offtakers)
    nO = int(nR * pc["offtakers_per_recycler"])
    o_cl = sample_clusters(rng, nO)
    o_names, o_cons = company_names(rng, nO, "offtaker")
    o_pan = pan_numbers(rng, nO, o_cons, o_names)
    o_state = np.array([CLUSTERS[c].state for c in o_cl])
    o_pool = {s: np.flatnonzero(o_state == s) for s in np.unique(o_state)}
    yields = np.array([rng.uniform(*pc["eol_types"][e]["yield"]) for e in eol])
    product = np.where(eol == "waste_to_oil", "pyrolysis_oil", np.where(rng.random(nR) < 0.7, "granules", "flakes"))
    shell_off = {}          # operator -> offtaker indices
    out_inv = []
    o_extra = []
    out_fab = P["output_fabrication"].to_numpy()
    for i in range(nR):
        if yields[i] <= 0:
            continue
        pool = o_pool.get(r_state[i], np.arange(nO))
        buyers = rng.choice(pool, min(len(pool), int(rng.integers(2, 11))), replace=False)
        sh = rng.dirichlet(np.ones(len(buyers)))
        outq = true_t[i] * yields[i]
        if fraud[i] and out_fab[i]:
            key_ = op[i]
            if key_ not in shell_off:
                k_ = int(rng.integers(1, 4))
                shell_off[key_] = list(range(nO + len(o_extra), nO + len(o_extra) + k_))
                o_extra.extend([i] * k_)
            fab = hollow_t[i] * yields[i]
            fb = np.array(shell_off[key_])
            fsh = rng.dirichlet(np.ones(len(fb)))
            for m in np.flatnonzero(fab > 0.05):
                for bb, ss in zip(fb, fsh):
                    out_inv.append((i, bb, m, fab[m] * ss, product[i], "fabricated"))
        for m in np.flatnonzero(outq > 0.05):
            n_inv = int(rng.integers(1, 4))
            for bb, ss in zip(buyers, sh):
                q = outq[m] * ss
                if q < 0.05:
                    continue
                for part in rng.dirichlet(np.ones(n_inv)):
                    out_inv.append((i, bb, m, q * part, product[i], "legit"))
    nOx = len(o_extra)
    o_all = nO + nOx
    x_cl = np.array([E_cl_idx[i] for i in o_extra], int) if nOx else np.zeros(0, int)
    off_names = np.concatenate([o_names, company_names(rng, nOx, "offtaker")[0]])
    off_cons = np.concatenate([o_cons, rng.choice(["Proprietorship", "Private Limited"], nOx)])
    off_pan = np.concatenate([o_pan, pan_numbers(rng, nOx, off_cons[nO:], off_names[nO:])])
    off_cl = np.concatenate([o_cl, x_cl]).astype(int)
    off_ids = make_ids(rng, "OFT", o_all)
    off_inc = [(pd.Timestamp("2008-01-01") + pd.Timedelta(days=int(x))).date() for x in rng.uniform(0, 5500, nO)] + \
              [(pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(x))).date() for x in rng.uniform(0, 900, nOx)]
    offtakers = pd.DataFrame({"offtaker_id": off_ids, "name": off_names, "state": [CLUSTERS[c].state for c in off_cl],
                              "gstin": gstins([CLUSTERS[c].gst_code for c in off_cl], off_pan),
                              "address": addresses(rng, off_cl), "incorporation_date": off_inc})
    off_labels = pd.DataFrame({"offtaker_id": off_ids, "is_shell": np.arange(o_all) >= nO,
                               "operator_id": [None] * nO + [op[i] for i in o_extra]})
    oi = pd.DataFrame(out_inv, columns=["r", "o", "m", "tonnes", "product", "kind"])
    day = (rng.random(len(oi)) * months[oi["m"]].days_in_month.to_numpy()).astype(int)
    odate = months[oi["m"]].to_numpy() + day.astype("timedelta64[D]")
    orate = np.where(oi["product"] == "pyrolysis_oil", rng.uniform(32000, 45000, len(oi)), rng.uniform(48000, 85000, len(oi)))
    sort_idx = np.argsort(pd.to_datetime(odate).to_numpy(), kind="stable")
    output_invoices = pd.DataFrame({
        "invoice_id": make_ids(rng, "OUT", len(oi), 9),
        "invoice_date": pd.to_datetime(odate[sort_idx]).date,
        "seller_id": E_ids[oi["r"].to_numpy()[sort_idx]], "buyer_id": off_ids[oi["o"].to_numpy()[sort_idx]],
        "product": oi["product"].to_numpy()[sort_idx], "tonnes": oi["tonnes"].to_numpy()[sort_idx].round(2),
        "rate_inr_per_t": (np.round(orate / 50) * 50)[sort_idx]})
    output_invoices["taxable_value_inr"] = (output_invoices["tonnes"] * output_invoices["rate_inr_per_t"]).round(2)
    out_labels = pd.DataFrame({"invoice_id": output_invoices["invoice_id"], "kind": oi["kind"].to_numpy()[sort_idx]})

    # ------------------------------------------------------------------ electricity bills (shared DISCOM model)
    tariff = reg["tariff"]
    bills = make_bills(rng, E_ids, r_state, E["connected_load_kw"].to_numpy(), kwh_eff, P["solar_fraction"].to_numpy(),
                       fraud, diff, P["doctored_bill"].to_numpy(), true_t, true_t + hollow_t, active, months, tariff,
                       lambda k: make_ids(rng, "BILL", k, 8))

    # ------------------------------------------------------------------ labels & write
    rec_labels = pd.DataFrame({"entity_id": E_ids, "legit_cat3_recycler": legit_cat3, "efficient_low_price_legit": efficient,
                               "certificate_markup": markup.round(3), "round_trip_ring": [f"RING{in_ring[i]}" if i in in_ring else None for i in range(nR)]})
    d = out / "plastic"
    L = out / "labels" / "plastic"
    write_table(col, d / "collectors.parquet", manifest)
    write_table(invoices, d / "feedstock_invoices.parquet", manifest)
    write_table(monthly, d / "monthly_processing_reported.parquet", manifest)
    write_table(lot, d / "certificate_lots.parquet", manifest)
    write_table(cert_sales, d / "certificate_sales.parquet", manifest)
    write_table(brands, d / "brands.parquet", manifest)
    write_table(obligations, d / "brand_obligations.parquet", manifest)
    write_table(offtakers, d / "offtakers.parquet", manifest)
    write_table(output_invoices, d / "output_invoices.parquet", manifest)
    write_table(bills, d / "electricity_bills.parquet", manifest)
    write_table(col_labels, L / "collector_labels.parquet", manifest)
    write_table(inv_labels, L / "invoice_labels.parquet", manifest)
    write_table(out_labels, L / "output_invoice_labels.parquet", manifest)
    write_table(off_labels, L / "offtaker_labels.parquet", manifest)
    write_table(lot_labels, L / "lot_labels.parquet", manifest)
    write_table(brand_labels, L / "brand_labels.parquet", manifest)
    write_table(monthly_truth, L / "monthly_truth.parquet", manifest)
    write_table(rec_labels, L / "recycler_extra_labels.parquet", manifest)
    write_table(pd.DataFrame(ring_rows, columns=["seller_id", "buyer_id", "ring_id"]), L / "round_trip_edges.parquet", manifest)
    return {"monthly": monthly, "bills": bills, "output_invoices": output_invoices, "invoices": invoices}
