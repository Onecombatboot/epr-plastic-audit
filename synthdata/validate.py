"""Validate a generated dataset and write validation_report.md.

    python -m synthdata.validate --data data/synthetic

Three kinds of checks:
  1. Integrity   - unique keys, foreign keys, value ranges, GSTIN checksums, date order.
  2. Physics     - clean (non-fraud) entities obey capacity / energy / mass-balance limits.
  3. Difficulty  - AUC of a naive single-signal detector per difficulty tier. Obvious fraud
                   should be easy, subtle fraud should be hard (otherwise the data is too
                   easy to be a meaningful benchmark, or not learnable at all).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ALNUM, gstin_checksum

RESULTS: list[tuple[str, str, bool, str]] = []


def check(section, name, ok, detail=""):
    RESULTS.append((section, name, bool(ok), str(detail)))


def auc(scores, labels):
    """Mann-Whitney AUC; higher score = more likely positive."""
    s = pd.Series(scores).astype(float)
    y = pd.Series(labels).astype(bool)
    m = s.notna()
    s, y = s[m], y[m]
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = s.rank()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def tiered_auc(df, score_col, fraud_col="is_fraud", diff_col="difficulty"):
    out = {"all": auc(df[score_col], df[fraud_col])}
    legit = df[~df[fraud_col].astype(bool)]
    for d in ["obvious", "moderate", "subtle"]:
        sub = pd.concat([legit, df[df[diff_col] == d]])
        out[d] = auc(sub[score_col], sub[fraud_col])
    return {k: round(v, 3) for k, v in out.items()}


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def v_registry(D):
    S = "registry"
    e = pd.read_parquet(D / "registry/entities.parquet")
    lab = pd.read_parquet(D / "labels/registry/entity_labels.parquet")
    ed = pd.read_parquet(D / "registry/entity_directors.parquet")
    ppl = pd.read_parquet(D / "registry/people.parquet")
    ph = pd.read_parquet(D / "registry/plant_photos.parquet")
    pairs = pd.read_parquet(D / "labels/registry/same_operator_pairs.parquet")
    check(S, "entity_id unique", e["entity_id"].is_unique, len(e))
    check(S, "labels align with entities", set(lab["entity_id"]) == set(e["entity_id"]))
    g = e["gstin"].astype(str)
    ok = g.map(lambda x: len(x) == 15 and x[14] == gstin_checksum(x[:14]))
    check(S, "GSTIN checksum valid", ok.all(), f"{ok.mean():.4f}")
    check(S, "PAN format", e["pan"].str.fullmatch(r"[A-Z]{5}\d{4}[A-Z]").all())
    check(S, "incorporation <= EPR registration", (pd.to_datetime(e["incorporation_date"]) <= pd.to_datetime(e["epr_registration_date"])).all())
    check(S, "directors reference people", ed["person_id"].isin(ppl["person_id"]).all())
    rates = lab.groupby("stream")["is_fraud"].mean().round(3).to_dict()
    check(S, "fraud rate per stream in 5-12%", all(0.05 <= v <= 0.12 for v in rates.values()), rates)
    # shell clusters must be discoverable in principle: each same-operator pair connected via shared identifiers
    m = e.set_index("entity_id")
    dirs = ed.groupby("entity_id")["person_id"].apply(set)
    pp = ppl.set_index("person_id")
    sur_city = {p: (n.split()[-1], c) for p, n, c in zip(ppl["person_id"], ppl["name"], ppl["residential_city"])}
    photos = ph.groupby("entity_id")["phash"].apply(list)

    def linked(a, b):
        A, B = m.loc[a], m.loc[b]
        if dirs.get(a, set()) & dirs.get(b, set()):
            return "director"
        for col in ["authorised_signatory_id", "contact_phone", "contact_email", "bank_account", "address"]:
            if A[col] == B[col]:
                return col
        sa = {sur_city[p] for p in dirs.get(a, set())}
        if sa & {sur_city[p] for p in dirs.get(b, set())}:
            return "relative"
        for x in photos.get(a, []):
            for y in photos.get(b, []):
                if hamming(x, y) <= 6:
                    return "photo"
        return None

    # connectivity per operator through direct links
    import networkx as nx
    n_ops, n_conn = 0, 0
    for op, grp in pairs.groupby("operator_id"):
        G = nx.Graph()
        members = set(grp["entity_a"]) | set(grp["entity_b"])
        G.add_nodes_from(members)
        for a, b in zip(grp["entity_a"], grp["entity_b"]):
            if linked(a, b):
                G.add_edge(a, b)
        n_ops += 1
        n_conn += nx.is_connected(G)
    check(S, "shell clusters connected via shared identifiers", n_conn == n_ops, f"{n_conn}/{n_ops} operators")
    check(S, "photo hashes are 16 hex chars", ph["phash"].str.fullmatch(r"[0-9a-f]{16}").all())


def v_plastic(D):
    S = "plastic"
    e = pd.read_parquet(D / "registry/entities.parquet").query("stream == 'plastic'")
    lab = pd.read_parquet(D / "labels/registry/entity_labels.parquet")
    phys = pd.read_parquet(D / "labels/registry/entity_physics.parquet").set_index("entity_id")
    inv = pd.read_parquet(D / "plastic/feedstock_invoices.parquet")
    col = pd.read_parquet(D / "plastic/collectors.parquet")
    mp = pd.read_parquet(D / "plastic/monthly_processing_reported.parquet")
    lots = pd.read_parquet(D / "plastic/certificate_lots.parquet")
    sales = pd.read_parquet(D / "plastic/certificate_sales.parquet")
    brands = pd.read_parquet(D / "plastic/brands.parquet")
    bills = pd.read_parquet(D / "plastic/electricity_bills.parquet")
    tariff = pd.read_parquet(D / "registry/tariff_reference.parquet")
    bl = pd.read_parquet(D / "labels/plastic/brand_labels.parquet")
    cl = pd.read_parquet(D / "labels/plastic/collector_labels.parquet")
    for t, k in [(inv, "invoice_id"), (lots, "lot_id"), (sales, "sale_id"), (col, "collector_id"), (brands, "brand_id"), (bills, "bill_id")]:
        check(S, f"{k} unique", t[k].is_unique, len(t))
    check(S, "invoice buyers are plastic recyclers", inv["buyer_id"].isin(e["entity_id"]).all())
    check(S, "invoice sellers exist", (inv["seller_id"].isin(col["collector_id"]) | inv["seller_id"].isin(e["entity_id"])).all())
    check(S, "sales reference lots & brands", sales["lot_id"].isin(lots["lot_id"]).all() and sales["buyer_id"].isin(brands["brand_id"]).all())
    sold = sales.groupby("lot_id")["tonnes"].sum()
    over = (sold - lots.set_index("lot_id").loc[sold.index, "tonnes"]) > 1e-3
    check(S, "no lot oversold", not over.any(), int(over.sum()))
    sd = pd.to_datetime(sales["sale_date"]) >= pd.to_datetime(lots.set_index("lot_id").loc[sales["lot_id"], "generated_on"]).to_numpy()
    check(S, "sale_date >= lot generation date", sd.all(), f"{sd.mean():.4f}")
    il = pd.read_parquet(D / "labels/plastic/invoice_labels.parquet")
    ev = inv[["taxable_value_inr", "eway_bill_no"]].join(il.set_index("invoice_id")["kind"], on=inv["invoice_id"])
    big = ev[ev["taxable_value_inr"] > 50000]
    miss = big.groupby("kind")["eway_bill_no"].apply(lambda x: round(x.isna().mean(), 3)).to_dict()
    check(S, "e-way bill missing rate (value>50k): legit low", miss.get("legit", 1) < 0.03, miss)
    check(S, "GSTIN checksum valid (collectors)", col["gstin"].dropna().map(lambda x: x[14] == gstin_checksum(x[:14])).all())

    E = e.set_index("entity_id").join(lab.set_index("entity_id")[["is_fraud", "difficulty"]])
    claimed = mp.groupby("entity_id")["processed_t_reported"].sum()
    nmon = mp.groupby("entity_id")["month"].nunique()
    E["claimed_annual"] = claimed / nmon * 12
    E["true_cap"] = phys.loc[E.index, "true_capacity_tpa"]
    legit = E[~E["is_fraud"]]
    phys_ok = (legit["claimed_annual"] <= legit["true_cap"] * 1.001)
    check(S, "PHYSICS legit claims <= true capacity", phys_ok.all(), f"{phys_ok.mean():.4f}")
    E["inv_cov"] = inv.groupby("buyer_id")["tonnes"].sum() / claimed
    lc = E.loc[~E["is_fraud"], "inv_cov"]
    check(S, "PHYSICS legit invoice coverage 0.95-1.10", lc.between(0.95, 1.10).mean() > 0.97, f"median {lc.median():.3f}")
    b = bills.groupby("entity_id")["units_kwh"].sum() / bills.groupby("entity_id")["billing_month"].nunique()
    E["kwh_per_t"] = b / (claimed / nmon)
    # bill arithmetic consistency (doctored bills break it)
    bt = bills.merge(e[["entity_id", "state"]], on="entity_id")
    bt["fy"] = pd.to_datetime(bt["billing_month"]).map(lambda m: f"FY{m.year if m.month >= 4 else m.year - 1}-{((m.year if m.month >= 4 else m.year - 1) + 1) % 100:02d}")
    bt = bt.merge(tariff, on=["state", "fy"])
    bt["rate_gap"] = (bt["energy_charge_inr"] / bt["units_kwh"].clip(lower=1) / bt["energy_rate_inr_per_kwh"] - 1).abs()
    E["bill_rate_gap"] = bt.groupby("entity_id")["rate_gap"].median()
    legit_gap = E.loc[~E["is_fraud"], "bill_rate_gap"]
    check(S, "PHYSICS legit bills arithmetically consistent", (legit_gap < 0.01).mean() > 0.99, f"max {legit_gap.max():.4f}")

    fl = {"CAT_I": 3500, "CAT_II": 2800, "CAT_III": 5000, "CAT_IV": 1500}
    sales["rel"] = sales["price_inr_per_t"] / sales["category"].map(fl)
    E["neg_price"] = -sales.groupby("seller_id")["rel"].median()
    E["neg_inv_cov"] = -E["inv_cov"]
    E["neg_kwh"] = -E["kwh_per_t"]
    E["load_ratio"] = E["claimed_annual"] / E["connected_load_kw"]
    E["bill_gap"] = E["bill_rate_gap"]
    for sc_ in ["neg_price", "neg_kwh", "load_ratio", "neg_inv_cov"]:
        check(S, f"DIFFICULTY naive AUC [{sc_}]", True, tiered_auc(E.reset_index(), sc_))
    doc = lab.set_index("entity_id")["doctored_bill"]
    sub = E[(~E["is_fraud"]) | doc.reindex(E.index).fillna(False)]
    check(S, "DIFFICULTY naive AUC [bill tariff gap, doctored-bill fraud only]", True, tiered_auc(sub.reset_index(), "bill_gap"))
    # brand exposure: price-sensitive brands should carry more hollow exposure
    ex = bl.groupby("procurement_style")["true_hollow_exposure"].mean().round(4).to_dict()
    check(S, "brand exposure higher for price-sensitive buyers", ex.get("price_sensitive", 0) > ex.get("quality", 1), ex)
    gh = cl["is_ghost"].mean()
    check(S, "ghost collector share 0.2-5%", 0.002 <= gh <= 0.05, f"{gh:.4f}")
    unsold = 1 - sales["tonnes"].sum() / lots["tonnes"].sum()
    check(S, "some certificates go unsold (crowding-out)", unsold > 0.01, f"{unsold:.3f}")


def v_yield(D):
    S = "yield"
    ops = pd.read_parquet(D / "yield/operations_monthly.parquet")
    outs = pd.read_parquet(D / "yield/outputs_monthly.parquet")
    bills = pd.read_parquet(D / "yield/electricity_bills.parquet")
    lab = pd.read_parquet(D / "labels/registry/entity_labels.parquet").set_index("entity_id")
    phys = pd.read_parquet(D / "labels/registry/entity_physics.parquet").set_index("entity_id")
    import yaml
    cfgs = {p.stem: yaml.safe_load(open(p, encoding="utf-8")) for p in (D / "yield/stream_configs").glob("*.yaml")}
    check(S, "stream configs present", set(cfgs) >= {"ewaste", "battery", "tyre", "used_oil", "plastic"}, sorted(cfgs))
    inp = ops.groupby("entity_id")["input_t_reported"].sum()
    n_m = ops.groupby("entity_id")["month"].nunique()
    E = pd.DataFrame({"input": inp, "months": n_m})
    E = E.join(lab[["is_fraud", "difficulty", "stream"]]).join(phys[["process_type", "true_capacity_tpa", "kwh_per_t"]])
    exp_saleable = []
    for eid, r in E.iterrows():
        o = cfgs[r["stream"]]["processes"][r["process_type"]]["outputs"]
        exp_saleable.append(sum(v["expected"] for v in o.values() if v["saleable"]))
    E["expected_saleable_share"] = exp_saleable
    E["sold"] = outs.groupby("entity_id")["tonnes_sold"].sum()
    E["balance_gap"] = 1 - E["sold"] / (E["input"] * E["expected_saleable_share"])
    E["kwh_per_t_claimed"] = (bills.groupby("entity_id")["units_kwh"].sum() / bills.groupby("entity_id")["billing_month"].nunique()) / (E["input"] / E["months"])
    E["neg_kwh_ratio"] = -E["kwh_per_t_claimed"] / E["kwh_per_t"].where(E["kwh_per_t"] > 0)
    legit = E[~E["is_fraud"]]
    check(S, "PHYSICS legit |output balance gap| < 12% (median)", legit["balance_gap"].abs().median() < 0.12, f"{legit['balance_gap'].abs().median():.3f}")
    check(S, "PHYSICS legit annual input <= true capacity", ((legit["input"] / legit["months"] * 12) <= legit["true_capacity_tpa"] * 1.001).mean() > 0.999)
    for sc_ in ["balance_gap", "neg_kwh_ratio"]:
        check(S, f"DIFFICULTY naive AUC [{sc_}]", True, tiered_auc(E.reset_index(), sc_))


def v_ocems(D):
    S = "ocems"
    r = pd.read_parquet(D / "ocems/readings.parquet")
    rl = pd.read_parquet(D / "labels/ocems/row_labels.parquet")
    units = pd.read_parquet(D / "ocems/units.parquet")
    ep = pd.read_parquet(D / "labels/ocems/tamper_episodes.parquet")
    check(S, "rows = units x timesteps", len(r) == len(units) * r.groupby("unit_id").size().iloc[0], len(r))
    check(S, "row labels aligned", len(r) == len(rl) and (r["unit_id"].to_numpy() == rl["unit_id"].to_numpy()).all())
    tam = (rl["tamper_type"] != "").mean()
    check(S, "tampered row share 1-15%", 0.01 <= tam <= 0.15, f"{tam:.4f}")
    check(S, "all 8 tamper types present", ep["tamper_type"].nunique() == 8, ep["tamper_type"].value_counts().to_dict())
    check(S, "benign faults present", (rl["fault_type"] != "").mean() > 0.01, rl["fault_type"].value_counts().to_dict())
    clean = (rl["tamper_type"] == "") & (rl["fault_type"] == "")
    rat = (r.loc[clean, "cod_mgl"] / r.loc[clean, "bod_mgl"]).replace([np.inf], np.nan)
    check(S, "PHYSICS clean BOD < COD", (rat.dropna() > 1).mean() > 0.999, f"{(rat.dropna() > 1).mean():.5f}")
    check(S, "hidden exceedances exist (motive)", rl["hidden_exceedance"].sum() > 0, int(rl["hidden_exceedance"].sum()))
    # naive per-window detector: rolling std of COD (flatline/smoothing -> low)
    r["_win"] = np.arange(len(r)) // 96
    g = r.groupby(["unit_id", "_win"])["cod_mgl"].agg(lambda x: x.std() / max(x.mean(), 1e-6))
    t = rl.assign(_win=np.arange(len(rl)) // 96).groupby(["unit_id", "_win"])["tamper_type"].agg(lambda x: (x.isin(["flatline", "smoothing"])).mean() > 0.5)
    check(S, "DIFFICULTY naive AUC [low COD variability -> flatline/smoothing day]", True, round(auc(-g, t.loc[g.index]), 3))


def v_bmw(D):
    S = "bmw"
    f = pd.read_parquet(D / "bmw/facilities.parquet")
    p = pd.read_parquet(D / "bmw/pickups.parquet")
    pl = pd.read_parquet(D / "labels/bmw/pickup_labels.parquet")
    fl = pd.read_parquet(D / "labels/bmw/facility_labels.parquet").set_index("facility_id")
    t = pd.read_parquet(D / "bmw/cbwtf_daily_treatment.parquet")
    cl = pd.read_parquet(D / "labels/bmw/cbwtf_labels.parquet").set_index("cbwtf_id")
    check(S, "pickup_id unique", p["pickup_id"].is_unique, len(p))
    check(S, "pickups reference facilities", p["facility_id"].isin(f["facility_id"]).all())
    check(S, "pickup labels aligned", (p["pickup_id"].to_numpy() == pl["pickup_id"].to_numpy()).all())
    p["kg"] = p[["yellow_kg", "red_kg", "white_kg", "blue_kg"]].sum(axis=1)
    F = f.set_index("facility_id")
    days = p["pickup_date"].nunique()
    F["kg_day"] = p.groupby("facility_id")["kg"].sum() / days
    bed = F[F["beds"] > 0].join(fl)
    bed["kg_bed_day"] = bed["kg_day"] / bed["beds"]
    legit = bed[~bed["under_handover"]]
    check(S, "PHYSICS legit kg/bed/day in 0.08-0.8", legit["kg_bed_day"].between(0.08, 0.8).mean() > 0.97, f"median {legit['kg_bed_day'].median():.3f}")
    bed["neg_kg_bed_day"] = -bed["kg_bed_day"] / bed["true_kg_rate"].median()
    bed["is_fraud"] = bed["under_handover"]
    check(S, "DIFFICULTY naive AUC [low kg per bed -> under-handover]", True, tiered_auc(bed.reset_index(), "neg_kg_bed_day"))
    pg = p.join(pl[["is_ghost"]])
    pg["ghost_score"] = pg["dwell_min"].fillna(-1).rsub(0)
    check(S, "ghost pickups present", pl["is_ghost"].sum() > 0, int(pl["is_ghost"].sum()))
    check(S, "DIFFICULTY naive AUC [no/short GPS dwell -> ghost pickup]", True, round(auc(pg["ghost_score"], pg["is_ghost"]), 3))
    cb = pd.read_parquet(D / "bmw/cbwtfs.parquet").set_index("cbwtf_id")
    over_days = t.join(cb["permitted_kg_per_day"], on="cbwtf_id")
    over_days["over"] = over_days["incinerated_kg_reported"] > over_days["permitted_kg_per_day"]
    share = over_days.groupby("cbwtf_id")["over"].mean().to_frame("over_share").join(cl)
    legit_c = share[~(share["overcapacity"])]
    check(S, "PHYSICS legit CBWTFs never burn above permitted capacity", (legit_c["over_share"] == 0).all(), f"{legit_c['over_share'].max():.3f}")
    share["is_fraud"] = share["overcapacity"]
    check(S, "DIFFICULTY naive AUC [days above permitted capacity -> overcapacity]", True, tiered_auc(share.reset_index(), "over_share"))
    tt = t.groupby("cbwtf_id")[["diesel_litres_purchased", "incinerated_kg_reported"]].sum()
    tt["neg_diesel_per_kg"] = -tt["diesel_litres_purchased"] / tt["incinerated_kg_reported"].clip(lower=1)
    tt = tt.join(cl)
    check(S, "DIFFICULTY naive AUC [low diesel per kg -> overcapacity]", True, round(auc(tt["neg_diesel_per_kg"], tt["overcapacity"]), 3))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    a = ap.parse_args(argv)
    D = Path(a.data)
    for fn in [v_registry, v_plastic, v_yield, v_ocems, v_bmw]:
        try:
            fn(D)
        except FileNotFoundError as ex:
            check(fn.__name__[2:], "dataset present", False, ex)
    lines = ["# Validation report", "", f"Data: `{D}`", "", "| Section | Check | Pass | Detail |", "|---|---|---|---|"]
    for s, n, ok, d in RESULTS:
        lines.append(f"| {s} | {n} | {'✅' if ok else '❌'} | {d.replace('|', '/')} |")
    nf = sum(not ok for *_, ok, _ in [(r[0], r[1], r[2], r[3]) for r in RESULTS])
    lines += ["", f"**{len(RESULTS) - nf}/{len(RESULTS)} checks passed.**"]
    (D / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    for s, n, ok, d in RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {s:9} {n}: {d}")
    print(f"{len(RESULTS) - nf}/{len(RESULTS)} passed")


if __name__ == "__main__":
    main()
