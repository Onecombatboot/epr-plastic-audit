"""Recycler-level feature engineering for Phantom Plastic.

Everything here is computed from *observable* tables (invoices, bills, certificates, registry).
No label is read. Each feature group answers one auditor question:

  capacity   can the plant physically process what it claims?
  energy     does its electricity use fit the claimed tonnes?
  feedstock  are the purchase invoices real-looking and independent?
  collectors are the sellers real businesses with real capacity?
  cert       how and when were certificates generated?
  price      are certificates sold below the cost of recycling?
  output     do sold products match processed tonnes, and who bought them?
  network    does money/material circulate (round-tripping)?
  benford    do invoice quantities look naturally distributed?
  identity   does the plant share phone/email/bank/address/director/photo with other plants?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_io import _fy_fast

EOLS = ["recycling", "waste_to_oil", "co_processing", "waste_to_energy"]
BENFORD = np.log10(1 + 1 / np.arange(1, 10))


def _safe_div(a, b):
    """Division that returns NaN where the denominator is <= 0; Series stay index-aligned."""
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        a = pd.Series(a) if not isinstance(a, pd.Series) else a
        b = pd.Series(b) if not isinstance(b, pd.Series) else b
        return a / b.where(b > 0)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def _hhi(df: pd.DataFrame, key: str, val: str, by: str = "entity_id") -> pd.DataFrame:
    g = df.groupby([by, key], observed=True)[val].sum().rename("v").reset_index()
    tot = g.groupby(by)["v"].transform("sum")
    g["s"] = g["v"] / tot
    out = g.groupby(by).agg(hhi=("s", lambda s: float((s ** 2).sum())), top1=("s", "max"), n=("s", "size"))
    return out


def _benford_mad(values: pd.Series, groups: pd.Series, min_n: int = 40) -> pd.Series:
    v = values.to_numpy(dtype=float)
    ok = v > 0
    v = v[ok]
    g = groups.to_numpy()[ok]
    first = np.floor(v / 10.0 ** np.floor(np.log10(v))).astype(int)
    first = np.clip(first, 1, 9)
    df = pd.DataFrame({"g": g, "d": first})
    ct = pd.crosstab(df["g"], df["d"]).reindex(columns=range(1, 10), fill_value=0)
    n = ct.sum(axis=1)
    freq = ct.div(n, axis=0)
    mad = (freq - BENFORD).abs().mean(axis=1)
    mad[n < min_n] = np.nan
    return mad


def collector_table(fi: pd.DataFrame, col: pd.DataFrame) -> pd.DataFrame:
    """Per-collector behaviour over all buyers: peak monthly volume vs declared capacity, buyer count, age at first invoice."""
    ci = fi[fi.seller_type == "collector"]
    cm = ci.groupby(["seller_id", "month"], observed=True)["tonnes"].sum().rename("t").reset_index()
    cm["cap"] = cm["seller_id"].map(col["declared_monthly_capacity_t"])
    cm["ratio"] = cm["t"] / cm["cap"]
    cstat = cm.groupby("seller_id").agg(c_ratio_max=("ratio", "max"), c_months=("month", "nunique"))
    cstat["c_n_buyers"] = ci.groupby("seller_id")["entity_id"].nunique()
    first_inv = ci.groupby("seller_id")["invoice_date"].min()
    cstat["c_age_at_first_inv_days"] = (first_inv - col["registration_date"].reindex(first_inv.index)).dt.days
    cstat["c_vehicles"] = col["vehicle_count"].reindex(cstat.index)
    return cstat


def offtaker_table(o: pd.DataFrame, ofk: pd.DataFrame) -> pd.DataFrame:
    ost = o.groupby("buyer_id").agg(o_n_sellers=("entity_id", "nunique"), o_first=("invoice_date", "min"), o_t=("tonnes", "sum"))
    ost["o_age_days"] = (ost["o_first"] - ofk["incorporation_date"].reindex(ost.index)).dt.days
    return ost


# ---------------------------------------------------------------------------------------------
def build_features(t: dict[str, pd.DataFrame], ref: dict, with_identity: bool = True,
                   registry_dir=None) -> pd.DataFrame:
    ent = t["entities"].set_index("entity_id")
    ids = ent.index
    F = pd.DataFrame(index=ids)
    hours = ref["hours_per_month"]
    kmin = {k: v[0] for k, v in ref["eol_kwh_per_t"].items()}
    kmax = {k: v[1] for k, v in ref["eol_kwh_per_t"].items()}

    # ------------------------------------------------------------------ monthly reported
    mpr = t["monthly_processing_reported"].copy()
    mpr["fy"] = _fy_fast(mpr["month"])
    em = mpr.groupby(["entity_id", "month", "eol_type"], observed=True)["processed_t_reported"].sum().unstack("eol_type").reindex(columns=EOLS).fillna(0.0)
    em["t"] = em[EOLS].sum(axis=1)
    em["kwh_min"] = sum(em[e] * kmin[e] for e in EOLS)
    em["kwh_max"] = sum(em[e] * kmax[e] for e in EOLS)
    em["saleable_yield_min"] = sum(em[e] * ref["eol_yield"][e][0] for e in EOLS)
    em = em.reset_index()
    em["fy"] = _fy_fast(em["month"])
    cap = ent["declared_capacity_tpa"]
    em["cap_m"] = em["entity_id"].map(cap) / 12.0
    em["cap_ratio_m"] = em["t"] / em["cap_m"]

    g = em.groupby("entity_id")
    F["total_reported_t"] = g["t"].sum()
    F["active_months"] = g["t"].apply(lambda s: int((s > 0).sum()))
    fyt = em.groupby(["entity_id", "fy"])["t"].sum().unstack("fy")
    F["cap_ratio_fy_max"] = fyt.max(axis=1) / cap
    F["cap_ratio_month_max"] = g["cap_ratio_m"].max()
    F["cap_ratio_month_p90"] = g["cap_ratio_m"].quantile(0.9)
    F["over_cap_month_share"] = g["cap_ratio_m"].apply(lambda s: float((s > 1.0).mean()))
    F["reported_cv"] = g["t"].apply(lambda s: float(s.std() / s.mean()) if s.mean() > 0 else np.nan)
    # growth: last-12 vs first-12 active months mean
    def _growth(s: pd.Series) -> float:
        s = s[s > 0]
        if len(s) < 8:
            return np.nan
        k = max(3, len(s) // 3)
        return float(np.log((s.iloc[-k:].mean() + 1e-6) / (s.iloc[:k].mean() + 1e-6)))
    F["log_growth"] = g["t"].apply(_growth)
    eolt = em.groupby("entity_id")[EOLS].sum()
    eol_share = eolt.div(eolt.sum(axis=1), axis=0).fillna(0)
    F["eol_recycling_share"] = eol_share["recycling"]
    F["kwh_min_per_t"] = _safe_div(g["kwh_min"].sum(), g["t"].sum())

    # ------------------------------------------------------------------ energy / bills
    b = t["electricity_bills"].copy()
    b["billing_month"] = pd.to_datetime(b["billing_month"])
    b = b.merge(em[["entity_id", "month", "t", "kwh_min", "kwh_max"]], left_on=["entity_id", "billing_month"],
                right_on=["entity_id", "month"], how="left")
    b[["t", "kwh_min", "kwh_max"]] = b[["t", "kwh_min", "kwh_max"]].fillna(0.0)
    bg = b.groupby("entity_id")
    F["bill_months"] = bg.size()
    F["bill_coverage"] = F["bill_months"].fillna(0) / F["active_months"].replace(0, np.nan)
    sums = bg[["units_kwh", "kwh_min", "kwh_max", "t"]].sum()
    F["energy_ratio"] = _safe_div(sums["units_kwh"], sums["kwh_min"])            # units / minimum plausible units
    F["kwh_per_t"] = _safe_div(sums["units_kwh"], sums["t"])
    b["er_m"] = _safe_div(b["units_kwh"], b["kwh_min"])
    F["energy_ratio_p10"] = b.groupby("entity_id")["er_m"].quantile(0.10)
    F["energy_ratio_median"] = b.groupby("entity_id")["er_m"].median()
    F["energy_low_month_share"] = b.groupby("entity_id")["er_m"].apply(lambda s: float((s < 0.8).mean()))
    load = ent["connected_load_kw"]
    em["load_util"] = em["kwh_min"] / (hours * em["entity_id"].map(load))
    F["load_util_max"] = em.groupby("entity_id")["load_util"].max()
    F["load_util_p90"] = em.groupby("entity_id")["load_util"].quantile(0.9)
    req_kw = cap * F["kwh_min_per_t"] / (8760 * ref["max_utilisation"])
    F["cap_vs_load"] = req_kw / load                                            # >1: capacity needs more power than connected
    F["solar_declared"] = (ent["captive_solar_kw"].fillna(0) > 0).astype(float)
    F["solar_frac_of_load"] = ent["captive_solar_kw"].fillna(0) / load
    # bill internal consistency (units x tariff vs amount charged)
    tar = t["tariff"].rename(columns={"fy": "fy"})
    b["state"] = b["entity_id"].map(ent["state"])
    b = b.merge(tar, on=["state", "fy"], how="left")
    b["e_ratio"] = _safe_div(b["energy_charge_inr"], b["units_kwh"] * b["energy_rate_inr_per_kwh"])
    b["f_ratio"] = _safe_div(b["fixed_charge_inr"], b["sanctioned_load_kw"] * b["fixed_rate_inr_per_kw_month"])
    b["tot_ratio"] = _safe_div(b["total_amount_inr"], b["energy_charge_inr"] + b["fixed_charge_inr"] + b["duty_inr"])
    bg = b.groupby("entity_id")
    F["bill_energy_ratio_mean"] = bg["e_ratio"].mean()
    F["bill_energy_ratio_min"] = bg["e_ratio"].min()
    F["bill_inconsistent_share"] = bg["e_ratio"].apply(lambda s: float(((s - 1).abs() > 0.04).mean()))
    F["bill_fixed_ratio_mean"] = bg["f_ratio"].mean()
    F["bill_sanctioned_vs_registry"] = bg["sanctioned_load_kw"].median() / load
    # a doctored bill shows up as units spiking above what the *neighbouring* months' amounts imply
    F["bill_units_cv"] = bg["units_kwh"].apply(lambda s: float(s.std() / s.mean()) if s.mean() > 0 else np.nan)

    # ------------------------------------------------------------------ feedstock (inbound invoices)
    fi = t["feedstock_invoices"]
    fi = fi.rename(columns={"buyer_id": "entity_id"})
    fi["invoice_date"] = pd.to_datetime(fi["invoice_date"])
    fi["month"] = fi["invoice_date"].dt.to_period("M").dt.to_timestamp()
    fi["is_round"] = (np.abs(fi["tonnes"] * 2 - np.round(fi["tonnes"] * 2)) < 1e-9).astype(np.int8)
    fi["is_sunday"] = (fi["invoice_date"].dt.weekday == 6).astype(np.int8)
    fi["is_monthend"] = (fi["invoice_date"].dt.day >= 28).astype(np.int8)
    big = fi["taxable_value_inr"] > ref["eway_threshold_inr"]
    fi["eway_missing_big"] = (big & fi["eway_bill_no"].isna()).astype(np.int8)
    fi["is_big"] = big.astype(np.int8)
    fi["far"] = (fi["distance_km"] > 800).astype(np.int8)
    med_rate = fi.groupby("category", observed=True)["rate_inr_per_t"].transform("median")
    fi["rate_dev"] = np.abs(np.log(fi["rate_inr_per_t"].clip(lower=1) / med_rate.clip(lower=1)))
    col = t["collectors"].set_index("collector_id")
    # collector-level behaviour (computed over all buyers)
    cstat = collector_table(fi, col)
    fc = fi[fi.seller_type == "collector"].merge(cstat.reset_index().rename(columns={"seller_id": "seller_id"}), on="seller_id", how="left")
    fc["c_over_cap"] = (fc["c_ratio_max"] > 1.5).astype(np.int8)
    fc["c_young"] = (fc["c_age_at_first_inv_days"] < 180).astype(np.int8)
    fc["c_single_buyer"] = (fc["c_n_buyers"] == 1).astype(np.int8)
    fc["w_over_cap"] = fc["c_over_cap"] * fc["tonnes"]
    fc["w_young"] = fc["c_young"] * fc["tonnes"]
    fc["w_single"] = fc["c_single_buyer"] * fc["tonnes"]
    # vehicle reuse: same vehicle appears on the same day for different buyers
    veh = fc.groupby(["vehicle_reg", "invoice_date"], observed=True)["entity_id"].transform("nunique")
    fc["veh_multi"] = (veh > 1).astype(np.int8)

    fg = fi.groupby("entity_id")
    F["feed_t"] = fg["tonnes"].sum()
    F["feed_n_inv"] = fg.size()
    F["feed_round_share"] = fg["is_round"].mean()
    F["feed_sunday_share"] = fg["is_sunday"].mean()
    F["feed_monthend_share"] = fg["is_monthend"].mean()
    F["feed_far_share"] = fg["far"].mean()
    F["feed_rate_dev"] = fg["rate_dev"].mean()
    bigsum = fg[["is_big", "eway_missing_big"]].sum()
    F["feed_eway_missing"] = _safe_div(bigsum["eway_missing_big"], bigsum["is_big"])
    F["feed_inv_per_t"] = F["feed_n_inv"] / F["feed_t"]
    F["feed_mean_inv_t"] = F["feed_t"] / F["feed_n_inv"]
    cg = fc.groupby("entity_id")
    ct = cg["tonnes"].sum()
    F["feed_collector_over_cap"] = _safe_div(cg["w_over_cap"].sum(), ct)
    F["feed_collector_young"] = _safe_div(cg["w_young"].sum(), ct)
    F["feed_collector_single_buyer"] = _safe_div(cg["w_single"].sum(), ct)
    F["feed_vehicle_multi"] = cg["veh_multi"].mean()
    F["feed_from_recyclers"] = _safe_div(fi[fi.seller_type == "recycler"].groupby("entity_id")["tonnes"].sum().reindex(ids).fillna(0), F["feed_t"])
    h = _hhi(fc, "seller_id", "tonnes")
    F["feed_collector_hhi"] = h["hhi"]
    F["feed_collector_top1"] = h["top1"]
    F["feed_n_collectors"] = h["n"]
    # coverage: invoiced tonnes vs reported tonnes (fabricated invoices tend to cover claims tightly)
    fm = fi.groupby(["entity_id", "month"])["tonnes"].sum().rename("inv_t").reset_index()
    cv = em[["entity_id", "month", "t"]].merge(fm, on=["entity_id", "month"], how="left").fillna({"inv_t": 0.0})
    cv["cov"] = _safe_div(cv["inv_t"], cv["t"])
    cvg = cv[cv["t"] > 0].groupby("entity_id")
    F["feed_cov_total"] = _safe_div(cvg["inv_t"].sum(), cvg["t"].sum())
    F["feed_cov_median"] = cvg["cov"].median()
    F["feed_cov_tight_share"] = cvg["cov"].apply(lambda s: float(((s >= 0.97) & (s <= 1.05)).mean()))
    F["feed_cov_std"] = cvg["cov"].std()

    # ------------------------------------------------------------------ certificates
    lots = t["certificate_lots"].rename(columns={"recycler_id": "entity_id"})
    lots["lag_days"] = (lots["generated_on"] - (lots["period_month"] + pd.offsets.MonthEnd(0))).dt.days
    lots["gen_ym"] = lots["generated_on"].dt.to_period("M").dt.to_timestamp()
    lg = lots.groupby("entity_id")
    F["cert_t"] = lg["tonnes"].sum()
    cm_ = lots.groupby(["entity_id", "category"], observed=True)["tonnes"].sum().unstack("category").fillna(0)
    for c in ["CAT_I", "CAT_II", "CAT_III", "CAT_IV"]:
        F[f"cert_share_{c}"] = _safe_div(cm_.get(c, pd.Series(0.0, index=cm_.index)).reindex(ids).fillna(0), F["cert_t"])
    exp_cat3 = sum(eol_share[e] * ref["eol_expected_cat_mix"][e][2] for e in EOLS)
    exp_cat1 = sum(eol_share[e] * ref["eol_expected_cat_mix"][e][0] for e in EOLS)
    F["cert_cat3_excess"] = F["cert_share_CAT_III"] - exp_cat3.reindex(ids)
    F["cert_cat1_deficit"] = exp_cat1.reindex(ids) - F["cert_share_CAT_I"]
    F["cert_vs_reported"] = _safe_div(F["cert_t"], F["total_reported_t"])
    lots["late"] = (lots["lag_days"] > 45).astype(np.int8)
    lots["w_late"] = lots["late"] * lots["tonnes"]
    F["cert_late_batch_share"] = _safe_div(lots.groupby("entity_id")["w_late"].sum(), F["cert_t"])
    F["cert_lag_mean"] = lg["lag_days"].mean()
    gm = lots.groupby(["entity_id", "gen_ym"])["tonnes"].sum().rename("v").reset_index()
    gm["s"] = gm["v"] / gm.groupby("entity_id")["v"].transform("sum")
    F["cert_gen_top_month_share"] = gm.groupby("entity_id")["s"].max()
    F["cert_gen_hhi"] = gm.groupby("entity_id")["s"].apply(lambda s: float((s ** 2).sum()))
    # certificates generated in the same calendar month as the compliance deadline (June)
    lots["june"] = (lots["generated_on"].dt.month == 6).astype(np.int8) * lots["tonnes"]
    F["cert_june_share"] = _safe_div(lots.groupby("entity_id")["june"].sum(), F["cert_t"])

    # ------------------------------------------------------------------ price
    s = t["certificate_sales"].rename(columns={"seller_id": "entity_id"})
    floor = pd.Series(ref["cert_cost_floor_inr_per_t"])
    s["floor"] = s["category"].map(floor).astype(float)
    s["pf"] = s["price_inr_per_t"] / s["floor"]
    s["market"] = s.groupby(["category", "fy"], observed=True)["price_inr_per_t"].transform("median")
    s["pm"] = s["price_inr_per_t"] / s["market"]
    s["w_pf"] = s["pf"] * s["tonnes"]
    s["w_pm"] = s["pm"] * s["tonnes"]
    s["below"] = (s["pf"] < 0.9).astype(np.int8) * s["tonnes"]
    sg = s.groupby("entity_id")
    st = sg["tonnes"].sum()
    F["sold_t"] = st
    F["price_to_floor"] = _safe_div(sg["w_pf"].sum(), st)
    F["price_vs_market"] = _safe_div(sg["w_pm"].sum(), st)
    F["price_below_floor_share"] = _safe_div(sg["below"].sum(), st)
    F["price_p10_vs_market"] = sg["pm"].quantile(0.10)
    F["sell_through"] = _safe_div(st, F["cert_t"])
    hb = _hhi(s.rename(columns={"buyer_id": "brand"}), "brand", "tonnes")
    F["sales_n_brands"] = hb["n"]
    F["sales_brand_hhi"] = hb["hhi"]
    F["sales_brand_top1"] = hb["top1"]
    # fraction sold within the last days of the financial year (rush to the compliance deadline)
    s["rush"] = (s["sale_date"].dt.month.isin([3])).astype(np.int8) * s["tonnes"]
    F["sales_fyend_share"] = _safe_div(sg["rush"].sum(), st)

    # ------------------------------------------------------------------ outputs
    o = t["output_invoices"].rename(columns={"seller_id": "entity_id"})
    o["invoice_date"] = pd.to_datetime(o["invoice_date"])
    o["is_round"] = (np.abs(o["tonnes"] * 2 - np.round(o["tonnes"] * 2)) < 1e-9).astype(np.int8)
    o["is_monthend"] = (o["invoice_date"].dt.day >= 28).astype(np.int8)
    o["month"] = o["invoice_date"].dt.to_period("M").dt.to_timestamp()
    prod_med = o.groupby("product", observed=True)["rate_inr_per_t"].transform("median")
    o["rate_dev"] = np.log(o["rate_inr_per_t"].clip(lower=1) / prod_med.clip(lower=1))
    ofk = t["offtakers"].set_index("offtaker_id")
    ost = offtaker_table(o, ofk)
    o = o.merge(ost[["o_n_sellers", "o_age_days"]], left_on="buyer_id", right_index=True, how="left")
    o["young"] = (o["o_age_days"] < 365).astype(np.int8) * o["tonnes"]
    o["thin"] = (o["o_n_sellers"] <= 2).astype(np.int8) * o["tonnes"]
    og = o.groupby("entity_id")
    ot = og["tonnes"].sum()
    F["out_t"] = ot
    saleable_in = em.groupby("entity_id")["saleable_yield_min"].sum()
    F["out_to_expected"] = _safe_div(ot.reindex(ids).fillna(0), saleable_in.reindex(ids))
    F["out_young_offtaker_share"] = _safe_div(og["young"].sum(), ot)
    F["out_thin_offtaker_share"] = _safe_div(og["thin"].sum(), ot)
    F["out_round_share"] = og["is_round"].mean()
    F["out_monthend_share"] = og["is_monthend"].mean()
    F["out_rate_dev_mean"] = og["rate_dev"].mean()
    ho = _hhi(o.rename(columns={"buyer_id": "ofk"}), "ofk", "tonnes")
    F["out_n_offtakers"] = ho["n"]
    F["out_top1"] = ho["top1"]
    # output vs input month by month
    om = o.groupby(["entity_id", "month"])["tonnes"].sum().rename("out_t_m").reset_index()
    mm = em[["entity_id", "month", "saleable_yield_min", "t"]].merge(om, on=["entity_id", "month"], how="left").fillna({"out_t_m": 0.0})
    mm = mm[mm["saleable_yield_min"] > 0]
    mm["r"] = mm["out_t_m"] / mm["saleable_yield_min"]
    F["out_month_ratio_p25"] = mm.groupby("entity_id")["r"].quantile(0.25)
    F["out_month_low_share"] = mm.groupby("entity_id")["r"].apply(lambda s: float((s < 0.6).mean()))

    # ------------------------------------------------------------------ network (recycler -> recycler flows)
    import networkx as nx
    rr = fi[fi.seller_type == "recycler"].groupby(["seller_id", "entity_id"])["tonnes"].sum().rename("t").reset_index()
    G = nx.DiGraph()
    for a, bb, w in rr.itertuples(index=False):
        G.add_edge(a, bb, t=float(w))
    in_cycle: dict[str, int] = {}
    cyc_edge_t: dict[str, float] = {}
    for cyc in nx.simple_cycles(G, length_bound=6):
        k = len(cyc)
        for i, n in enumerate(cyc):
            in_cycle[n] = min(in_cycle.get(n, 99), k)
            nxt = cyc[(i + 1) % k]
            cyc_edge_t[nxt] = cyc_edge_t.get(nxt, 0.0) + G[n][nxt]["t"]
    F["net_in_cycle"] = pd.Series(in_cycle).reindex(ids).notna().astype(float)
    F["net_cycle_len"] = pd.Series(in_cycle).reindex(ids)
    F["net_cycle_feed_share"] = _safe_div(pd.Series(cyc_edge_t).reindex(ids).fillna(0), F["feed_t"])
    recip = set((a, bb) for a, bb in G.edges if G.has_edge(bb, a))
    r_in = pd.Series({n: sum(1 for p in G.predecessors(n) if (p, n) in recip) for n in G.nodes}).reindex(ids).fillna(0)
    F["net_reciprocal_edges"] = r_in
    F["net_recycler_in_degree"] = pd.Series(dict(G.in_degree())).reindex(ids).fillna(0)
    F["net_recycler_out_degree"] = pd.Series(dict(G.out_degree())).reindex(ids).fillna(0)

    # ------------------------------------------------------------------ Benford
    F["benford_feed_mad"] = _benford_mad(fi["tonnes"], fi["entity_id"]).reindex(ids)
    F["benford_out_mad"] = _benford_mad(o["tonnes"], o["entity_id"]).reindex(ids)
    F["benford_reported_mad"] = _benford_mad(mpr["processed_t_reported"], mpr["entity_id"], min_n=24).reindex(ids)

    # ------------------------------------------------------------------ registry-derived
    reg_date = ent["epr_registration_date"]
    F["age_at_registration_days"] = (reg_date - ent["incorporation_date"]).dt.days
    first_lot = lots.groupby("entity_id")["generated_on"].min()
    F["first_cert_before_reg_days"] = (reg_date - first_lot.reindex(ids)).dt.days
    F["declared_capacity_tpa"] = cap
    F["connected_load_kw"] = load
    F["cap_per_kw"] = cap / load
    F["declared_staff"] = ent["declared_staff"]
    F["t_per_staff"] = F["total_reported_t"] / 3.0 / ent["declared_staff"].replace(0, np.nan)

    if with_identity:
        F = F.join(identity_features(ent.reset_index(), registry_dir or None, t), how="left")
    return F


def identity_features(ent: pd.DataFrame, registry_dir, t) -> pd.DataFrame:
    """Shared-identifier degree against *all* registered entities (every stream), consultant excluded
    (consultants legitimately file for many clients)."""
    from pathlib import Path

    from .data_io import DEFAULT_DATA
    d = Path(registry_dir) if registry_dir else DEFAULT_DATA / "registry"
    allent = pd.read_parquet(d / "entities.parquet")
    dirs = pd.read_parquet(d / "entity_directors.parquet")
    photos = pd.read_parquet(d / "plant_photos.parquet")
    cons = pd.read_parquet(d / "consultants.parquet")
    allent = allent.copy()
    # consultants legitimately file for many clients: a filing phone that belongs to a consultant is not evidence of common control
    allent["filing_phone"] = allent["filing_phone"].where(~allent["filing_phone"].isin(set(cons["phone"])))
    # e-mail addresses are derived from company-name stems and collide by chance; only phone is used
    cols = ["contact_phone", "filing_phone", "bank_account", "address", "authorised_signatory_id"]
    ids = ent["entity_id"]
    out = pd.DataFrame(index=ids)
    per_key = []
    for c in cols:
        grp = allent.groupby(c)["entity_id"].transform("nunique") - 1
        per_key.append(grp.rename(c).set_axis(allent["entity_id"]))
    k = pd.concat(per_key, axis=1)
    out["id_shared_phone"] = (k[["contact_phone", "filing_phone"]].max(axis=1)).reindex(ids)
    out["id_shared_bank"] = k["bank_account"].reindex(ids)
    out["id_shared_address"] = k["address"].reindex(ids)
    out["id_shared_signatory"] = k["authorised_signatory_id"].reindex(ids)
    dcol = [c for c in dirs.columns if c != "entity_id"][0]
    dd = dirs.copy()
    dd["n"] = dd.groupby(dcol)["entity_id"].transform("nunique") - 1
    out["id_shared_director"] = dd.groupby("entity_id")["n"].max().reindex(ids)
    # near-duplicate photo hashes (Hamming <= 6) across different entities (chunked all-pairs, numpy popcount)
    ph = photos[["entity_id", "phash"]].copy()
    vals = np.array([int(x, 16) for x in ph["phash"]], dtype=np.uint64)
    ent_codes = pd.factorize(ph["entity_id"])[0]
    near = np.zeros(len(ph), dtype=int)
    for i0 in range(0, len(vals), 1024):
        x = np.bitwise_xor(vals[i0:i0 + 1024, None], vals[None, :])
        ham = np.bitwise_count(x)
        same = ent_codes[i0:i0 + 1024, None] == ent_codes[None, :]
        near[i0:i0 + 1024] = ((ham <= 6) & ~same).sum(axis=1)
    ph["near"] = near
    out["id_photo_near_dups"] = ph.groupby("entity_id")["near"].max().reindex(ids)
    return out.fillna(0.0)


GROUPS = {
    "capacity": ["cap_ratio_fy_max", "cap_ratio_month_max", "cap_ratio_month_p90", "over_cap_month_share", "reported_cv", "log_growth", "cap_per_kw", "t_per_staff"],
    "energy": ["energy_ratio", "energy_ratio_p10", "energy_ratio_median", "energy_low_month_share", "kwh_per_t", "load_util_max", "load_util_p90",
               "cap_vs_load", "solar_declared", "solar_frac_of_load", "bill_coverage", "bill_units_cv", "bill_sanctioned_vs_registry"],
    "bills": ["bill_energy_ratio_mean", "bill_energy_ratio_min", "bill_inconsistent_share", "bill_fixed_ratio_mean"],
    "feedstock": ["feed_round_share", "feed_sunday_share", "feed_monthend_share", "feed_far_share", "feed_rate_dev", "feed_eway_missing", "feed_inv_per_t",
                  "feed_mean_inv_t", "feed_cov_total", "feed_cov_median", "feed_cov_tight_share", "feed_cov_std"],
    "collectors": ["feed_collector_over_cap", "feed_collector_young", "feed_collector_single_buyer", "feed_vehicle_multi", "feed_collector_hhi",
                   "feed_collector_top1", "feed_n_collectors", "feed_from_recyclers"],
    "cert": ["cert_share_CAT_I", "cert_share_CAT_III", "cert_cat3_excess", "cert_cat1_deficit", "cert_vs_reported", "cert_late_batch_share", "cert_lag_mean",
             "cert_gen_top_month_share", "cert_gen_hhi", "cert_june_share"],
    "price": ["price_to_floor", "price_vs_market", "price_below_floor_share", "price_p10_vs_market", "sell_through", "sales_n_brands", "sales_brand_hhi",
              "sales_brand_top1", "sales_fyend_share"],
    "output": ["out_to_expected", "out_young_offtaker_share", "out_thin_offtaker_share", "out_round_share", "out_monthend_share", "out_rate_dev_mean",
               "out_n_offtakers", "out_top1", "out_month_ratio_p25", "out_month_low_share"],
    "network": ["net_in_cycle", "net_cycle_feed_share", "net_reciprocal_edges", "net_recycler_in_degree", "net_recycler_out_degree"],
    "benford": ["benford_feed_mad", "benford_out_mad", "benford_reported_mad"],
    "registry": ["age_at_registration_days", "first_cert_before_reg_days"],
    "identity": ["id_shared_phone", "id_shared_bank", "id_shared_address", "id_shared_signatory", "id_shared_director", "id_photo_near_dups"],
}
