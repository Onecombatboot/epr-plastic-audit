"""Evidence collection and case files: turn a flagged recycler into something an inspector can act on.

Every reason in a case file points at concrete rows (invoice ids, bill ids, lot ids, collector ids,
partner entity ids) so the claim can be checked, and every triggered rule adds a concrete inspection step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rules import RULES, rule_matrix, rule_reasons

CHECKLIST = {
    "energy": "Obtain the DISCOM consumption statement and meter readings for the flagged months; compare connected machinery (kW) against the claimed tonnage.",
    "bills": "Verify the electricity bills directly with the DISCOM (units, tariff, amount); the uploaded copies do not reconcile.",
    "capacity": "Check installed machinery and shift records against the declared capacity; request production registers for the flagged months.",
    "price": "Ask for the plant's cost sheet; certificates priced below the cost of recycling imply the processing cost was never incurred.",
    "network": "Trace the material flow between the plants in the loop (weighbridge slips, vehicle GPS); the same tonnes may be invoiced round a ring.",
    "feedstock": "Physically verify a sample of purchase invoices with the sellers (weighbridge slips, e-way bills, vehicle logs).",
    "collectors": "Visit the named collectors: confirm premises, vehicles and storage; check their registration dates and sales to other buyers.",
    "cert": "Cross-check the certificate generation dates and category mix against the plant's process route and stock registers.",
    "output": "Reconcile product sales with processed tonnes; contact the product buyers listed and verify that they received the material.",
    "identity": "Compare the plant against the entities sharing its identifiers (bank, address, director, photos); consider a joint inspection.",
    "benford": "Review the invoice quantity distribution; digit patterns can indicate fabricated figures.",
    "registry": "Check the company's incorporation and registration history and the source of its capital.",
}


class EvidenceStore:
    def __init__(self, t: dict[str, pd.DataFrame], F: pd.DataFrame):
        self.t = t
        self.F = F
        fi = t["feedstock_invoices"]
        self._fi_idx = fi.groupby("buyer_id", observed=True).indices
        self._b_idx = t["electricity_bills"].groupby("entity_id", observed=True).indices
        self._lot_idx = t["certificate_lots"].groupby("recycler_id", observed=True).indices
        self._sale_idx = t["certificate_sales"].groupby("seller_id", observed=True).indices
        self._out_idx = t["output_invoices"].groupby("seller_id", observed=True).indices
        self._mpr_idx = t["monthly_processing_reported"].groupby("entity_id", observed=True).indices
        self.col = t["collectors"].set_index("collector_id")
        self.ent = t["entities"].set_index("entity_id")
        self.brands = t["brands"].set_index("brand_id")

    def _rows(self, name: str, idx: dict, key: str) -> pd.DataFrame:
        i = idx.get(key)
        return self.t[name].iloc[i] if i is not None else self.t[name].iloc[0:0]

    # ---------------------------------------------------------------------------------------------
    def evidence_for(self, eid: str, groups: set[str]) -> dict:
        ev: dict[str, list] = {}
        F = self.F.loc[eid]
        if groups & {"energy", "bills", "capacity"}:
            b = self._rows("electricity_bills", self._b_idx, eid).copy()
            mpr = self._rows("monthly_processing_reported", self._mpr_idx, eid)
            ref_kwh = {"recycling": 300, "waste_to_oil": 150, "co_processing": 20, "waste_to_energy": 30}
            m = mpr.assign(kmin=mpr["eol_type"].map(ref_kwh) * mpr["processed_t_reported"]).groupby("month")[["processed_t_reported", "kmin"]].sum()
            b = b.merge(m, left_on="billing_month", right_index=True, how="left")
            b["ratio"] = b["units_kwh"] / b["kmin"]
            worst = b.sort_values("ratio").head(3)
            ev["electricity_bills"] = [f"plastic/electricity_bills:{r.bill_id} ({r.billing_month:%b %Y}: {r.units_kwh:,.0f} kWh for {r.processed_t_reported:,.1f} t claimed)"
                                       for r in worst.itertuples() if pd.notna(r.ratio)]
        if groups & {"feedstock", "collectors", "benford"}:
            fi = self._rows("feedstock_invoices", self._fi_idx, eid)
            if len(fi):
                odd = fi[(np.abs(fi.tonnes * 2 - np.round(fi.tonnes * 2)) < 1e-9) | (fi.invoice_date.dt.weekday == 6)]
                ev["feedstock_invoices"] = [f"plastic/feedstock_invoices:{r.invoice_id} ({r.invoice_date:%d %b %Y}, {r.tonnes:g} t from {r.seller_id})" for r in odd.head(4).itertuples()]
                cs = fi[fi.seller_type == "collector"].groupby("seller_id")["tonnes"].sum().sort_values(ascending=False).head(5)
                rows = []
                for cid, tt in cs.items():
                    c = self.col.loc[cid]
                    rows.append(f"plastic/collectors:{cid} ({tt:,.0f} t supplied; declared capacity {c.declared_monthly_capacity_t:.0f} t/month; {c.vehicle_count} vehicles; registered {c.registration_date:%b %Y})")
                ev["top_collectors"] = rows
        if "cert" in groups:
            lots = self._rows("certificate_lots", self._lot_idx, eid)
            if len(lots):
                lots = lots.assign(ym=lots.generated_on.dt.to_period("M"))
                top = lots.groupby("ym")["tonnes"].sum().sort_values(ascending=False).head(1)
                ev["certificate_lots"] = [f"plastic/certificate_lots (month {top.index[0]}: {top.iloc[0]:,.0f} t of {lots.tonnes.sum():,.0f} t generated)"]
        if "network" in groups:
            fi = self._rows("feedstock_invoices", self._fi_idx, eid)
            partners = fi[fi.seller_type == "recycler"].groupby("seller_id")["tonnes"].sum().sort_values(ascending=False).head(4)
            ev["ring_partners"] = [f"registry/entities:{p} ({tt:,.0f} t sold to this plant)" for p, tt in partners.items()]
        if "output" in groups:
            o = self._rows("output_invoices", self._out_idx, eid)
            if len(o):
                top = o.groupby("buyer_id")["tonnes"].sum().sort_values(ascending=False).head(3)
                ev["output_buyers"] = [f"plastic/offtakers:{b} ({tt:,.0f} t bought)" for b, tt in top.items()]
        if "identity" in groups:
            ev["identity"] = [f"registry: shares identifiers with other entities (bank {F.get('id_shared_bank', 0):.0f}, address {F.get('id_shared_address', 0):.0f}, "
                              f"director {F.get('id_shared_director', 0):.0f}, photos {F.get('id_photo_near_dups', 0):.0f})"]
        if "price" in groups:
            s = self._rows("certificate_sales", self._sale_idx, eid)
            if len(s):
                cheap = s.assign(r=s.price_inr_per_t).sort_values("r").head(3)
                ev["cheapest_sales"] = [f"plastic/certificate_sales:{r.sale_id} ({r.category}, Rs {r.price_inr_per_t:,.0f}/t to {r.buyer_id})" for r in cheap.itertuples()]
        return ev

    def brands_exposed(self, eid: str, top: int = 5) -> list[str]:
        s = self._rows("certificate_sales", self._sale_idx, eid)
        if not len(s):
            return []
        g = s.groupby("buyer_id")["tonnes"].sum().sort_values(ascending=False).head(top)
        return [f"{b} {self.brands.loc[b, 'name']} ({tt:,.0f} t)" for b, tt in g.items()]

    # ---------------------------------------------------------------------------------------------
    def case_file(self, eid: str, scores: pd.Series, shap_reasons: list[dict] | None = None) -> dict:
        reasons = rule_reasons(self.F, eid, top=6)
        groups = {r["group"] for r in reasons} | ({"feedstock", "cert"} if not reasons else set())
        ev = self.evidence_for(eid, groups)
        e = self.ent.loc[eid]
        checklist = [CHECKLIST[g] for g in dict.fromkeys(r["group"] for r in reasons) if g in CHECKLIST]
        return {
            "entity_id": eid,
            "profile": {"name": e.legal_name, "state": e.state, "district": e.district, "declared_capacity_tpa": float(e.declared_capacity_tpa),
                        "connected_load_kw": float(e.connected_load_kw), "epr_registered": str(e.epr_registration_date.date())},
            "scores": {k: round(float(v), 4) for k, v in scores.items()},
            "reasons": reasons,
            "model_signals": shap_reasons or [],
            "evidence": ev,
            "brands_exposed": self.brands_exposed(eid),
            "inspection_checklist": checklist,
        }
