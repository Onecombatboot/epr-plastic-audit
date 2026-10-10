"""Layer 1: transparent rule detectors with human-readable reasons.

Each rule maps one feature to a severity in [0, 1] through a linear ramp between a "normal" value and
an "impossible/obviously wrong" value. Thresholds come from physics, tariffs or published norms where
possible (see reference.yaml); the rest are set at the tail of the population. Rule severities combine
by noisy-OR, weighted by how conclusive each rule is on its own:

    score = 1 - prod(1 - weight_i * severity_i)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


def ramp(x, lo: float, hi: float):
    """0 at lo, 1 at hi, linear in between (works in either direction)."""
    x = np.asarray(x, dtype=float)
    out = (x - lo) / (hi - lo)
    return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0)


@dataclass
class Rule:
    name: str
    group: str
    weight: float
    fn: Callable[[pd.DataFrame], np.ndarray]
    reason: str                      # str.format with the row's feature values


def _solar_adj_energy_ratio(F: pd.DataFrame) -> pd.Series:
    # Captive solar supplies part of the energy, so grid units understate real use. Heuristic correction:
    # solar kW as a share of connected load, capped, converted to an energy share.
    share = (0.8 * F["solar_frac_of_load"].fillna(0)).clip(upper=0.6)
    return F["energy_ratio"] / (1 - share)


RULES: list[Rule] = [
    Rule("impossible_load", "energy", 0.90, lambda F: ramp(F["load_util_max"], 0.9, 2.0),
         "Claimed tonnage in the busiest month would need {load_util_max:.0%} of the plant's connected load running non-stop "
         "(above 100% is physically impossible)"),
    Rule("capacity_needs_more_power", "energy", 0.55, lambda F: ramp(F["cap_vs_load"], 1.0, 2.0),
         "Declared capacity of {declared_capacity_tpa:,.0f} t/yr needs about {cap_vs_load:.1f}x the power the plant is connected for"),
    Rule("over_declared_capacity", "capacity", 0.80, lambda F: ramp(F["over_cap_month_share"], 0.0, 0.3),
         "Certificates claim more than the declared monthly capacity in {over_cap_month_share:.0%} of months"),
    Rule("energy_deficit", "energy", 0.80, lambda F: ramp(_solar_adj_energy_ratio(F), 1.0, 0.3),
         "Electricity bought is only {energy_ratio:.2f}x the minimum needed for the tonnes claimed "
         "({kwh_per_t:,.0f} kWh/t against at least {kwh_min_per_t:,.0f})"),
    Rule("energy_low_months", "energy", 0.45, lambda F: ramp(F["energy_low_month_share"], 0.05, 0.5),
         "In {energy_low_month_share:.0%} of billed months the units consumed cannot support the tonnes claimed"),
    Rule("missing_bills", "energy", 0.35, lambda F: ramp(F["bill_coverage"], 0.85, 0.4),
         "Electricity bills exist for only {bill_coverage:.0%} of active months"),
    Rule("doctored_bills", "bills", 0.85, lambda F: np.maximum(ramp(F["bill_inconsistent_share"], 0.0, 0.3),
                                                               ramp(F["bill_energy_ratio_min"], 0.99, 0.85)),
         "Energy charges do not match units x tariff on {bill_inconsistent_share:.0%} of bills (worst bill charged {bill_energy_ratio_min:.0%} of expected)"),
    Rule("below_cost_certificates", "price", 0.60, lambda F: ramp(F["price_below_floor_share"], 0.3, 1.0),
         "{price_below_floor_share:.0%} of certificate tonnes were sold below the cost of recycling (average {price_to_floor:.2f}x the cost floor)"),
    Rule("underpriced_vs_market", "price", 0.50, lambda F: ramp(F["price_vs_market"], 0.85, 0.4),
         "Certificates sold at {price_vs_market:.2f}x the market median price"),
    Rule("round_trip_ring", "network", 0.95, lambda F: (F["net_in_cycle"].fillna(0) > 0).astype(float),
         "Buys feedstock inside a closed loop of {net_cycle_len:.0f} recyclers trading the same material ({net_cycle_feed_share:.0%} of its feedstock)"),
    Rule("round_tonnage_invoices", "feedstock", 0.70, lambda F: ramp(F["feed_round_share"], 0.05, 0.3),
         "{feed_round_share:.0%} of purchase invoices have suspiciously round tonnages (typical plant: about 2%)"),
    Rule("weekend_invoices", "feedstock", 0.40, lambda F: ramp(F["feed_sunday_share"], 0.04, 0.10),
         "{feed_sunday_share:.1%} of purchase invoices are dated on Sundays (typical: 1.4%)"),
    Rule("no_eway_bills", "feedstock", 0.60, lambda F: ramp(F["feed_eway_missing"], 0.10, 0.4),
         "{feed_eway_missing:.0%} of high-value consignments have no e-way bill"),
    Rule("benford_feedstock", "benford", 0.30, lambda F: ramp(F["benford_feed_mad"], 0.05, 0.10),
         "First-digit pattern of invoice quantities departs from Benford's law (MAD {benford_feed_mad:.3f})"),
    Rule("collectors_over_capacity", "collectors", 0.80, lambda F: ramp(F["feed_collector_over_cap"], 0.1, 0.6),
         "{feed_collector_over_cap:.0%} of feedstock comes from collectors invoicing far beyond their declared capacity"),
    Rule("collectors_brand_new", "collectors", 0.40, lambda F: ramp(F["feed_collector_young"], 0.4, 0.8),
         "{feed_collector_young:.0%} of feedstock comes from collectors registered under 6 months before their first invoice"),
    Rule("collectors_single_client", "collectors", 0.30, lambda F: ramp(F["feed_collector_single_buyer"], 0.6, 0.9),
         "{feed_collector_single_buyer:.0%} of feedstock comes from collectors who sell to this plant only"),
    Rule("vehicle_reuse", "collectors", 0.40, lambda F: ramp(F["feed_vehicle_multi"], 0.25, 0.5),
         "{feed_vehicle_multi:.0%} of invoices use a vehicle that also appears on another plant's invoice the same day"),
    Rule("category_3_inflation", "cert", 0.70, lambda F: ramp(F["cert_cat3_excess"], 0.10, 0.4),
         "Category III share of certificates is {cert_share_CAT_III:.0%}, {cert_cat3_excess:+.0%} versus what its process route can produce"),
    Rule("deadline_burst", "cert", 0.60, lambda F: ramp(F["cert_gen_top_month_share"], 0.15, 0.5),
         "{cert_gen_top_month_share:.0%} of all certificates were generated in a single month"),
    Rule("late_batching", "cert", 0.50, lambda F: ramp(F["cert_late_batch_share"], 0.1, 0.6),
         "{cert_late_batch_share:.0%} of certificate tonnes were generated more than 45 days after the processing month"),
    Rule("output_shortfall", "output", 0.70, lambda F: ramp(F["out_to_expected"], 0.8, 0.2),
         "Sold products equal only {out_to_expected:.0%} of the output expected from the tonnes claimed"),
    Rule("output_low_months", "output", 0.50, lambda F: ramp(F["out_month_low_share"], 0.1, 0.6),
         "In {out_month_low_share:.0%} of months, product sales fall far short of the claimed input"),
    Rule("shell_offtakers", "output", 0.50, lambda F: ramp(F["out_young_offtaker_share"], 0.3, 0.8),
         "{out_young_offtaker_share:.0%} of product sales go to buyers incorporated under a year before the invoice"),
    Rule("duplicate_plant_photos", "identity", 0.70, lambda F: (F["id_photo_near_dups"].fillna(0) > 0).astype(float),
         "Uploaded plant photos are near-duplicates of photos registered by {id_photo_near_dups:.0f} other plant(s)"),
    Rule("shared_bank_account", "identity", 0.50, lambda F: ramp(F["id_shared_bank"], 0, 1),
         "Shares a bank account with {id_shared_bank:.0f} other registered entity(ies)"),
    Rule("shared_contacts", "identity", 0.30, lambda F: ramp(F["id_shared_phone"], 0, 2),
         "Shares a contact phone number with {id_shared_phone:.0f} other registered entity(ies)"),
    Rule("shared_address", "identity", 0.30, lambda F: ramp(F["id_shared_address"], 0, 1),
         "Registered at an address shared with {id_shared_address:.0f} other entity(ies)"),
    Rule("shared_director", "identity", 0.20, lambda F: ramp(F["id_shared_director"], 0, 3),
         "A director also sits on {id_shared_director:.0f} other registered entities"),
    Rule("very_young_company", "registry", 0.30, lambda F: ramp(F["age_at_registration_days"], 365, 60),
         "Company was only {age_at_registration_days:.0f} days old when it registered under EPR"),
]


def rule_matrix(F: pd.DataFrame, rules: list[Rule] = RULES, exclude_groups: set[str] | None = None) -> pd.DataFrame:
    exclude_groups = exclude_groups or set()
    cols = {r.name: r.fn(F) * r.weight for r in rules if r.group not in exclude_groups}
    return pd.DataFrame(cols, index=F.index)


def rule_scores(F: pd.DataFrame, exclude_groups: set[str] | None = None) -> pd.Series:
    M = rule_matrix(F, exclude_groups=exclude_groups)
    return (1 - (1 - M).prod(axis=1)).rename("rule_score")


def rule_reasons(F: pd.DataFrame, entity_id: str, top: int = 5, min_contrib: float = 0.15) -> list[dict]:
    M = rule_matrix(F.loc[[entity_id]])
    row = F.loc[entity_id]
    by_name = {r.name: r for r in RULES}
    out = []
    for name, c in M.iloc[0].sort_values(ascending=False).items():
        if c < min_contrib or len(out) >= top:
            break
        r = by_name[name]
        try:
            text = r.reason.format(**{k: (float(v) if pd.notna(v) else float("nan")) for k, v in row.items()})
        except Exception:
            text = r.name
        out.append({"rule": name, "group": r.group, "strength": round(float(c), 3), "text": text})
    return out
