"""Layer 2: Isolation Forest on direction-aware suspicion features, with SHAP explanations.

Why direction-aware: a plain Isolation Forest flags *rare* values in either direction, so an unusually
HIGH certificate price would look as suspicious as an unusually LOW one. Each feature is therefore
converted to a one-sided "suspicion" level in [0, 1] using prior domain knowledge about which direction
is worse (DIRECTION below; never learned from labels):

    +1  higher is more suspicious      -1  lower is more suspicious      0  either extreme is unusual
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .features import GROUPS

DIRECTION: dict[str, int] = {
    # capacity
    "cap_ratio_fy_max": 1, "cap_ratio_month_max": 1, "cap_ratio_month_p90": 1, "over_cap_month_share": 1, "reported_cv": 1,
    "log_growth": 1, "cap_per_kw": 1, "t_per_staff": 1,
    # energy
    "energy_ratio": -1, "energy_ratio_p10": -1, "energy_ratio_median": -1, "energy_low_month_share": 1, "kwh_per_t": -1,
    "load_util_max": 1, "load_util_p90": 1, "cap_vs_load": 1, "bill_coverage": -1, "bill_units_cv": 1,
    # bills
    "bill_energy_ratio_mean": -1, "bill_energy_ratio_min": -1, "bill_inconsistent_share": 1,
    # feedstock
    "feed_round_share": 1, "feed_sunday_share": 1, "feed_monthend_share": 1, "feed_far_share": 1, "feed_rate_dev": 1,
    "feed_eway_missing": 1, "feed_inv_per_t": 0, "feed_mean_inv_t": 0, "feed_cov_total": 0, "feed_cov_median": 0,
    "feed_cov_tight_share": 1, "feed_cov_std": -1,
    # collectors
    "feed_collector_over_cap": 1, "feed_collector_young": 1, "feed_collector_single_buyer": 1, "feed_vehicle_multi": 1,
    "feed_collector_hhi": 0, "feed_collector_top1": 0, "feed_n_collectors": 0, "feed_from_recyclers": 1,
    # certificates
    "cert_share_CAT_I": -1, "cert_share_CAT_III": 1, "cert_cat3_excess": 1, "cert_cat1_deficit": 1, "cert_vs_reported": 0,
    "cert_late_batch_share": 1, "cert_lag_mean": 1, "cert_gen_top_month_share": 1, "cert_gen_hhi": 1, "cert_june_share": 1,
    # price
    "price_to_floor": -1, "price_vs_market": -1, "price_below_floor_share": 1, "price_p10_vs_market": -1, "sell_through": 0,
    "sales_n_brands": 0, "sales_brand_hhi": 0, "sales_brand_top1": 0, "sales_fyend_share": 0,
    # output
    "out_to_expected": -1, "out_young_offtaker_share": 1, "out_thin_offtaker_share": 1, "out_round_share": 1,
    "out_monthend_share": 1, "out_rate_dev_mean": 0, "out_n_offtakers": 0, "out_top1": 0, "out_month_ratio_p25": -1,
    "out_month_low_share": 1,
    # network
    "net_in_cycle": 1, "net_cycle_feed_share": 1, "net_recycler_in_degree": 1, "net_recycler_out_degree": 1,
    # benford
    "benford_feed_mad": 1, "benford_out_mad": 1, "benford_reported_mad": 1,
    # registry
    "age_at_registration_days": -1, "first_cert_before_reg_days": 1,
    # identity
    "id_shared_phone": 1, "id_shared_bank": 1, "id_shared_address": 1, "id_shared_signatory": 1, "id_shared_director": 1,
    "id_photo_near_dups": 1,
}

FEATURE_TEXT: dict[str, str] = {
    "energy_ratio": "electricity bought vs minimum needed for claimed tonnes", "energy_ratio_p10": "worst-month electricity vs needed",
    "energy_ratio_median": "typical-month electricity vs needed", "kwh_per_t": "kWh used per claimed tonne",
    "load_util_max": "share of connected load the claimed tonnage would need", "load_util_p90": "connected-load utilisation implied (p90)",
    "cap_vs_load": "power needed for declared capacity vs connected load", "over_cap_month_share": "months above declared capacity",
    "price_to_floor": "certificate price vs cost of recycling", "price_vs_market": "certificate price vs market", "price_p10_vs_market": "cheapest sales vs market",
    "price_below_floor_share": "certificates sold below cost", "cert_gen_top_month_share": "certificates generated in one month",
    "cert_gen_hhi": "concentration of certificate generation in time", "cert_late_batch_share": "certificates batched late",
    "cert_cat3_excess": "Category III share above what the process can produce", "cert_share_CAT_III": "Category III share",
    "feed_round_share": "round-tonnage purchase invoices", "feed_sunday_share": "Sunday-dated invoices", "feed_eway_missing": "high-value invoices without e-way bill",
    "feed_collector_over_cap": "feedstock from over-capacity collectors", "feed_collector_young": "feedstock from brand-new collectors",
    "feed_collector_single_buyer": "feedstock from single-client collectors", "feed_vehicle_multi": "vehicles reused across plants the same day",
    "out_to_expected": "sold products vs expected output", "out_month_low_share": "months with output far below input",
    "out_young_offtaker_share": "product sales to newly formed buyers", "out_thin_offtaker_share": "product sales to buyers with few suppliers",
    "bill_inconsistent_share": "bills whose charges do not match units x tariff", "bill_energy_ratio_min": "worst bill charge vs units x tariff",
    "bill_energy_ratio_mean": "average bill charge vs units x tariff", "bill_coverage": "months with an electricity bill",
    "net_in_cycle": "membership of a closed trading loop", "net_cycle_feed_share": "feedstock bought inside a loop",
    "id_photo_near_dups": "plant photos duplicated at other plants", "id_shared_bank": "shared bank account", "id_shared_phone": "shared contact phone",
    "id_shared_address": "shared address", "id_shared_director": "shared director", "id_shared_signatory": "shared signatory",
    "benford_feed_mad": "Benford deviation of invoice quantities", "reported_cv": "month-to-month volatility of reported tonnes",
    "feed_cov_tight_share": "months where invoices exactly cover claims", "age_at_registration_days": "company age at EPR registration",
}


def feature_columns(exclude_groups: set[str] | None = None) -> list[str]:
    exclude_groups = exclude_groups or set()
    return [c for g, cols in GROUPS.items() if g not in exclude_groups for c in cols if c in DIRECTION]


def suspicion_matrix(F: pd.DataFrame, cols: list[str], ref_F: pd.DataFrame | None = None, direction: dict | None = None) -> pd.DataFrame:
    """Map each feature to a [0,1] one-sided suspicion level via its percentile in the reference population.

    Percentiles are taken against `ref_F` (default: F itself), so a new batch can be scored on the same scale
    as the population the model was fit on.
    """
    ref_F = F if ref_F is None else ref_F
    direction = DIRECTION if direction is None else direction
    U = pd.DataFrame(index=F.index)
    for c in cols:
        d = direction[c]
        x = F[c].astype(float)
        r = ref_F[c].astype(float).dropna().to_numpy()
        if len(r) == 0:
            U[c] = 0.0
            continue
        r = np.sort(r)
        lo = np.searchsorted(r, x.fillna(np.nanmedian(r)).to_numpy(), side="left") / len(r)
        hi = np.searchsorted(r, x.fillna(np.nanmedian(r)).to_numpy(), side="right") / len(r)
        pct = (lo + hi) / 2.0
        if d > 0:
            u = np.clip((pct - 0.5) * 2.0, 0, 1)
        elif d < 0:
            u = np.clip((0.5 - pct) * 2.0, 0, 1)
        else:
            u = np.abs(pct - 0.5) * 2.0
        U[c] = u
    return U


class IForestDetector:
    def __init__(self, n_estimators: int = 600, max_samples: int | str = 512, seed: int = 0, exclude_groups: set[str] | None = None):
        self.cols = feature_columns(exclude_groups)
        self.model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination="auto", random_state=seed, n_jobs=-1)

    def fit(self, F: pd.DataFrame) -> "IForestDetector":
        self.ref = F[self.cols].copy()
        self.model.fit(suspicion_matrix(F, self.cols, self.ref).to_numpy())
        return self

    def score(self, F: pd.DataFrame) -> pd.Series:
        U = suspicion_matrix(F, self.cols, self.ref)
        s = -self.model.score_samples(U.to_numpy())
        return pd.Series(s, index=F.index, name="iforest_score")

    def shap_values(self, F: pd.DataFrame) -> pd.DataFrame:
        import shap
        U = suspicion_matrix(F, self.cols, self.ref)
        ex = shap.TreeExplainer(self.model)
        sv = ex.shap_values(U.to_numpy(), check_additivity=False)
        return pd.DataFrame(sv, index=F.index, columns=self.cols)

    def reasons(self, F: pd.DataFrame, sv: pd.DataFrame, entity_id: str, top: int = 4) -> list[dict]:
        row = F.loc[entity_id]
        s = sv.loc[entity_id]
        # In sklearn's convention higher score_samples = more normal, so a NEGATIVE shap value pushes toward anomalous.
        order = s.sort_values().index[:top]
        out = []
        for c in order:
            if s[c] >= 0:
                break
            pct = float((self.ref[c].astype(float) < row[c]).mean()) if pd.notna(row[c]) else float("nan")
            out.append({"feature": c, "label": FEATURE_TEXT.get(c, c), "value": None if pd.isna(row[c]) else round(float(row[c]), 4),
                        "population_percentile": round(pct * 100, 1), "shap": round(float(s[c]), 4)})
        return out


def generic_iforest_score(X: pd.DataFrame, direction: dict, seed: int = 0, n_estimators: int = 500) -> pd.Series:
    """Isolation Forest on direction-aware suspicion levels for any entity table (collectors, offtakers, ...)."""
    cols = list(direction)
    U = suspicion_matrix(X, cols, direction=direction)
    m = IsolationForest(n_estimators=n_estimators, max_samples=512, random_state=seed, n_jobs=-1).fit(U.to_numpy())
    return pd.Series(-m.score_samples(U.to_numpy()), index=X.index)


def rank01(s: pd.Series) -> pd.Series:
    return s.rank(method="average", pct=True)
