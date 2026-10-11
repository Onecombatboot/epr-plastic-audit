"""Conservative ceilings: inventory windows preserve lag and absent bills stay unknown."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .prep import monthly_frame

GROUPS = {"energy": ["energy_deficit"], "output": ["output_deficit"],
          "capacity": ["capacity_deficit", "load_deficit"], "bill": ["bill_error"],
          "drift": ["energy_drift", "output_drift", "mix_deviation", "price_deviation"]}


def rolling(frame: pd.DataFrame, column: str, window: int, minimum: int = 1) -> pd.Series:
    return frame.groupby("entity_id", sort=False)[column].transform(
        lambda x: x.rolling(window, min_periods=minimum).sum())


def shortfall(claimed: pd.Series, supported: pd.Series) -> pd.Series:
    return (1 - supported.div(claimed.replace(0, np.nan))).clip(0, 1)


def compute(tables: dict[str, pd.DataFrame], ref: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = monthly_frame(tables)
    claim = m.certificates_t
    rate = m.energy_rate_inr_per_kwh
    m["bill_present"] = m.bill_id.notna()
    expected_charge = m.units_kwh * rate
    m["bill_error"] = (m.energy_charge_inr-expected_charge).abs().div(expected_charge.replace(0, np.nan))
    m.loc[m.units_kwh.eq(0) & m.energy_charge_inr.eq(0) & rate.notna(), "bill_error"] = 0
    m["bill_testable"] = m.bill_present & rate.notna()
    # A mismatching rupee amount supplies an alternate ceiling, never proof of true consumption.
    m["charge_implied_kwh"] = m.energy_charge_inr.div(rate)
    m["effective_kwh"] = np.where(m.bill_testable, np.minimum(m.units_kwh, m.charge_implied_kwh), m.units_kwh)
    solar = m.captive_solar_kw.fillna(0).clip(lower=0) * ref["solar_hours_per_month"] * ref["solar_conversion_efficiency"]
    residual = (m.effective_kwh + solar - m.input_t_reported*m.kwh_min).clip(lower=0)
    m["quiet_residual"] = residual.where(m.active & m.bill_present)
    base = m.groupby("entity_id").quiet_residual.transform(lambda x: x.quantile(ref["base_load_quantile"]))
    cap = m.groupby("entity_id").effective_kwh.transform("median") * ref["base_load_cap_fraction"]
    m["base_load_kwh"] = np.minimum(base.fillna(0), cap.fillna(0))
    m["solar_credit_kwh"] = solar
    m["energy_supported_t"] = (m.effective_kwh + solar - m.base_load_kwh).clip(lower=0) / m.kwh_min
    m["capacity_supported_t"] = m.declared_capacity_tpa / 12
    m["load_supported_t"] = m.connected_load_kw * ref["hours_per_month"] * ref["max_load_utilisation"] / m.kwh_min
    m["capacity_ratio"] = claim / m.capacity_supported_t.replace(0, np.nan)
    m["capacity_deficit"] = shortfall(claim, m.capacity_supported_t)
    m["load_deficit"] = shortfall(claim, m.load_supported_t)
    m["energy_deficit"] = shortfall(claim, m.energy_supported_t)
    m["grid_kwh_per_t"] = m.effective_kwh.div(claim.replace(0, np.nan))

    sales = tables["outputs_monthly"].merge(m[["entity_id", "month", "stream", "process_type"]],
                                            on=["entity_id", "month"], validate="many_to_one")
    sales = sales.merge(tables["products"], on=["stream", "process_type", "product"], how="left", validate="many_to_one")
    if sales.expected.isna().any():
        raise ValueError("An observed output product has no saleable configuration")
    grouped = sales.groupby(["entity_id", "month"]).tonnes_sold.sum()
    m["sold_t"] = pd.MultiIndex.from_frame(m[["entity_id", "month"]]).map(grouped).fillna(0).to_numpy()
    lag = ref["sales_lag_months"]
    for window in ref["output_windows"]:
        minimum = min(window, ref["minimum_window_months"])
        m[f"claimed_{window}m"] = rolling(m, "certificates_t", window, minimum)
        sold = rolling(m, "sold_t", window+lag, minimum+lag)
        sold = sold.groupby(m.entity_id, sort=False).shift(-lag)
        m[f"sold_{window}m_lag"] = sold
        m[f"output_support_{window}m"] = sold.div(m.saleable_lower)
        m[f"output_ratio_{window}m"] = sold.div(m[f"claimed_{window}m"].replace(0, np.nan))
        m[f"output_deficit_{window}m"] = shortfall(m[f"claimed_{window}m"], m[f"output_support_{window}m"])
    window = max(ref["output_windows"])
    m["output_deficit"] = m[f"output_deficit_{window}m"]
    m["output_supported_t"] = claim * (1-m.output_deficit)

    # Logistic-normal residual, centred on configured saleable shares; small missing products stay visible.
    mix = sales.groupby(["entity_id", "product"], as_index=False).agg(sold=("tonnes_sold", "sum"),
                                                                       value=("value_inr", "sum"))
    combinations = m[["entity_id", "stream", "process_type"]].drop_duplicates().merge(
        tables["products"], on=["stream", "process_type"], validate="many_to_many")
    mix = combinations.merge(mix, on=["entity_id", "product"], how="left", validate="one_to_one")
    mix[["sold", "value"]] = mix[["sold", "value"]].fillna(0)
    totals = mix.groupby("entity_id").sold.transform("sum")
    expected_total = mix.groupby("entity_id").expected.transform("sum")
    eps = 1e-6
    log_residual = np.log((mix.sold/totals.replace(0, np.nan)+eps)/(mix.expected/expected_total+eps))
    log_residual -= log_residual.groupby(mix.entity_id).transform("mean")
    scale = (mix.tolerance / mix.expected.clip(lower=eps)).clip(lower=0.05)
    mix["mix_residual"] = (log_residual/scale).pow(2)
    mix["price"] = mix.value.div(mix.sold.replace(0, np.nan))
    peer = mix.groupby(["stream", "process_type", "product"])
    med = peer.price.transform("median")
    deviation = (np.log1p(mix.price)-np.log1p(med)).abs()
    mix["price_deviation"] = deviation
    mix_stats = mix.groupby("entity_id").agg(mix_deviation=("mix_residual", "mean"),
                                              price_deviation=("price_deviation", "mean"))
    m = m.join(mix_stats, on="entity_id")

    # An early observed baseline can miss persistent fraud; it is only an auxiliary change signal.
    m["age_months"] = m.active.groupby(m.entity_id).cumsum()
    first = m.active & m.age_months.le(ref["baseline_min_months"])
    energy_baseline = m.grid_kwh_per_t.where(first).groupby(m.entity_id).transform("median")
    out_baseline = m[f"output_ratio_{window}m"].where(first).groupby(m.entity_id).transform("median")
    m["energy_drift"] = (1-m.grid_kwh_per_t.div(energy_baseline.replace(0, np.nan))).clip(lower=0)
    m["output_drift"] = (1-m[f"output_ratio_{window}m"].div(out_baseline.replace(0, np.nan))).clip(lower=0)
    m.loc[first, ["energy_drift", "output_drift"]] = 0
    supports = m[["energy_supported_t", "output_supported_t", "capacity_supported_t", "load_supported_t"]]
    m["supported_t"] = np.minimum(claim, supports.min(axis=1, skipna=True))
    m["suspected_hollow_t"] = (claim-m.supported_t).clip(lower=0)
    m["checks_testable"] = supports.notna().sum(axis=1)
    m["month_flag"] = m.suspected_hollow_t.gt(claim*ref["priority_min_deficit"]) & claim.gt(0)
    m = m[m.operation_present].copy()
    groups = m.groupby("entity_id", sort=False)
    features = groups.agg(stream=("stream", "first"), process_type=("process_type", "first"), state=("state", "first"),
                          legal_name=("legal_name", "first"), months=("month", "size"), claimed_t=("certificates_t", "sum"),
                          suspected_hollow_t=("suspected_hollow_t", "sum"), bill_coverage=("bill_present", "mean"),
                          energy_supported_t=("energy_supported_t", "sum"), sold_t=("sold_t", "sum"),
                          saleable_expected=("saleable_expected", "first"), saleable_lower=("saleable_lower", "first"),
                          mix_deviation=("mix_deviation", "first"), price_deviation=("price_deviation", "first"))
    for name in ["energy_deficit", "output_deficit", "capacity_deficit", "load_deficit", "bill_error", "energy_drift", "output_drift"]:
        features[name] = groups[name].mean()
        features[name+"_max"] = groups[name].max()
    features["hollow_fraction"] = features.suspected_hollow_t.div(features.claimed_t.replace(0, np.nan)).fillna(0)
    features["output_coverage"] = features.sold_t.div(features.claimed_t.replace(0, np.nan))
    observed_claim = m.certificates_t.where(m.bill_present).groupby(m.entity_id).sum()
    features["energy_pressure"] = observed_claim.div(features.energy_supported_t.replace(0, np.nan))
    features["output_pressure"] = features.saleable_expected.div(features.output_coverage.replace(0, np.nan))
    features["capacity_pressure"] = groups.capacity_ratio.quantile(0.90)
    features["missing_bill_months"] = groups.bill_present.agg(lambda x: int((~x).sum()))
    return m.reset_index(drop=True), features


def what_if(months: pd.DataFrame, fraction: float) -> dict:
    """Hold all evidence fixed and proportionally lower claims to their observable ceilings."""
    if not 0 <= fraction <= 1:
        raise ValueError("Claim fraction must be in [0, 1]")
    claimed = months.certificates_t * fraction
    excess = (claimed-months.supported_t).clip(lower=0)
    positive = months.certificates_t.gt(0)
    safe = months.supported_t[positive].div(months.certificates_t[positive]).min() if positive.any() else 1.0
    return {"claimed_t": float(claimed.sum()), "unexplained_t": float(excess.sum()),
            "safe_fraction": float(np.clip(safe, 0, 1)), "tonnage_checks_pass": bool(excess.le(1e-7).all()),
            "bill_checks_pass": bool(months.loc[months.bill_testable, "bill_error"].le(0.03).all()),
            "unknown_energy_months": int((~months.bill_present).sum())}


def product_balance(months: pd.DataFrame, outputs: pd.DataFrame, products: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """Product-specific expectations make mix gaps reviewable without treating every yield prior as a hard limit."""
    window, lag = max(ref["output_windows"]), ref["sales_lag_months"]
    columns = ["entity_id", "month", "stream", "process_type", f"claimed_{window}m"]
    balance = months[columns].merge(products, on=["stream", "process_type"], validate="many_to_many")
    sales = outputs.groupby(["entity_id", "month", "product"], as_index=False).tonnes_sold.sum()
    balance = balance.merge(sales, on=["entity_id", "month", "product"], how="left", validate="many_to_one")
    balance["tonnes_sold"] = balance.tonnes_sold.fillna(0)
    balance = balance.sort_values(["entity_id", "product", "month"]).reset_index(drop=True)
    group = balance.groupby(["entity_id", "product"], sort=False)
    cumulative = group.tonnes_sold.transform(lambda x: x.rolling(window+lag, min_periods=ref["minimum_window_months"]+lag).sum())
    balance["sold_window_t"] = cumulative.groupby([balance.entity_id, balance["product"]]).shift(-lag)
    claim = balance[f"claimed_{window}m"]
    balance["expected_product_t"] = claim*balance.expected
    balance["lower_product_t"] = claim*balance.lower
    balance["shortfall_t"] = (balance.lower_product_t-balance.sold_window_t).clip(lower=0)
    return balance[["entity_id", "month", "product", "expected_product_t", "lower_product_t", "sold_window_t", "shortfall_t"]]
