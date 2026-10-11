"""Direction-aware residual ranks complement physical findings without label fitting."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURES = {"energy": ["energy_pressure"], "output": ["output_pressure"],
            "capacity": ["capacity_pressure", "load_deficit"], "bill": ["bill_error"],
            "drift": ["energy_drift", "output_drift", "mix_deviation", "price_deviation"]}


def score(features: pd.DataFrame, ref: dict, include: set[str] | None = None,
          explain: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = features.copy()
    include = set(FEATURES) if include is None else include
    if not include or not include.issubset(FEATURES):
        raise ValueError("Scoring requires known evidence groups")
    suspicion, residuals, ranks = pd.DataFrame(index=result.index), pd.DataFrame(index=result.index), {}
    cohort = [result.stream, result.process_type]
    for group in sorted(include):
        cols = FEATURES[group]
        for col in cols:
            x = result[col].replace([np.inf, -np.inf], np.nan)
            med = x.groupby(cohort).transform("median")
            mad = (x-med).abs().groupby(cohort).transform("median")
            q25 = x.groupby(cohort).transform(lambda s: s.quantile(0.25))
            q75 = x.groupby(cohort).transform(lambda s: s.quantile(0.75))
            scale = np.maximum(mad*1.4826, (q75-q25)/1.349)
            scale = np.maximum(scale.fillna(0), 0.01)
            residuals[col] = ((x-med)/scale).clip(lower=0).fillna(0)
            pct = x.groupby(cohort).rank(method="average", pct=True)
            suspicion[col] = ((pct-0.5)*2).clip(0, 1).fillna(0)
        ranks[group] = suspicion[cols].mean(axis=1)
        result["rank_"+group] = ranks[group]
    weights = pd.Series({g: ref["weights"][g] for g in sorted(include)})
    rank_score = pd.DataFrame(ranks).mul(weights).sum(axis=1)/weights.sum()
    robust = residuals.clip(upper=10).mean(axis=1)
    robust_rank = robust.rank(method="average", pct=True)*suspicion.max(axis=1)
    model = IsolationForest(n_estimators=ref["iforest_trees"], max_samples=min(512, len(result)),
                            random_state=ref["iforest_seed"], n_jobs=-1)
    model.fit(suspicion.to_numpy())
    raw = pd.Series(-model.score_samples(suspicion.to_numpy()), index=result.index)
    forest_rank = raw.rank(method="average", pct=True)*suspicion.max(axis=1)
    result["robust_score"] = robust
    result["iforest_score"] = raw
    result["score"] = (0.60*rank_score + 0.25*robust_rank + 0.15*forest_rank).clip(0, 1)
    strength = pd.DataFrame({"energy": result.energy_deficit, "output": result.output_deficit,
                             "capacity": result[["capacity_deficit", "load_deficit"]].max(axis=1),
                             "bill": result.bill_error}).fillna(0)
    corroborating = (strength[["energy", "output", "capacity"]] >= ref["priority_min_deficit"]).sum(axis=1)
    priority = corroborating.ge(ref["priority_min_groups"]) | strength[["energy", "output", "capacity"]].max(axis=1).ge(0.50)
    priority |= result.bill_error.ge(2*ref["bill_relative_tolerance"]) & result.months.ge(3)
    screening = strength[["energy", "output", "capacity"]].max(axis=1).ge(ref["screening_min_deficit"])
    screening |= result.bill_error.ge(ref["bill_relative_tolerance"])
    result["priority"] = priority
    result["screening"] = screening | priority
    result["tier"] = np.select([priority, screening], ["priority", "screening"], default="below_threshold")
    result = result.sort_values(["score", "suspected_hollow_t"], ascending=False, kind="mergesort")
    result["rank"] = np.arange(1, len(result)+1)
    shap_values = pd.DataFrame(index=features.index)
    if explain:
        import shap
        shap_values = pd.DataFrame(shap.TreeExplainer(model).shap_values(suspicion.to_numpy(), check_additivity=False),
                                   index=features.index, columns=suspicion.columns)
    return result, shap_values


def reasons(row: pd.Series, months: pd.DataFrame) -> list[str]:
    def percent(value: float) -> str:
        return f"{value:.1%}" if pd.notna(value) and np.isfinite(value) else "unavailable"

    candidates = [
        (float(row.get("rank_energy", 0)), f"Electricity evidence covers {row.bill_coverage:.1%} of {int(row.months)} months; mean unsupported claim share is {percent(row.energy_deficit)} at the configured minimum kWh/t, after declared solar and base load."),
        (float(row.get("rank_output", 0)), f"Sold outputs cover {percent(row.output_coverage)} of claimed input across {int(row.months)} months; configured saleable share is {row.saleable_lower:.1%}–{row.saleable_expected:.1%}; lag-adjusted 6-month shortfall is {percent(row.output_deficit)} on average."),
        (float(row.get("rank_capacity", 0)), f"Certificates exceed monthly capacity support by {row.capacity_deficit:.1%} on average and connected-load support by {row.load_deficit:.1%}, using 730 hours at 90% utilisation."),
        (float(row.get("rank_bill", 0)), f"Energy charges differ from units × state/FY tariff by {percent(row.bill_error)} on average; {int(row.missing_bill_months)} months lack an electricity bill."),
        (float(row.get("rank_drift", 0)), f"Against the first 6 active months, the mean fall in electricity per claimed tonne is {percent(row.energy_drift)} and in output coverage is {percent(row.output_drift)}.")]
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in candidates[:3]]


def signals(scores: pd.DataFrame, months: pd.DataFrame, shap_values: pd.DataFrame) -> list[dict]:
    indexed = {entity: group for entity, group in months.groupby("entity_id", sort=False)}
    output = []
    for entity, row in scores.iterrows():
        if not row.screening:
            continue
        observed = indexed[entity]
        worst = observed.sort_values("suspected_hollow_t", ascending=False)
        evidence = [f"operations_monthly:{entity}:{worst.iloc[0].month:%Y-%m}"]
        bills = observed[observed.bill_present].sort_values("bill_error", ascending=False).bill_id.head(2).tolist()
        evidence.extend(str(b) for b in bills)
        sold = observed[observed.sold_t.gt(0)]
        if len(sold):
            evidence.append(f"outputs_monthly:{entity}:{sold.iloc[0].month:%Y-%m}")
        contributions = []
        if entity in shap_values.index:
            contributions = [{"feature": col, "value": round(float(row[col]), 5),
                              "shap": round(float(value), 5)}
                             for col, value in shap_values.loc[entity].sort_values().head(3).items()
                             if value < 0 and pd.notna(row[col])]
        output.append({"entity_id": entity, "entity_kind": "recycler", "agent": "mass_balance_yield",
                       "stream": row.stream, "state": row.state,
                       "score": float(row.score), "tier": row.tier, "reasons": reasons(row, observed),
                       "evidence_refs": evidence, "suspected_hollow_t": float(row.suspected_hollow_t),
                       "model_contributions": contributions})
    return output
