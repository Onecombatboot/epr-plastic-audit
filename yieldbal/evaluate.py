"""Ground-truth access is confined here; evaluation never feeds thresholds or features."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .checks import compute
from .prep import DEFAULT_DATA, DEFAULT_RUN, load, load_reference
from .scoring import FEATURES, score

TIERS = ("obvious", "moderate", "subtle")


def auc(values: pd.Series, truth: pd.Series) -> float | None:
    valid = values.notna() & np.isfinite(values)
    return float(roc_auc_score(truth[valid], values[valid])) if truth[valid].nunique() == 2 else None


def population(scores: pd.DataFrame, labels: pd.DataFrame, hollow: pd.Series) -> dict:
    y = labels.is_fraud.reindex(scores.index).astype(bool)
    difficulty = labels.difficulty.reindex(scores.index)
    metric = {"entities": len(scores), "fraudsters": int(y.sum()), "auc": auc(scores.score, y),
              "precision_at_k": {str(k): float(y.iloc[:min(k, len(y))].mean()) for k in (10, 25, 50)},
              "auc_by_tier": {}, "policies": {}, "hollow_tonnes_recall_at_k": {}}
    for tier in TIERS:
        mask = ~y | difficulty.eq(tier)
        metric["auc_by_tier"][tier] = auc(scores.score[mask], y[mask])
    for policy in ("priority", "screening"):
        flagged = scores[policy]
        metric["policies"][policy] = {"flagged": int(flagged.sum()),
            "precision": float(y[flagged].mean()) if flagged.any() else None,
            "recall_by_tier": {t: float(flagged[y & difficulty.eq(t)].mean()) if (y & difficulty.eq(t)).any() else None for t in TIERS}}
    total = float(hollow.reindex(scores.index).sum())
    for k in (10, 25, 50, 100, 250):
        metric["hollow_tonnes_recall_at_k"][str(k)] = float(hollow.reindex(scores.index[:k]).sum()/total) if total else None
    patterns = {"lazy": y & ~labels.output_fabrication.reindex(scores.index),
                "output_fabrication": y & labels.output_fabrication.reindex(scores.index),
                "doctored_bill": y & labels.doctored_bill.reindex(scores.index),
                "capacity_inflated": y & labels.capacity_inflated.reindex(scores.index)}
    metric["auc_by_behaviour"] = {name: {"n": int(mask.sum()), "auc": auc(scores.score[~y | mask], y[~y | mask])}
                                   for name, mask in patterns.items()}
    metric["single_features"] = {}
    for col in ("energy_pressure", "output_pressure", "capacity_pressure", "bill_error"):
        metric["single_features"][col] = {"auc": auc(scores[col], y),
            "subtle_auc": auc(scores.loc[~y | difficulty.eq("subtle"), col], y[~y | difficulty.eq("subtle")])}
    return metric


def signal_validation(data: Path, run: Path) -> dict:
    signals = [json.loads(line) for line in (run/"signals.jsonl").read_text().splitlines()]
    operations = pd.read_parquet(data/"yield"/"operations_monthly.parquet")
    outputs = pd.read_parquet(data/"yield"/"outputs_monthly.parquet")
    bills = pd.read_parquet(data/"yield"/"electricity_bills.parquet")
    evidence = set(bills.bill_id)
    for table, frame in (("operations_monthly", operations), ("outputs_monthly", outputs)):
        evidence.update(table+":"+frame.entity_id+":"+pd.to_datetime(frame.month).dt.strftime("%Y-%m"))
    invalid = 0
    for signal in signals:
        valid = (signal["agent"] == "mass_balance_yield" and signal["entity_kind"] == "recycler"
                 and 0 <= signal["score"] <= 1 and 2 <= len(signal["reasons"]) <= 4
                 and all(re.search(r"\d", reason) and "nan" not in reason.lower() for reason in signal["reasons"])
                 and len(signal["evidence_refs"]) > 0 and all(ref in evidence for ref in signal["evidence_refs"]))
        invalid += not valid
    return {"count": len(signals), "invalid": int(invalid), "valid_rate": 1-invalid/len(signals) if signals else 1.0}


def evaluate(data: Path, run: Path, ablations: bool = True) -> dict:
    data, run = Path(data), Path(run)
    labels = pd.read_parquet(data/"labels"/"registry"/"entity_labels.parquet").set_index("entity_id")
    extra = pd.read_parquet(data/"labels"/"yield"/"entity_extra_labels.parquet").set_index("entity_id")
    truth = pd.read_parquet(data/"labels"/"yield"/"monthly_truth.parquet")
    truth["month"] = pd.to_datetime(truth.month)
    hollow = truth.groupby("entity_id").hollow_t.sum()
    scores = pd.read_parquet(run/"scores.parquet")
    derived_features = None
    if (run/"monthly_checks.parquet").exists():
        monthly = pd.read_parquet(run/"monthly_checks.parquet")
    else:
        monthly, derived_features = compute(load(data), load_reference())
    meta = json.loads((run/"run.json").read_text())
    res = {"synthetic": True, "pooled": population(scores, labels, hollow),
           "per_stream": {stream: population(group, labels, hollow) for stream, group in scores.groupby("stream", sort=True)},
           "runtime_s": meta["runtime_s"], "suspected_hollow_t": meta["suspected_hollow_t"],
           "signals": signal_validation(data, run)}
    observed = monthly[monthly.bill_present].copy()
    observed["minimum_kwh_claim"] = observed.certificates_t*observed.kwh_min
    raw_energy = observed.groupby("entity_id").minimum_kwh_claim.sum().div(
        observed.groupby("entity_id").units_kwh.sum().replace(0, np.nan)).reindex(scores.index)
    res["raw_grid_energy_baseline"] = {}
    for name, group in [("pooled", scores), *list(scores.groupby("stream", sort=True))]:
        y = labels.is_fraud.reindex(group.index)
        difficulty = labels.difficulty.reindex(group.index)
        values = raw_energy.reindex(group.index)
        res["raw_grid_energy_baseline"][name] = {"auc": auc(values, y),
            "auc_by_tier": {tier: auc(values[~y | difficulty.eq(tier)], y[~y | difficulty.eq(tier)]) for tier in TIERS}}
    clean = ~labels.is_fraud.reindex(scores.index)
    confounders = labels.confounders.reindex(scores.index).fillna("")
    masks = {"campaign": extra.campaign_processor.reindex(scores.index).fillna(False),
             "atypical_yield": extra.atypical_legit_yield.reindex(scores.index).fillna(False),
             "declared_solar": confounders.str.contains("captive_solar_declared"),
             "undeclared_solar": confounders.str.contains("captive_solar_undeclared"),
             "new_company": confounders.str.contains("new_legit_company"),
             "fast_growing": confounders.str.contains("fast_growing_legit")}
    masks["ordinary"] = ~pd.DataFrame(masks).any(axis=1)
    res["false_positives"] = {}
    for policy in ("priority", "screening"):
        res["false_positives"][policy] = {}
        for name, mask in masks.items():
            population_mask = clean & mask
            n = int(population_mask.sum())
            flagged = int(scores.loc[population_mask, policy].sum())
            res["false_positives"][policy][name] = {"n": n, "flagged": flagged, "rate": flagged/n if n else None}
    joined = monthly.merge(truth, on=["entity_id", "month"], validate="one_to_one")
    positive = joined.hollow_t.gt(0)
    flagged = joined.month_flag
    caught = int((flagged & positive).sum())
    res["monthly"] = {"positive_months": int(positive.sum()), "flagged": int(flagged.sum()),
                      "precision": caught/max(int(flagged.sum()), 1), "recall": caught/max(int(positive.sum()), 1),
                      "true_hollow_t": float(joined.hollow_t.sum()),
                      "estimated_hollow_t": float(joined.suspected_hollow_t.sum()),
                      "absolute_error_t": float((joined.suspected_hollow_t-joined.hollow_t).abs().sum())}
    raw = labels.is_fraud.reindex(scores.index)
    realised = hollow.reindex(scores.index).gt(0)
    capacity_labels = labels.capacity_inflated.reindex(scores.index)
    res["label_audit"] = {"raw_fraud": int(raw.sum()), "realised_hollow_entities": int((raw & realised).sum()),
                          "vacuous_hollow_labels": int((raw & ~realised).sum()),
                          "capacity_inflated_labels": int((raw & capacity_labels).sum()),
                          "capacity_label_with_no_declared_overrun": int((raw & capacity_labels & scores.capacity_deficit.eq(0)).sum()),
                          "note": "Capacity inflation does not imply declared capacity overrun; labels remain unchanged."}
    res["ablations"] = {}
    if ablations:
        if (run/"features.parquet").exists():
            features = pd.read_parquet(run/"features.parquet")
        else:
            features = derived_features if derived_features is not None else scores.sort_index()
        for group in FEATURES:
            for name, included in (("without_"+group, set(FEATURES)-{group}), ("only_"+group, {group})):
                alternative, _ = score(features, load_reference(), included, explain=False)
                metrics = population(alternative, labels, hollow)
                res["ablations"][name] = {"auc": metrics["auc"], "auc_by_tier": metrics["auc_by_tier"],
                                           "precision_at_25": metrics["precision_at_k"]["25"]}
    p = res["pooled"]
    screening_recall = p["policies"]["screening"]["recall_by_tier"]
    fp = res["false_positives"]["priority"]
    res["acceptance"] = {
        "precision_at_25": {"pass": p["precision_at_k"]["25"] >= 0.85, "value": p["precision_at_k"]["25"]},
        "recall_obvious_moderate_screening": {"pass": all((screening_recall[t] or 0) >= 0.8 for t in TIERS[:2]),
                "value": screening_recall, "precision": p["policies"]["screening"]["precision"]},
        "subtle_auc_over_0_84": {"pass": (p["auc_by_tier"]["subtle"] or 0) > 0.84, "value": p["auc_by_tier"]["subtle"]},
        "campaign_fpr_within_2x": {"pass": (fp["campaign"]["rate"] or 0) <= 2*(fp["ordinary"]["rate"] or 0), "value": fp["campaign"]["rate"]},
        "declared_solar_fpr_within_2x": {"pass": (fp["declared_solar"]["rate"] or 0) <= 2*(fp["ordinary"]["rate"] or 0), "value": fp["declared_solar"]["rate"]},
        "all_signals_valid": {"pass": res["signals"]["invalid"] == 0, "value": res["signals"]["valid_rate"]},
        "runtime_under_30s": {"pass": res["runtime_s"] < 30, "value": res["runtime_s"]}}
    return res


def print_table(res: dict) -> None:
    pooled = res["pooled"]
    print(f"Synthetic yield audit: {pooled['entities']} entities; runtime {res['runtime_s']:.3f}s")
    print("AUC by tier: "+json.dumps(pooled["auc_by_tier"]))
    print("Precision at k: "+json.dumps(pooled["precision_at_k"]))
    for name, result in pooled["policies"].items():
        print(f"{name}: "+json.dumps(result))
    for name, result in res["acceptance"].items():
        print(f"{'PASS' if result['pass'] else 'FAIL'} {name}: "+json.dumps(result))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--skip-ablations", action="store_true")
    args = parser.parse_args()
    result = evaluate(args.data, args.run, not args.skip_ablations)
    (args.run/"metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print_table(result)


if __name__ == "__main__":
    main()
