"""End-to-end Phantom Plastic pipeline.

    python -m phantom.pipeline --data data/synthetic --out phantom/outputs/seed42 --epochs 200 --evaluate

Steps: features -> rules -> Isolation Forest (+SHAP) -> graph autoencoder -> rank ensemble -> brand exposure
-> signals (shared agent contract) -> case files. With --evaluate, scores are compared with the quarantined labels.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import __version__
from .data_io import DEFAULT_DATA, load_all, load_reference
from .evaluate import auc, load_labels, report
from .evidence import EvidenceStore
from .exposure import brand_exposure, hollow_fraction_estimate, risk_from_rank
from .features import GROUPS, build_features, collector_table, offtaker_table
from .graph import dominant_scores
from .models import IForestDetector, feature_columns, generic_iforest_score, rank01, suspicion_matrix
from .rules import rule_scores

COLLECTOR_DIR = {"c_ratio_max": 1, "c_months": -1, "c_n_buyers": -1, "c_age_at_first_inv_days": -1, "c_vehicles": -1,
                 "round": 1, "sunday": 1, "mean_t": 1, "declared_cap": -1}
OFFTAKER_DIR = {"o_n_sellers": -1, "o_age_days": -1, "n_inv": 0, "mean_inv_t": 0, "round": 1}


def collector_features(t) -> pd.DataFrame:
    fi = t["feedstock_invoices"].rename(columns={"buyer_id": "entity_id"})
    fi["month"] = fi["invoice_date"].dt.to_period("M").dt.to_timestamp()
    col = t["collectors"].set_index("collector_id")
    cs = collector_table(fi, col)
    ci = fi[fi.seller_type == "collector"].copy()
    ci["round"] = (np.abs(ci["tonnes"] * 2 - np.round(ci["tonnes"] * 2)) < 1e-9).astype(np.int8)
    ci["sunday"] = (ci["invoice_date"].dt.weekday == 6).astype(np.int8)
    g = ci.groupby("seller_id")
    cs["round"] = g["round"].mean()
    cs["sunday"] = g["sunday"].mean()
    cs["mean_t"] = g["tonnes"].mean()
    cs["declared_cap"] = col["declared_monthly_capacity_t"].reindex(cs.index)
    return cs[list(COLLECTOR_DIR)]


def offtaker_features(t) -> pd.DataFrame:
    o = t["output_invoices"].rename(columns={"seller_id": "entity_id"})
    ofk = t["offtakers"].set_index("offtaker_id")
    ost = offtaker_table(o, ofk)
    o["round"] = (np.abs(o["tonnes"] * 2 - np.round(o["tonnes"] * 2)) < 1e-9).astype(np.int8)
    g = o.groupby("buyer_id")
    ost["n_inv"] = g.size()
    ost["mean_inv_t"] = g["tonnes"].mean()
    ost["round"] = g["round"].mean()
    return ost[list(OFFTAKER_DIR)]


def evaluate_run(out: Path | str, data_dir=DEFAULT_DATA) -> dict:
    """Score a finished run (files in `out`) against the quarantined labels."""
    from scipy.stats import spearmanr
    out = Path(out)
    S = pd.read_parquet(out / "recycler_scores.parquet")
    BX = pd.read_parquet(out / "brand_exposure.parquet")
    cscore = pd.read_parquet(out / "collector_scores.parquet")["collector_score"]
    oscore = pd.read_parquet(out / "offtaker_scores.parquet")["offtaker_score"]
    lab = load_labels(data_dir)
    res: dict = {"recycler": {c: report(S[c], lab) for c in ["rule_score", "iforest_score", "dominant_score", "ens_score"]}}
    cl = pd.read_parquet(Path(data_dir) / "labels" / "plastic" / "collector_labels.parquet").set_index("collector_id")
    ol = pd.read_parquet(Path(data_dir) / "labels" / "plastic" / "offtaker_labels.parquet").set_index("offtaker_id")
    res["collector_ghost_auc"] = auc(cscore.reindex(cl.index), cl["is_ghost"])
    res["offtaker_shell_auc"] = auc(oscore.reindex(ol.index), ol["is_shell"])
    bl = pd.read_parquet(Path(data_dir) / "labels" / "plastic" / "brand_labels.parquet").set_index("brand_id")
    j = BX.join(bl, how="inner", rsuffix="_true")
    res["brand"] = {"spearman_exposure_vs_true": float(spearmanr(j["exposure"], j["true_hollow_exposure"]).statistic),
                    "spearman_hollow_t_est_vs_true": float(spearmanr(j["hollow_t_est"], j["hollow_t"]).statistic),
                    "total_hollow_t_true": float(j["hollow_t"].sum()), "total_hollow_t_est": float(j["hollow_t_est"].sum())}
    top_k = j.sort_values("hollow_t_est", ascending=False).head(100).index
    res["brand"]["top100_share_of_true_hollow_t"] = float(j.loc[top_k, "hollow_t"].sum() / j["hollow_t"].sum())
    res["brand"]["top100_overlap_with_true_top100"] = int(len(set(top_k) & set(j.sort_values("hollow_t", ascending=False).head(100).index)))
    (out / "metrics.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


def run(data_dir=DEFAULT_DATA, out_dir="phantom/outputs/seed42", epochs=200, seed=0, evaluate=True, exclude_groups: set[str] | None = None,
        top_cases=100, prevalence=0.10, weights=(1.0, 1.0, 0.25), dominant_attrs=True, verbose=True) -> dict:
    t0 = time.time()
    exclude_groups = exclude_groups or set()
    out = Path(out_dir)
    (out / "cases").mkdir(parents=True, exist_ok=True)
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    t = load_all(data_dir)
    ref = load_reference()
    F = build_features(t, ref, with_identity="identity" not in exclude_groups, registry_dir=Path(data_dir) / "registry")
    F.to_parquet(out / "features.parquet")
    log(f"[{time.time() - t0:5.0f}s] features {F.shape}")

    S = pd.DataFrame(index=F.index)
    S["rule_score"] = rule_scores(F, exclude_groups)
    det = IForestDetector(seed=seed, exclude_groups=exclude_groups).fit(F)
    S["iforest_score"] = det.score(F)
    log(f"[{time.time() - t0:5.0f}s] rules + isolation forest")

    cols = feature_columns(exclude_groups)
    U = suspicion_matrix(F, cols, F) if dominant_attrs else None
    D = dominant_scores(t, F, U, epochs=epochs, seed=seed, verbose=verbose)
    S["dominant_score"] = D.loc[D["type"] == "recycler", "score"].reindex(F.index)
    log(f"[{time.time() - t0:5.0f}s] graph autoencoder")

    r = pd.concat([rank01(S["rule_score"]), rank01(S["iforest_score"]), rank01(S["dominant_score"])], axis=1)
    w = np.array(weights, dtype=float) / np.sum(weights)
    S["ens_score"] = (r.to_numpy() * w).sum(axis=1)
    S["ens_pct"] = rank01(S["ens_score"])
    S["rank"] = S["ens_score"].rank(ascending=False, method="first").astype(int)
    S.to_parquet(out / "recycler_scores.parquet")

    # collectors and offtakers
    cf = collector_features(t)
    cscore = pd.concat([rank01(generic_iforest_score(cf, COLLECTOR_DIR, seed)), rank01(D.loc[D.type == "collector", "score"].reindex(cf.index))], axis=1).mean(axis=1)
    of = offtaker_features(t)
    oscore = pd.concat([rank01(generic_iforest_score(of, OFFTAKER_DIR, seed)), rank01(D.loc[D.type == "offtaker", "score"].reindex(of.index))], axis=1).mean(axis=1)
    pd.DataFrame({"collector_score": cscore}).to_parquet(out / "collector_scores.parquet")
    pd.DataFrame({"offtaker_score": oscore}).to_parquet(out / "offtaker_scores.parquet")

    # brand exposure
    risk = risk_from_rank(S["ens_pct"], prevalence)
    hf = hollow_fraction_estimate(F)
    BX = brand_exposure(t, risk, hf)
    BX.to_parquet(out / "brand_exposure.parquet")
    log(f"[{time.time() - t0:5.0f}s] brand exposure")

    # signals (shared contract) + case files
    ev = EvidenceStore(t, F)
    top = S.sort_values("ens_score", ascending=False).head(top_cases).index
    sv = det.shap_values(F.loc[top])
    from .rules import rule_reasons
    with open(out / "signals.jsonl", "w", encoding="utf-8") as fh:
        for eid in S.sort_values("ens_score", ascending=False).index:
            if S.at[eid, "ens_pct"] < 0.90:
                break
            reasons = [x["text"] for x in rule_reasons(F, eid, top=4)]
            if not reasons:
                shap_r = det.reasons(F, sv, eid, 3) if eid in sv.index else []
                reasons = [f"{x['label']}: {x['value']} (higher suspicion than {x['population_percentile']:.0f}% of plants)" for x in shap_r]
            fh.write(json.dumps({"entity_id": eid, "agent": "phantom_plastic", "score": round(float(S.at[eid, "ens_pct"]), 4), "reasons": reasons,
                                 "evidence_refs": [f"plastic/case:{eid}"]}) + "\n")
    for eid in top:
        sr = det.reasons(F, sv, eid, 4)
        case = ev.case_file(eid, S.loc[eid, ["rule_score", "iforest_score", "dominant_score", "ens_score", "ens_pct"]], sr)
        case["rank"] = int(S.at[eid, "rank"])
        (out / "cases" / f"{eid}.json").write_text(json.dumps(case, indent=1, default=str), encoding="utf-8")
    log(f"[{time.time() - t0:5.0f}s] signals + {len(top)} case files")

    res: dict = {"version": __version__, "runtime_s": round(time.time() - t0, 1), "n_recyclers": int(len(F))}
    if evaluate:
        res.update(evaluate_run(out, data_dir))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default="phantom/outputs/seed42")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-evaluate", action="store_true")
    ap.add_argument("--exclude", default="", help="comma-separated feature groups to hold out (ablation)")
    a = ap.parse_args(argv)
    ex = set(filter(None, a.exclude.split(",")))
    res = run(a.data, a.out, a.epochs, a.seed, not a.no_evaluate, ex)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
