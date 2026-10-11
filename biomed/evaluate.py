"""Score the biomedical-waste detectors against the quarantined labels (labels are never used for fitting).

    python -m biomed.evaluate --data data/synthetic --run biomed/outputs/seed42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .prep import DEFAULT_DATA, load_reference

EMIT = 0.20


def auc(score, y) -> float:
    s = pd.Series(np.asarray(score, dtype=float))
    y = np.asarray(y).astype(bool)
    ok = ~s.isna().to_numpy()
    s, y = s[ok], y[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = s.rank(method="average").to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def p_at_k(score: pd.Series, y: pd.Series, k: int) -> float:
    return float(y.reindex(score.sort_values(ascending=False).index[:k]).astype(float).mean())


def evaluate(data_dir=DEFAULT_DATA, run_dir="biomed/outputs/seed42") -> dict:
    d, r = Path(data_dir), Path(run_dir)
    L = d / "labels" / "bmw"
    fl = pd.read_parquet(L / "facility_labels.parquet").set_index("facility_id")
    cl = pd.read_parquet(L / "cbwtf_labels.parquet").set_index("cbwtf_id")
    fac = pd.read_parquet(r / "facility_scores.parquet")
    cb = pd.read_parquet(r / "cbwtf_scores.parquet")
    flagged = pd.read_parquet(r / "flagged_pickups.parquet")
    pl = pd.read_parquet(L / "pickup_labels.parquet", columns=["pickup_id", "is_ghost"]).set_index("pickup_id")["is_ghost"]
    res: dict = {}

    # ---------------------------------------------------------------- facilities
    y = fl["under_handover"].reindex(fac.index)
    s = fac["score"]
    res["facility"] = {"auc": auc(s, y), "precision": {k: p_at_k(s, y, k) for k in (10, 25, 50, 100, 300)},
                       "n_positive": int(y.sum())}
    emitted = fac["tier"] == "priority"                  # high-confidence tier
    queued = fac["tier"] != "none"                        # priority + screening (review) tier
    res["facility"]["emitted"] = int(emitted.sum())
    res["facility"]["precision_of_emitted"] = float(y[emitted].mean())
    res["facility"]["review_queue"] = {"flagged": int(queued.sum()), "precision": float(y[queued].mean())}
    tier = fl["difficulty"].reindex(fac.index)
    res["facility"]["recall_by_tier_at_emit"] = {t: float(emitted[(tier == t) & y].mean()) for t in ["obvious", "moderate", "subtle"]}
    res["facility"]["recall_by_tier_review_queue"] = {t: float(queued[(tier == t) & y].mean()) for t in ["obvious", "moderate", "subtle"]}
    kpos = int(y.sum())
    top = set(s.sort_values(ascending=False).index[:kpos])
    res["facility"]["recall_by_tier_at_k_positives"] = {t: float(np.mean([i in top for i in y.index[(tier == t) & y]])) for t in ["obvious", "moderate", "subtle"]}
    order = s.sort_values(ascending=False).index
    res["facility"]["inspection_budget"] = {}
    for k in (500, 1000, 1234, 2000, 3000, 5000):
        top_k = set(order[:k])
        res["facility"]["inspection_budget"][k] = {"precision": float(y.reindex(list(top_k)).mean()),
                                                   **{t: float(np.mean([i in top_k for i in y.index[(tier == t) & y]])) for t in ["obvious", "moderate", "subtle"]}}
    inter = fl["intermittent"].reindex(fac.index)
    res["facility"]["recall_at_emit_persistent_vs_intermittent"] = {"persistent": float(emitted[y & ~inter].mean()), "intermittent": float(emitted[y & inter].mean())}
    res["facility"]["recall_review_queue_persistent_vs_intermittent"] = {"persistent": float(queued[y & ~inter].mean()), "intermittent": float(queued[y & inter].mean())}
    res["facility"]["auc_by_tier"] = {t: auc(s[(tier == t) | ~y], y[(tier == t) | ~y]) for t in ["obvious", "moderate", "subtle"]}
    res["facility"]["component_auc"] = {c: auc(fac[c], y) for c in ["w_level", "w_dips", "w_shift"]}

    # confounders: false-positive rate among facilities that did NOT under-hand-over
    ch = pd.read_parquet(d / "bmw" / "facility_changes.parquet")
    exp_f = set(ch.loc[ch.change_type == "bed_expansion", "facility_id"])
    clo_f = set(ch.loc[ch.change_type == "temporary_closure", "facility_id"])
    clean = ~y
    fpr = lambda m: {"n": int((clean & m).sum()), "flagged": int((emitted & clean & m).sum()), "rate": float((emitted & clean & m).sum() / max((clean & m).sum(), 1))}
    idx = pd.Series(fac.index, index=fac.index)
    emitted = emitted                                      # false positives below are for the priority tier
    res["facility"]["false_positives"] = {
        "all clean facilities": fpr(pd.Series(True, index=fac.index)),
        "declared bed expansion": fpr(idx.isin(exp_f)), "declared temporary closure": fpr(idx.isin(clo_f)),
        **{f"type={t}": fpr(fac["facility_type"] == t) for t in sorted(fac["facility_type"].unique())}}

    # ---------------------------------------------------------------- pickups (ghost detector)
    gh = pl[pl].index
    fl_ids = set(flagged.loc[flagged["ghost"], "pickup_id"])
    n_ghost = int(pl.sum())
    n_clean = int((~pl).sum())
    caught = sum(1 for i in gh if i in fl_ids)
    false = len(fl_ids) - caught
    res["ghost_pickups"] = {"planted": n_ghost, "flagged": len(fl_ids), "recall": caught / n_ghost, "false_alarm_rate_on_clean": false / n_clean, "false_alarms": false}
    # by test
    by = {}
    gp = flagged[flagged["pickup_id"].isin(gh)]
    for k in ["in_closure", "off_cadence", "displaced", "short_dwell", "duplicate_copy", "gps_missing"]:
        by[k] = float(gp[k].mean())
    res["ghost_pickups"]["share_of_planted_failing_each_test"] = by

    # ---------------------------------------------------------------- CBWTFs
    yc = cl[["ghost_pickups", "weight_padding", "overcapacity"]].any(axis=1).reindex(cb.index)
    sc = cb["score"]
    res["cbwtf"] = {"auc_any": auc(sc, yc), "precision": {k: p_at_k(sc, yc, k) for k in (10, 25, 50)}, "n_positive": int(yc.sum())}
    tc = cl["difficulty"].reindex(cb.index)
    # A plant labelled overcapacity whose true intake never exceeded its incinerator has no overcapacity in its data at all (zero untreated
    # kg in the truth table): the label is vacuous there, so it is not counted as a positive. Raw counts are reported alongside.
    truth = pd.read_parquet(L / "cbwtf_treatment_truth.parquet").groupby("cbwtf_id")["untreated_kg"].sum().reindex(cb.index).fillna(0)
    cl_eff = cl.copy()
    cl_eff["overcapacity_effective"] = cl["overcapacity"] & (truth.reindex(cl.index).fillna(0) > 0)
    res["cbwtf"]["overcapacity_label_artifacts"] = int((cl["overcapacity"] & ~cl_eff["overcapacity_effective"]).sum())
    comp = {"ghost_pickups": "w_ghost", "weight_padding": "w_padding", "overcapacity": "w_capacity"}
    res["cbwtf"]["by_pattern"] = {}
    for pat, col in comp.items():
        yp = (cl_eff["overcapacity_effective"] if pat == "overcapacity" else cl[pat]).reindex(cb.index)
        det = cb[col] >= 0.2
        res["cbwtf"]["by_pattern"][pat] = {"n": int(yp.sum()), "auc_component": auc(cb[col], yp), "recall": float(det[yp].mean()),
                                           "false_alarms_on_clean_plants": int((det & ~yc).sum()),
                                           "recall_by_tier": {t: (float(det[yp & (tc == t)].mean()) if (yp & (tc == t)).any() else None) for t in ["obvious", "moderate", "subtle"]},
                                           "n_by_tier": {t: int((yp & (tc == t)).sum()) for t in ["obvious", "moderate", "subtle"]}}
    res["cbwtf"]["emitted"] = int((sc >= EMIT).sum())
    res["cbwtf"]["precision_of_emitted"] = float(yc[sc >= EMIT].mean())

    # ---------------------------------------------------------------- signal validity
    sigs = [json.loads(x) for x in (r / "signals.jsonl").read_text(encoding="utf-8").splitlines()]
    bad = [x for x in sigs if not (0 <= x["score"] <= 1 and len(x["reasons"]) >= 2 and len(x["evidence_refs"]) >= 1)]
    res["signals"] = {"count": len(sigs), "invalid": len(bad), "facility": sum(x["entity_kind"] == "facility" for x in sigs), "cbwtf": sum(x["entity_kind"] == "cbwtf" for x in sigs)}
    return res


def print_table(res: dict):
    f, g, c, s = res["facility"], res["ghost_pickups"], res["cbwtf"], res["signals"]
    print("\nBIOMEDICAL WASTE GAP: evaluation against planted fraud (labels never used for fitting)\n" + "=" * 86)
    print(f"Facility under-handover   AUC {f['auc']:.3f}   positives {f['n_positive']}   emitted {f['emitted']} (precision {f['precision_of_emitted']:.2f})")
    print("  precision@k           " + "  ".join(f"@{k}: {v:.2f}" for k, v in f["precision"].items()))
    print("  AUC by tier           " + "  ".join(f"{k}: {v:.3f}" for k, v in f["auc_by_tier"].items()))
    print("  recall at emit        " + "  ".join(f"{k}: {v:.2f}" for k, v in f["recall_by_tier_at_emit"].items()) + "    (persistent/intermittent: " +
          " / ".join(f"{v:.2f}" for v in f["recall_at_emit_persistent_vs_intermittent"].values()) + ")")
    rq = f["recall_by_tier_review_queue"]
    print(f"  + screening queue     flagged {f['review_queue']['flagged']} (precision {f['review_queue']['precision']:.2f})   recall " + "  ".join(f"{k}: {v:.2f}" for k, v in rq.items()))
    print("  recall at k=positives " + "  ".join(f"{k}: {v:.2f}" for k, v in f["recall_by_tier_at_k_positives"].items()))
    print("  inspection budget k -> precision | recall obvious / moderate / subtle")
    for k, v in f["inspection_budget"].items():
        print(f"      k={int(k):5d}   {v['precision']:.2f} | {v['obvious']:.2f} / {v['moderate']:.2f} / {v['subtle']:.2f}")
    print("  false-positive rate among clean facilities:")
    for k, v in f["false_positives"].items():
        print(f"      {k:28s} {v['flagged']:4d} / {v['n']:5d}  = {v['rate']:.3%}")
    print(f"Ghost pickups             planted {g['planted']:,}   flagged {g['flagged']:,}   recall {g['recall']:.3f}   false-alarm rate on clean {g['false_alarm_rate_on_clean']:.4%} ({g['false_alarms']} pickups)")
    print("  planted ghosts failing each test: " + "  ".join(f"{k}: {v:.0%}" for k, v in g["share_of_planted_failing_each_test"].items()))
    print(f"  (overcapacity positives exclude {c.get('overcapacity_label_artifacts', 0)} plant(s) whose true intake never exceeded capacity)")
    print(f"CBWTF (any fraud)         AUC {c['auc_any']:.3f}   positives {c['n_positive']}   emitted {c['emitted']} (precision {c['precision_of_emitted']:.2f})   precision@10/25/50: " +
          " / ".join(f"{v:.2f}" for v in c["precision"].values()))
    for pat, v in c["by_pattern"].items():
        tiers = "  ".join(f"{t}: " + ("n/a" if v["recall_by_tier"][t] is None else f"{v['recall_by_tier'][t]:.2f}") + f" (n={v['n_by_tier'][t]})" for t in ["obvious", "moderate", "subtle"])
        print(f"  {pat:15s} n={v['n']:2d}  AUC {v['auc_component']:.3f}  recall {v['recall']:.2f}  false alarms {v['false_alarms_on_clean_plants']}   by tier: {tiers}")
    print(f"Signals                   {s['count']} emitted ({s['facility']} facilities, {s['cbwtf']} CBWTFs); invalid (score range / <2 reasons / no evidence): {s['invalid']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--run", default="biomed/outputs/seed42")
    a = ap.parse_args(argv)
    res = evaluate(a.data, a.run)
    (Path(a.run) / "metrics.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print_table(res)


if __name__ == "__main__":
    main()
