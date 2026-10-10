"""Stress tests that keep the headline numbers honest.

  holdout   : the same untouched detectors on an independent world (different seed)
  evasive   : fraudsters who hide the by-products an auditor looks for (fewer side traces, more covering)
  blackout  : remove whole evidence sources (e.g. no electricity data) and see what survives
  single    : one evidence group at a time (how much does each source detect on its own?)

Rules + Isolation Forest only (seconds); the graph autoencoder is run by the main pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from .data_io import DEFAULT_DATA, load_all, load_reference
from .evaluate import load_labels, report
from .features import GROUPS, build_features
from .models import IForestDetector, rank01
from .rules import rule_scores

ALL_GROUPS = list(GROUPS)


def fast_scores(F: pd.DataFrame, exclude: set[str] | None = None, seed: int = 0) -> pd.DataFrame:
    exclude = exclude or set()
    S = pd.DataFrame(index=F.index)
    S["rules"] = rule_scores(F, exclude)
    S["iforest"] = IForestDetector(seed=seed, exclude_groups=exclude).fit(F).score(F)
    S["ensemble"] = (rank01(S["rules"]) + rank01(S["iforest"])) / 2
    return S


def summarise(S: pd.DataFrame, lab: pd.DataFrame, col: str = "ensemble") -> dict:
    r = report(S[col], lab)
    return {"auc_all": r["auc"]["all"], "auc_subtle": r["auc"]["subtle"], "p@50": r["precision_at_k"][50], "p@100": r["precision_at_k"][100],
            "p@225": r["precision_at_k"][225], "hollow_t_recall@100": r["hollow_tonnes_recall_at_k"][100], "hollow_t_recall@225": r["hollow_tonnes_recall_at_k"][225],
            "recall_subtle@225": r["recall_by_tier_at_225"]["subtle"]}


def world(data_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = load_all(data_dir)
    F = build_features(t, load_reference(), registry_dir=Path(data_dir) / "registry")
    return F, load_labels(data_dir)


def main(out="phantom/outputs/experiments.json"):
    res: dict = {}
    worlds = {"baseline_seed42": DEFAULT_DATA, "holdout_seed7": DEFAULT_DATA.parent / "synthetic_seed7", "evasive_seed42": DEFAULT_DATA.parent / "synthetic_evasive"}
    for name, d in worlds.items():
        if not Path(d).exists():
            print("skip", name)
            continue
        print("==", name, flush=True)
        F, lab = world(d)
        S = fast_scores(F)
        res[name] = {c: summarise(S, lab, c) for c in ["rules", "iforest", "ensemble"]}
        print(json.dumps(res[name]["ensemble"], default=float), flush=True)
        if name in ("baseline_seed42", "evasive_seed42"):
            res[name + "_blackout"] = {}
            for g in ALL_GROUPS:
                res[name + "_blackout"][f"without_{g}"] = summarise(fast_scores(F, {g}), lab)
            combos = {"no_electricity_data": {"energy", "bills"}, "no_price_data": {"price"}, "no_identity_data": {"identity"},
                      "no_invoice_data": {"feedstock", "collectors", "benford"}, "transactions_only (no bills, no registry)": {"energy", "bills", "identity", "registry", "capacity"}}
            for k, ex in combos.items():
                res[name + "_blackout"][k] = summarise(fast_scores(F, ex), lab)
            res[name + "_single"] = {}
            for g in ALL_GROUPS:
                res[name + "_single"][f"only_{g}"] = summarise(fast_scores(F, set(ALL_GROUPS) - {g}), lab)
    Path(out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


if __name__ == "__main__":
    main(*sys.argv[1:])
