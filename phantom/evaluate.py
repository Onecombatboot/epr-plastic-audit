"""Evaluation against the quarantined labels. Used for scoring only, never for fitting."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_io import DEFAULT_DATA

BEHAVIOURS = ["ghost_feedstock", "underprice", "cat3_inflation", "deadline_burst", "round_trip",
              "output_fabrication", "doctored_bill", "capacity_inflated", "photo_reuse"]


def load_labels(data_dir: Path | str = DEFAULT_DATA) -> pd.DataFrame:
    d = Path(data_dir) / "labels"
    lab = pd.read_parquet(d / "registry" / "entity_labels.parquet")
    lab = lab[lab.stream == "plastic"].set_index("entity_id")
    extra = pd.read_parquet(d / "plastic" / "recycler_extra_labels.parquet").set_index("entity_id")
    lab = lab.join(extra, rsuffix="_x")
    mt = pd.read_parquet(d / "plastic" / "monthly_truth.parquet").groupby("entity_id")[["true_t", "hollow_t"]].sum()
    lab = lab.join(mt)
    lab["difficulty"] = lab["difficulty"].where(lab["is_fraud"], "legit")
    return lab


def auc(score: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), ties handled by average rank."""
    s = pd.Series(np.asarray(score, dtype=float))
    y = np.asarray(y).astype(bool)
    ok = ~s.isna().to_numpy()
    s, y = s[ok], y[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = s.rank(method="average").to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def precision_at_k(score: pd.Series, y: pd.Series, k: int) -> float:
    top = score.sort_values(ascending=False).index[:k]
    return float(y.reindex(top).astype(float).mean())


def tonnes_recall_at_k(score: pd.Series, hollow_t: pd.Series, k: int) -> float:
    """Share of all hollow tonnes sitting in the top-k entities (what an inspector would actually recover)."""
    top = score.sort_values(ascending=False).index[:k]
    return float(hollow_t.reindex(top).sum() / hollow_t.sum())


def tiered_auc(score: pd.Series, lab: pd.DataFrame) -> dict:
    out = {"all": auc(score.reindex(lab.index), lab["is_fraud"])}
    for tier in ["obvious", "moderate", "subtle"]:
        m = lab["difficulty"].isin(["legit", tier])
        out[tier] = auc(score.reindex(lab.index)[m], lab["is_fraud"][m])
    return out


def recall_at_k_by(score: pd.Series, lab: pd.DataFrame, col: str, k: int) -> float:
    top = set(score.sort_values(ascending=False).index[:k])
    pos = lab.index[lab[col].astype(bool)]
    return float(np.mean([i in top for i in pos])) if len(pos) else float("nan")


def report(score: pd.Series, lab: pd.DataFrame, ks=(25, 50, 100, 225)) -> dict:
    sc = score.reindex(lab.index)
    r = {"auc": tiered_auc(sc, lab)}
    r["precision_at_k"] = {k: precision_at_k(sc, lab["is_fraud"], k) for k in ks}
    r["hollow_tonnes_recall_at_k"] = {k: tonnes_recall_at_k(sc, lab["hollow_t"], k) for k in ks}
    r["recall_by_behaviour_at_225"] = {b: recall_at_k_by(sc, lab, b, 225) for b in BEHAVIOURS}
    r["recall_by_tier_at_225"] = {t: float(np.mean([i in set(sc.sort_values(ascending=False).index[:225]) for i in lab.index[lab["difficulty"] == t]]))
                                   for t in ["obvious", "moderate", "subtle"]}
    return r
