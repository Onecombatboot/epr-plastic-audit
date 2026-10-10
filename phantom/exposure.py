"""Layer 4: propagate recycler risk to the brands that bought their certificates."""
from __future__ import annotations

import numpy as np
import pandas as pd


def risk_from_rank(pct: pd.Series, prevalence: float = 0.10, sharpness: float = 40.0) -> pd.Series:
    """Map a rank percentile to a probability-like 'is hollow' risk.

    The auditor supplies the assumed share of hollow recyclers (default 10%); recyclers above that rank cut-off get
    risk near 1, below it near 0. This is a stated assumption, not a calibrated probability.
    """
    cut = 1 - prevalence * 1.3
    return pd.Series(1 / (1 + np.exp(-(pct - cut) * sharpness)), index=pct.index, name="risk")


def hollow_fraction_estimate(F: pd.DataFrame) -> pd.Series:
    """Physics-based estimate of the hollow share of a flagged recycler's claims.

    Compare electricity and product sales with what the *typical* plant shows for the same tonnes; the shortfall is
    the part the physical footprint cannot support. Falls back to 0.5 when neither signal exists.
    """
    med_e = F["energy_ratio"].median()
    med_o = F["out_to_expected"].median()
    solar = (0.8 * F["solar_frac_of_load"].fillna(0)).clip(upper=0.6)
    he = (1 - (F["energy_ratio"] / (1 - solar)) / med_e).clip(0, 1)
    ho = (1 - F["out_to_expected"] / med_o).clip(0, 1)
    h = pd.concat([he, ho], axis=1).max(axis=1, skipna=True)
    return h.fillna(0.5).clip(lower=0.2).rename("hollow_frac_est")


def brand_exposure(t: dict, risk: pd.Series, hollow_frac: pd.Series | None = None) -> pd.DataFrame:
    s = t["certificate_sales"][["seller_id", "buyer_id", "tonnes"]]
    g = s.groupby(["buyer_id", "seller_id"], observed=True)["tonnes"].sum().reset_index()
    g["risk"] = g["seller_id"].map(risk).fillna(0.0)
    g["h"] = g["seller_id"].map(hollow_frac).fillna(0.5) if hollow_frac is not None else 1.0
    g["risky_t"] = g["tonnes"] * g["risk"]
    g["hollow_est"] = g["risky_t"] * g["h"]
    agg = g.groupby("buyer_id").agg(bought_t=("tonnes", "sum"), risky_t=("risky_t", "sum"), hollow_t_est=("hollow_est", "sum"),
                                     n_recyclers=("seller_id", "nunique"))
    agg["exposure"] = agg["risky_t"] / agg["bought_t"]
    agg["hollow_share_est"] = agg["hollow_t_est"] / agg["bought_t"]
    top = (g.sort_values("hollow_est", ascending=False).groupby("buyer_id").head(3)
            .groupby("buyer_id").apply(lambda d: [f"{r.seller_id}:{r.hollow_est:,.0f}t" for r in d.itertuples()], include_groups=False))
    agg["top_sources"] = top
    return agg
