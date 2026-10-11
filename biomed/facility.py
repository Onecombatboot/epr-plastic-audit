"""Facility audit: how much waste should this site have handed over, and how much did it?

Chain of reasoning (all unsupervised, no labels):
  1. drop pickups that fail a physical possibility test (ghosts), so they cannot distort the windows;
  2. for each remaining pickup compute the waste window (days since previous collection, net of declared closures)
     and expected generation (beds in service x occupancy x kg per bed-day, or patients x kg per patient);
  3. divide out calendar effects (season, weekday) learned from the population medians;
  4. judge the facility on four views:  level against peers and against the published norm range (persistent),
     dips relative to its own normal level (intermittent), and a CUSUM level shift (change-point, with onset date).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .prep import accumulation, expected_rates


def ramp(x, lo, hi):
    return np.clip((np.asarray(x, dtype=float) - lo) / (hi - lo), 0.0, 1.0)


def _calendar_factors(p: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Season and weekday multipliers from population medians (6% fraud cannot move a median of 20,000 sites)."""
    r = p["r0"].to_numpy()
    mon = p["date"].dt.month.to_numpy()
    glob = np.median(r)
    mf = np.ones(13)
    for m in range(1, 13):
        sel = mon == m
        if sel.sum() > 500:
            mf[m] = np.median(r[sel]) / glob
    wd = p["date"].dt.weekday.to_numpy()
    one_day = (p["acc_days"].to_numpy() == 1.0)
    wf = np.ones(7)
    g1 = np.median(r[one_day]) if one_day.sum() > 1000 else glob
    for d in range(7):
        sel = one_day & (wd == d)
        if sel.sum() > 500:
            wf[d] = np.median(r[sel]) / g1
    return mf, wf


def _window_weekday_factor(date: pd.Series, acc_days: np.ndarray, wf: np.ndarray) -> np.ndarray:
    """Average weekday factor over the days a window covers (a window of 7+ days averages out to the mean factor)."""
    dow = date.dt.weekday.to_numpy()
    n = np.clip(np.round(acc_days).astype(int), 1, 7)
    tot = np.zeros(len(dow))
    for k in range(7):
        tot += np.where(k < n, wf[(dow - k) % 7], 0.0)
    return tot / n


def prepare(p_clean: pd.DataFrame, f: pd.DataFrame, ch: pd.DataFrame, ref: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-pickup normalised handover ratio r (1.0 = the mid of the published norm range)."""
    er = expected_rates(f, ref)
    p = accumulation(p_clean, f, ch, ref)
    p = p[(p["prev_date"].notna()) & (p["acc_days"] >= 0.75)].copy()
    fc = p["fcode"].to_numpy()
    p["exp_mid"] = er["e_mid"].to_numpy()[fc] * p["bed_mult"].to_numpy() * p["acc_days"].to_numpy()
    p["r0"] = np.clip(p["kg"].to_numpy(), 1e-3, None) / p["exp_mid"].to_numpy()
    mf, wf = _calendar_factors(p)
    p["r"] = p["r0"] / (mf[p["date"].dt.month.to_numpy()] * _window_weekday_factor(p["date"], p["acc_days"].to_numpy(), wf))
    p["lr"] = np.log(p["r"])
    p.attrs["calendar"] = {"month": mf[1:].round(3).tolist(), "weekday": wf.round(3).tolist()}
    return p, er


def facility_scores(p: pd.DataFrame, f: pd.DataFrame, er: pd.DataFrame, ch: pd.DataFrame, ref: dict) -> pd.DataFrame:
    g = p.groupby("fcode")["lr"]
    S = pd.DataFrame({"n": g.size(), "med": g.median(), "p25": g.quantile(0.25), "p75": g.quantile(0.75), "mean": g.mean()})
    S = S.reindex(range(len(f)))
    S.index = f["facility_id"].to_numpy()
    S["facility_type"] = f["facility_type"].to_numpy()
    S["state"] = f["state"].to_numpy()
    S["cbwtf_id"] = f["cbwtf_id"].to_numpy()
    S["n"] = S["n"].fillna(0)

    # --- persistent: level against peers of the same type, and against the published minimum
    tg = S.groupby("facility_type")["med"]
    S["peer_med"] = tg.transform("median")
    S["peer_mad"] = tg.transform(lambda s: 1.4826 * np.median(np.abs(s.dropna() - np.median(s.dropna()))))
    S["z_level"] = (S["med"] - S["peer_med"]) / S["peer_mad"].clip(lower=0.05)
    r_low = np.log(er["e_lo"].to_numpy() / er["e_mid"].to_numpy())
    S["below_norm"] = r_low - S["med"].to_numpy()                       # >0: median handover under the published minimum
    # Legitimate spread is bounded but skewed (occupancy and kg per bed are drawn from ranges), so a Gaussian or mirrored tail would
    # misjudge the lower side. Use Tukey's "far out" fence per facility type (Q1 - 3 IQR): standard, distribution-free, and untouched
    # by the minority of anomalous sites.
    q = S.groupby("facility_type")["z_level"].quantile([0.25, 0.75]).unstack()
    S["z_fence"] = S["facility_type"].map(q[0.25] - 3.0 * (q[0.75] - q[0.25]))
    S["above_norm"] = S["med"].to_numpy() - np.log(er["e_hi"].to_numpy() / er["e_mid"].to_numpy())
    s_z = np.clip((S["z_fence"] - S["z_level"]) / 2.0 + 0.0, 0.0, 1.0)             # 0 at the fence, 0.2 at 0.4 robust-z beyond it
    s_norm = ramp(S["below_norm"], 0.05, 0.35)                                   # under the lowest handover any legitimate site can have
    S["s_level"] = np.maximum(s_z, s_norm)

    # --- intermittent: dips relative to the facility's own normal level
    p75 = S["p75"].to_numpy()
    dip = (p["lr"].to_numpy() < p75[p["fcode"].to_numpy()] - 0.6)
    ds = pd.Series(dip.astype(float)).groupby(p["fcode"].to_numpy()).mean().reindex(range(len(f)))
    S["dip_share"] = ds.to_numpy()
    tdg = S.groupby("facility_type")["dip_share"]
    S["peer_dip"] = tdg.transform("median")
    S["peer_dip_mad"] = tdg.transform(lambda s: 1.4826 * np.median(np.abs(s.dropna() - np.median(s.dropna()))))
    S["z_dip"] = (S["dip_share"] - S["peer_dip"]) / S["peer_dip_mad"].clip(lower=0.015)
    S["s_dip"] = ramp(S["z_dip"], 3.0, 12.0)

    # --- change-point: CUSUM of monthly medians
    mi = ((p["date"].dt.year - 2025) * 12 + p["date"].dt.month - 4).to_numpy()
    mi = np.clip(mi, 0, 11)
    M = pd.DataFrame({"fc": p["fcode"].to_numpy(), "mi": mi, "lr": p["lr"].to_numpy()}).groupby(["fc", "mi"])["lr"].median().unstack("mi")
    M = M.reindex(index=range(len(f)), columns=range(12))
    X = M.to_numpy()
    mean_f = np.nanmean(X, axis=1, keepdims=True)
    Xf = np.where(np.isnan(X), mean_f, X)
    dev = Xf - Xf.mean(axis=1, keepdims=True)
    sig_m = pd.Series(np.nanstd(Xf - np.nanmedian(Xf, axis=1, keepdims=True), axis=1)).groupby(f["facility_type"].to_numpy()).transform("median").to_numpy()
    sig_m = np.clip(sig_m, 0.03, None)
    cs = np.cumsum(dev, axis=1)[:, :-1] / sig_m[:, None]
    kstar = np.abs(cs).argmax(axis=1)                                   # change after month kstar
    stat = np.abs(cs).max(axis=1) / np.sqrt(12)
    rows = np.arange(len(f))
    before = np.array([Xf[i, :kstar[i] + 1].mean() for i in rows])
    after = np.array([Xf[i, kstar[i] + 1:].mean() for i in rows])
    S["cp_drop"] = before - after                                       # >0: level fell
    S["cp_stat"] = stat
    S["cp_month"] = kstar + 1                                           # first month at the new level (0 = Apr 2025)
    S["s_cp"] = ramp(S["cp_stat"], 1.4, 3.5) * ramp(S["cp_drop"], 0.25, 0.9)

    # --- combine (noisy-OR) plus a continuous tie-breaker so extreme cases stay ordered
    parts = {"level": S["s_level"], "dips": 0.90 * S["s_dip"], "shift": 0.75 * S["s_cp"]}
    S["score"] = 1 - (1 - parts["level"]) * (1 - parts["dips"]) * (1 - parts["shift"])
    S["score"] = (S["score"] * 0.98 + 0.02 * ramp(-S["z_level"], 0, 12)).fillna(0.0)
    for k, v in parts.items():
        S["w_" + k] = v
    S["n_changes"] = f["facility_id"].map(ch.groupby("facility_id").size()).fillna(0).to_numpy()
    return S


def onset_date(p: pd.DataFrame, fcode: int) -> pd.Timestamp | None:
    """Daily-resolution CUSUM onset for one facility (used in case files)."""
    q = p[p["fcode"] == fcode].sort_values("date")
    if len(q) < 20:
        return None
    x = q["lr"].to_numpy()
    k = int(np.abs(np.cumsum(x - x.mean())).argmax())
    if k < 5 or k > len(x) - 5:
        return None
    return q["date"].iloc[k + 1]


def reasons(row: pd.Series, f_row: pd.Series, unit: str, norm_range: tuple[float, float], n_changes: int, p_onset: pd.Timestamp | None) -> list[str]:
    out = []
    kg_peer = np.exp(row["peer_med"])
    kg_this = np.exp(row["med"])
    if row["s_level"] > 0.05:
        out.append(f"Median handover is {kg_this / kg_peer:.0%} of the typical {row['facility_type'].replace('_', ' ')} (robust z = {row['z_level']:.1f}); "
                   f"{'below' if row['below_norm'] > 0 else 'within'} the published norm range of {norm_range[0]:.3f}-{norm_range[1]:.3f} kg per {unit}-day")
    if row["s_dip"] > 0.05:
        out.append(f"{row['dip_share']:.0%} of its pickups fall below 55% of its own normal quantity (peers: {row['peer_dip']:.0%}; z = {row['z_dip']:.1f})")
    if row["s_cp"] > 0.05:
        when = p_onset.strftime("%d %b %Y") if p_onset is not None else f"month {int(row['cp_month'])}"
        out.append(f"Handover fell by {1 - np.exp(-row['cp_drop']):.0%} from about {when} (CUSUM statistic {row['cp_stat']:.1f}, threshold 1.4)")
    out.append(f"Based on {int(row['n'])} accumulation-adjusted pickups; {int(n_changes)} declared change(s) already accounted for")
    return out
