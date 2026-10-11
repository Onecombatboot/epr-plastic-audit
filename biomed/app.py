"""Inspector dashboard for the Biomedical Waste Gap agent.

    streamlit run biomed/app.py -- --run biomed/outputs/seed42 --data data/synthetic

Reads pipeline outputs and raw tables only; labels are never read here (the inspector does not have an answer key).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from biomed.cbwtf import cbwtf_reasons  # noqa: E402
from biomed.prep import load_reference  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=str(ROOT / "biomed" / "outputs" / "seed42"))
ap.add_argument("--data", default=str(ROOT / "data" / "synthetic"))
args, _ = ap.parse_known_args()
RUN, DATA = Path(args.run), Path(args.data)
REF = load_reference()

st.set_page_config(page_title="Biomedical Waste Gap", page_icon=":hospital:", layout="wide")


@st.cache_data
def load():
    fac = pd.read_parquet(RUN / "facility_scores.parquet")
    cb = pd.read_parquet(RUN / "cbwtf_scores.parquet")
    f = pd.read_parquet(DATA / "bmw" / "facilities.parquet").set_index("facility_id")
    ch = pd.read_parquet(DATA / "bmw" / "facility_changes.parquet")
    top = pd.read_parquet(RUN / "top_facility_pickups.parquet")
    flagged = pd.read_parquet(RUN / "flagged_pickups.parquet")
    sig = {}
    for line in (RUN / "signals.jsonl").read_text(encoding="utf-8").splitlines():
        x = json.loads(line)
        sig[x["entity_id"]] = x
    metrics = json.loads((RUN / "metrics.json").read_text()) if (RUN / "metrics.json").exists() else {}
    return fac, cb, f, ch, top, flagged, sig, metrics


fac, cb, f, ch, top, flagged, sig, metrics = load()

st.title("Biomedical Waste Gap")
st.caption("Is infectious hospital waste reaching a treatment plant, or being dumped? Barcode pickups, GPS stops and plant treatment logs are cross-checked "
           "against physical limits. **Synthetic data**: no real facility appears here.")
tab_rank, tab_fac, tab_cb, tab_map, tab_method = st.tabs(["Ranked lists", "Facility case file", "Treatment plant case file", "Map", "How it works & results"])

# ------------------------------------------------------------------------------------------ ranked lists
with tab_rank:
    n_f = int((fac["score"] >= 0.2).sum())
    n_c = int((cb["score"] >= 0.2).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Facilities screened", f"{len(fac):,}")
    c2.metric("Facilities flagged", f"{n_f:,}")
    c3.metric("Treatment plants flagged", f"{n_c:,} of {len(cb)}")
    c4.metric("Pickups failing a physical test", f"{int(flagged['ghost'].sum()):,}")
    left, right = st.columns(2)
    with left:
        st.subheader("Healthcare facilities")
        ft = st.multiselect("Type", sorted(fac["facility_type"].unique()))
        stt = st.multiselect("State", sorted(fac["state"].unique()))
        d = fac.join(f[["name", "district"]])
        if ft:
            d = d[d["facility_type"].isin(ft)]
        if stt:
            d = d[d["state"].isin(stt)]
        d = d.sort_values("score", ascending=False).head(100)
        d["why"] = [(sig[i]["reasons"][0] if i in sig else "") for i in d.index]
        st.dataframe(d.rename_axis("facility_id").reset_index()[["facility_id", "name", "facility_type", "state", "score", "why"]], width="stretch", height=420,
                     column_config={"score": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f")})
    with right:
        st.subheader("Treatment plants (CBWTFs)")
        d = cb.sort_values("score", ascending=False).head(100).copy()
        d["why"] = [(sig[i]["reasons"][0] if i in sig else "") for i in d.index]
        st.dataframe(d.rename_axis("cbwtf_id").reset_index()[["cbwtf_id", "name", "state", "score", "ghost_pickups", "why"]], width="stretch", height=420,
                     column_config={"score": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f")})

# ------------------------------------------------------------------------------------------ facility case
with tab_fac:
    ids = list(top.groupby("facility_id").size().index)
    ranked = [i for i in fac.sort_values("score", ascending=False).index if i in set(ids)]
    pick = st.selectbox("Facility", ranked[:300], format_func=lambda i: f"{fac.at[i, 'score']:.2f}  {i}  {f.at[i, 'name']} ({fac.at[i, 'facility_type'].replace('_', ' ')}, {fac.at[i, 'state']})")
    row = fac.loc[pick]
    q = top[top["facility_id"] == pick].sort_values("date")
    spec = REF["types"][row["facility_type"]]
    lo = np.log(spec["kg_per_unit_day"][0] * (REF["occupancy"][0] if spec["basis"] == "beds" else 1.0)) - np.log(np.sqrt(spec["kg_per_unit_day"][0] * REF["occupancy"][0] * spec["kg_per_unit_day"][1] * REF["occupancy"][1])) if spec["basis"] == "beds" else np.log(np.sqrt(spec["kg_per_unit_day"][0] / spec["kg_per_unit_day"][1]))
    hi = -lo
    left, right = st.columns([3, 2])
    with left:
        st.subheader(f.at[pick, "name"])
        st.write(f"{f.at[pick, 'district']}, {f.at[pick, 'state']} · {row['facility_type'].replace('_', ' ')} · "
                 + (f"{int(f.at[pick, 'beds'])} beds" if spec["basis"] == "beds" else f"{int(f.at[pick, 'avg_daily_patients'])} patients/day") + f" · collected {f.at[pick, 'pickup_schedule'].replace('_', ' ')}")
        st.markdown("**Why it is flagged**")
        if pick in sig:
            for r in sig[pick]["reasons"]:
                st.markdown(f"- {r}")
        else:
            st.write("Below the flagging threshold.")
        fig = go.Figure()
        fig.add_hrect(y0=np.exp(lo), y1=np.exp(hi), fillcolor="#2f6f73", opacity=0.10, line_width=0,
                      annotation_text="range a legitimate site of this type can produce", annotation_position="top left")
        fig.add_trace(go.Scatter(x=q["date"], y=q["r"], mode="markers", marker=dict(size=4, color="#999"), name="each pickup"))
        roll = q.set_index("date")["r"].rolling("28D", min_periods=3).median()
        fig.add_trace(go.Scatter(x=roll.index, y=roll.values, mode="lines", line=dict(color="#b3261e", width=3), name="28-day median"))
        for c in ch[ch["facility_id"] == pick].itertuples():
            fig.add_vline(x=c.effective_date.timestamp() * 1000, line_dash="dot", line_color="#555")
            fig.add_annotation(x=c.effective_date, y=1.02, yref="paper", text=c.change_type.replace("_", " "), showarrow=False, font=dict(size=10))
        fig.add_hline(y=1.0, line_color="#2f6f73", line_dash="dash", annotation_text="expected (mid-norm)", annotation_position="bottom right")
        fig.update_layout(title="Waste handed over relative to expected generation (1.0 = middle of the published range, accumulation-adjusted)",
                          yaxis_title="handover / expected", yaxis_type="log", height=420, margin=dict(l=10, r=10, t=60, b=10), legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("**Detector components**")
        comp = pd.DataFrame({"component": ["level vs peers / norm", "intermittent dips", "level shift (CUSUM)"], "strength": [row["w_level"], row["w_dips"], row["w_shift"]]})
        bar = go.Figure(go.Bar(x=comp["strength"], y=comp["component"], orientation="h", marker_color="#2f6f73"))
        bar.update_layout(xaxis_range=[0, 1], height=200, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(bar, width="stretch")
        st.markdown("**Declared changes (already accounted for)**")
        cc = ch[ch["facility_id"] == pick]
        st.dataframe(cc[["effective_date", "change_type", "old_value", "new_value"]] if len(cc) else pd.DataFrame({"none": ["no declared changes"]}), width="stretch", hide_index=True)
        if pick in sig:
            st.markdown("**Evidence to pull**")
            for e in sig[pick]["evidence_refs"]:
                st.write("• " + e)

# ------------------------------------------------------------------------------------------ CBWTF case
with tab_cb:
    ranked_c = list(cb.sort_values("score", ascending=False).index)
    pc = st.selectbox("Treatment plant", ranked_c[:60], format_func=lambda i: f"{cb.at[i, 'score']:.2f}  {cb.at[i, 'name']} ({cb.at[i, 'state']})  {i}")
    r = cb.loc[pc]
    st.subheader(f"{r['name']}, {r['district']}, {r['state']}")
    st.write(f"Incinerator {r['incinerator_kg_per_hr']:.0f} kg/hr · permit {r['permitted_kg_per_day']:,.0f} kg/day · serves {int(r['n_facilities'])} facilities · {int(r['pickups']):,} pickups")
    a, b = st.columns([3, 2])
    with a:
        st.markdown("**Why it is flagged**")
        for t_ in cbwtf_reasons(r):
            st.markdown(f"- {t_}")
        d = pd.read_parquet(DATA / "bmw" / "cbwtf_daily_treatment.parquet", filters=[("cbwtf_id", "==", pc)])
        d["date"] = pd.to_datetime(d["date"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d["date"], y=d["incinerated_kg_reported"], name="reported incinerated", line=dict(color="#b3261e")))
        fig.add_trace(go.Scatter(x=d["date"], y=d["intake_kg"], name="intake", line=dict(color="#999")))
        fig.add_hline(y=r["incinerator_kg_per_hr"] * 24, line_dash="dash", line_color="#2f6f73", annotation_text="physical maximum (rated kg/hr x 24 h)")
        fig.add_hline(y=r["permitted_kg_per_day"], line_dash="dot", line_color="#555", annotation_text="permit")
        fig.update_layout(title="Daily treatment log against physical capacity", yaxis_title="kg per day", height=380, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")
    with b:
        comp = pd.DataFrame({"test": ["ghost pickups", "weight padding", "burn above capacity", "diesel too low", "intake above permit"],
                             "strength": [r["w_ghost"], r["w_padding"], r["w_capacity"], r["w_diesel"], r["w_intake"]]})
        bar = go.Figure(go.Bar(x=comp["strength"], y=comp["test"], orientation="h", marker_color="#2f6f73"))
        bar.update_layout(xaxis_range=[0, 1], height=240, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(bar, width="stretch")
        if pc in sig:
            st.markdown("**Evidence to pull**")
            for e in sig[pc]["evidence_refs"]:
                st.write("• " + e)

# ------------------------------------------------------------------------------------------ map
with tab_map:
    pm = st.selectbox("Treatment plant", ranked_c[:60], key="mapplant", format_func=lambda i: f"{cb.at[i, 'score']:.2f}  {cb.at[i, 'name']}  {i}")
    pr_ = cb.loc[pm]
    served = f[f["cbwtf_id"] == pm]
    gp = flagged[(flagged["cbwtf_id"] == pm) & flagged["ghost"]]
    gdet = pd.DataFrame()
    if len(gp):
        gdet = pd.read_parquet(DATA / "bmw" / "pickups.parquet", columns=["pickup_id", "gps_lat", "gps_lon"], filters=[("pickup_id", "in", list(gp["pickup_id"]))])
        gdet = gdet.merge(gp[["pickup_id", "facility_id", "date", "in_closure", "off_cadence", "displaced", "short_dwell", "duplicate_copy"]], on="pickup_id")
        gdet = gdet.join(f[["lat", "lon"]], on="facility_id")
    ghost_fac = set(gp["facility_id"]) if len(gp) else set()
    fig = go.Figure()
    ok = served[~served.index.isin(ghost_fac)]
    fig.add_trace(go.Scattermap(lat=ok["lat"], lon=ok["lon"], mode="markers", marker=dict(size=5, color="#2f6f73"), name="facilities served", text=ok["name"], hoverinfo="text"))
    gf = served[served.index.isin(ghost_fac)]
    fig.add_trace(go.Scattermap(lat=gf["lat"], lon=gf["lon"], mode="markers", marker=dict(size=9, color="#b3261e"), name="facilities with a pickup that cannot have happened", text=gf["name"], hoverinfo="text"))
    shown = gdet.dropna(subset=["gps_lat"]) if len(gdet) else gdet
    shown = shown[shown["displaced"]] if len(shown) else shown
    for rr in shown.head(300).itertuples():
        fig.add_trace(go.Scattermap(lat=[rr.lat, rr.gps_lat], lon=[rr.lon, rr.gps_lon], mode="lines+markers", line=dict(width=1.5, color="#b3261e"),
                                    marker=dict(size=4, color="#b3261e"), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scattermap(lat=[pr_["lat"]], lon=[pr_["lon"]], mode="markers", marker=dict(size=16, color="#1d3557"), name="treatment plant", text=[pr_["name"]], hoverinfo="text"))
    fig.update_layout(map=dict(style="open-street-map", center=dict(lat=float(pr_["lat"]), lon=float(pr_["lon"])), zoom=7), height=620, margin=dict(l=0, r=0, t=0, b=0),
                      legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"{len(served)} facilities served · {len(gp)} flagged pickups ({int(gp['displaced'].sum()) if len(gp) else 0} with GPS displaced more than 1 km, drawn as red trails from the facility to where the vehicle's GPS was). "
               "Map tiles load from OpenStreetMap, so they need an internet connection.")
    if len(gdet):
        st.dataframe(gdet[["pickup_id", "facility_id", "date", "in_closure", "off_cadence", "displaced", "short_dwell", "duplicate_copy"]].head(200), width="stretch", hide_index=True)

# ------------------------------------------------------------------------------------------ method
with tab_method:
    st.markdown("""
**Facilities.** Waste accumulates between collections, so each pickup is compared with what the site should have produced since the previous one
(beds in service x occupancy x kg per bed-day, net of declared closures and bed expansions), after dividing out season and weekday effects learned from the population.
Three views: level against peers and the published range, dips against its own normal level, and a CUSUM level shift with its onset date.

**Pickups that cannot have happened.** Recorded during a declared closure, off the facility's own collection rhythm, GPS more than 1 km away, under 3 minutes at the stop,
or a copy of another pickup's weights with no GPS fix. A missing GPS fix alone is never evidence (about 3% of stops lose GPS).

**Treatment plants.** Ghost-pickup counts, facilities pushed above anything legitimate (weight padding), burning above the incinerator's rating or permit, diesel too low for the burn.

No model was trained on fraud labels; all detectors use physical limits and robust statistics.
""")
    if metrics:
        fm, gm, cm = metrics["facility"], metrics["ghost_pickups"], metrics["cbwtf"]
        st.dataframe(pd.DataFrame([
            {"check": "Facility under-handover: precision@25 / @50 / @100", "result": " / ".join(f"{fm['precision'][k]:.2f}" for k in ("25", "50", "100"))},
            {"check": "Facility under-handover: AUC (obvious / moderate / subtle)", "result": " / ".join(f"{fm['auc_by_tier'][k]:.2f}" for k in ("obvious", "moderate", "subtle"))},
            {"check": "Facility recall at flagging threshold (obvious / moderate / subtle)", "result": " / ".join(f"{fm['recall_by_tier_at_emit'][k]:.2f}" for k in ("obvious", "moderate", "subtle"))},
            {"check": "Ghost pickups: recall / false-alarm rate on clean pickups", "result": f"{gm['recall']:.3f} / {gm['false_alarm_rate_on_clean']:.4%}"},
            {"check": "Plants: ghost / padding / overcapacity recall", "result": " / ".join(f"{cm['by_pattern'][k]['recall']:.2f}" for k in ("ghost_pickups", "weight_padding", "overcapacity"))},
        ]), width="stretch", hide_index=True)
    st.warning("Results are on synthetic data with planted fraud. The detectors were designed knowing the mechanisms the generator plants, so these numbers show the mechanisms "
               "are detectable, not how the system would perform on real barcode data. Norms (kg per bed-day, occupancy, diesel per kg) are assumptions to be replaced by published CPCB values.")
