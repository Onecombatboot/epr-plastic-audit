"""Auditor dashboard for Phantom Plastic.   streamlit run phantom/app.py -- --run phantom/outputs/seed42 --data data/synthetic

Reads the pipeline outputs only (no recomputation on page load). Labels are never read here: the dashboard shows
what an inspector would see, not the answer key.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from phantom.rules import RULES, rule_reasons  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=str(ROOT / "phantom" / "outputs" / "seed42"))
ap.add_argument("--data", default=str(ROOT / "data" / "synthetic"))
args, _ = ap.parse_known_args()
RUN, DATA = Path(args.run), Path(args.data)

st.set_page_config(page_title="Phantom Plastic", page_icon=":recycle:", layout="wide")


@st.cache_data
def load():
    S = pd.read_parquet(RUN / "recycler_scores.parquet")
    F = pd.read_parquet(RUN / "features.parquet")
    ent = pd.read_parquet(DATA / "registry" / "entities.parquet").set_index("entity_id")
    BX = pd.read_parquet(RUN / "brand_exposure.parquet")
    brands = pd.read_parquet(DATA / "plastic" / "brands.parquet").set_index("brand_id")
    metrics = json.loads((RUN / "metrics.json").read_text()) if (RUN / "metrics.json").exists() else {}
    return S, F, ent, BX, brands, metrics


S, F, ent, BX, brands, metrics = load()
cases = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (RUN / "cases").glob("*.json")}

st.title("Phantom Plastic")
st.caption("Which plastic recyclers are most likely issuing certificates for recycling that never happened? "
           "Built on **synthetic data** calibrated to published CPCB figures; no real company appears here.")

tab_rank, tab_case, tab_brand, tab_ring, tab_method = st.tabs(["Inspection list", "Case file", "Brand exposure", "Trading loops", "How it works & results"])

# ----------------------------------------------------------------------------------------------- list
with tab_rank:
    c1, c2, c3, c4 = st.columns(4)
    flagged = S[S["ens_pct"] >= 0.90]
    c1.metric("Recyclers screened", f"{len(S):,}")
    c2.metric("Flagged for inspection (top 10%)", f"{len(flagged):,}")
    cert_t = F["cert_t"].sum()
    c3.metric("Certificate tonnes screened", f"{cert_t:,.0f}")
    c4.metric("Tonnes held by flagged plants", f"{F.loc[flagged.index, 'cert_t'].sum():,.0f}")
    state = st.multiselect("State", sorted(ent["state"].unique()))
    top_n = st.slider("Show top", 10, 200, 50, 10)
    df = S.join(ent[["legal_name", "state", "declared_capacity_tpa"]]).join(F[["cert_t", "energy_ratio", "price_vs_market"]])
    if state:
        df = df[df["state"].isin(state)]
    df = df.sort_values("ens_score", ascending=False).head(top_n)
    df["why"] = [" | ".join(r["rule"] for r in rule_reasons(F, i, top=3)) or "model signal" for i in df.index]
    show = df[["rank", "legal_name", "state", "ens_pct", "cert_t", "why"]].rename(columns={"ens_pct": "risk percentile", "cert_t": "certificate t"})
    st.dataframe(show, width="stretch", column_config={"risk percentile": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.3f")})

# ----------------------------------------------------------------------------------------------- case
with tab_case:
    ids = list(S.sort_values("ens_score", ascending=False).index[:300])
    pick = st.selectbox("Recycler", ids, format_func=lambda i: f"#{int(S.at[i, 'rank'])}  {ent.at[i, 'legal_name']}  ({ent.at[i, 'state']})")
    case = cases.get(pick)
    left, right = st.columns([3, 2])
    with left:
        st.subheader(ent.at[pick, "legal_name"])
        st.write(f"{ent.at[pick, 'district']}, {ent.at[pick, 'state']} · declared capacity {ent.at[pick, 'declared_capacity_tpa']:,.0f} t/yr · "
                 f"connected load {ent.at[pick, 'connected_load_kw']:,.0f} kW")
        st.markdown("**Why it is flagged**")
        reasons = rule_reasons(F, pick, top=7)
        if reasons:
            for r in reasons:
                st.markdown(f"- {r['text']}  \n  <span style='opacity:.6'>{r['group']} · strength {r['strength']:.2f}</span>", unsafe_allow_html=True)
        else:
            st.write("No single rule fires; the statistical model sees an unusual combination of signals:")
        if case and case["model_signals"]:
            st.markdown("**Statistical model: strongest contributors**")
            for x in case["model_signals"]:
                st.markdown(f"- {x['label']}: {x['value']} (more extreme than {x['population_percentile']:.0f}% of plants)")
        if case:
            st.markdown("**Evidence to pull**")
            for k, v in case["evidence"].items():
                with st.expander(k.replace("_", " ")):
                    for line in v:
                        st.write("• " + line)
    with right:
        sc = S.loc[pick, ["rule_score", "iforest_score", "dominant_score", "ens_pct"]]
        fig = go.Figure(go.Bar(x=["Rules", "Isolation Forest", "Graph model", "Overall"],
                               y=[S["rule_score"].rank(pct=True)[pick], S["iforest_score"].rank(pct=True)[pick], S["dominant_score"].rank(pct=True)[pick], sc["ens_pct"]],
                               marker_color="#2f6f73"))
        fig.update_layout(title="Risk percentile by detector", yaxis_range=[0, 1], height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
        if case:
            st.markdown("**Inspection checklist**")
            for line in case["inspection_checklist"]:
                st.checkbox(line, key=f"{pick}-{line[:20]}")
            st.markdown("**Brands holding its certificates**")
            for b in case["brands_exposed"]:
                st.write("• " + b)

# ----------------------------------------------------------------------------------------------- brands
with tab_brand:
    st.markdown("Share of each brand's purchased certificates that trace back to high-risk recyclers (volume-weighted). "
                "The assumed prevalence of hollow recyclers is a setting, not a measurement.")
    bx = BX.join(brands[["name", "sector", "pibo_type"]]).sort_values("hollow_t_est", ascending=False)
    c1, c2 = st.columns([2, 3])
    c1.metric("Brands screened", f"{len(bx):,}")
    c1.metric("Estimated hollow tonnes held by brands", f"{bx['hollow_t_est'].sum():,.0f}")
    fig = px.scatter(bx.head(2000), x="bought_t", y="exposure", color="sector", hover_name="name", log_x=True,
                     labels={"bought_t": "certificate tonnes bought", "exposure": "share from high-risk recyclers"})
    c2.plotly_chart(fig, width="stretch")
    st.dataframe(bx.head(100)[["name", "sector", "bought_t", "exposure", "hollow_t_est", "n_recyclers", "top_sources"]], width="stretch")

# ----------------------------------------------------------------------------------------------- rings
with tab_ring:
    st.markdown("Closed loops of recyclers invoicing each other the same material (round-tripping).")
    fi = pd.read_parquet(DATA / "plastic" / "feedstock_invoices.parquet", columns=["seller_id", "buyer_id", "seller_type", "tonnes"])
    rr = fi[fi.seller_type == "recycler"].groupby(["seller_id", "buyer_id"], observed=True)["tonnes"].sum().reset_index()
    G = nx.DiGraph()
    for a, b, w in rr.itertuples(index=False):
        G.add_edge(a, b, t=float(w))
    cyc_nodes = {n for c in nx.simple_cycles(G, length_bound=6) for n in c}
    H = G.subgraph(cyc_nodes)
    comps = [c for c in nx.weakly_connected_components(H)]
    st.write(f"{len(comps)} loops involving {len(cyc_nodes)} recyclers.")
    if comps:
        k = st.selectbox("Loop", range(len(comps)), format_func=lambda i: f"Loop {i + 1} ({len(comps[i])} recyclers)")
        sub = H.subgraph(comps[k])
        pos = nx.spring_layout(sub, seed=3)
        ex, ey = [], []
        for u, v in sub.edges:
            ex += [pos[u][0], pos[v][0], None]
            ey += [pos[u][1], pos[v][1], None]
        fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=1.5, color="#999"), hoverinfo="none"))
        fig.add_trace(go.Scatter(x=[pos[n][0] for n in sub], y=[pos[n][1] for n in sub], mode="markers+text",
                                 text=[ent.at[n, "legal_name"][:22] for n in sub], textposition="top center",
                                 marker=dict(size=22, color=[S.at[n, "ens_pct"] for n in sub], colorscale="Reds", cmin=0.5, cmax=1, showscale=True,
                                             colorbar=dict(title="risk pct"))))
        fig.update_layout(showlegend=False, height=450, xaxis_visible=False, yaxis_visible=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------------------------- method
with tab_method:
    st.markdown("""
**Three layers, all unsupervised (no model was trained on fraud labels):**
1. **Rules** from physics and market facts: the electricity needed for the tonnes claimed, capacity, certificate price against the cost of recycling,
   closed trading loops, round-number invoices, new or over-capacity collectors.
2. **Isolation Forest + SHAP** on 100 recycler features, each converted to a one-sided suspicion level using prior knowledge of which direction is worse.
3. **Graph autoencoder** (DOMINANT-style) on the recycler/collector/offtaker/brand network, as a weaker complementary signal.

Recycler risk is then propagated to the brands that bought the certificates.
""")
    if metrics:
        rc = metrics["recycler"]
        rows = [{"detector": k.replace("_score", ""), **{f"AUC {t}": v["auc"][t] for t in ["all", "obvious", "moderate", "subtle"]},
                 "precision@100": v["precision_at_k"]["100"], "precision@225": v["precision_at_k"]["225"]} for k, v in rc.items()]
        st.dataframe(pd.DataFrame(rows).round(3), width="stretch")
        st.caption("Scored against planted fraud labels in the synthetic data. See the evaluation report for held-out seed, evasive-fraudster and data-blackout tests.")
    st.warning("All results are on synthetic data. The detectors were designed with knowledge of the fraud mechanisms the generator plants, so these numbers show that "
               "the mechanisms are detectable, not how accurate the system would be on real portal data.")
