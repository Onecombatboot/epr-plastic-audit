"""Inspector dashboard: observable evidence, separate queues and transparent claim reduction."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from yieldbal.checks import compute, product_balance, what_if
from yieldbal.prep import DEFAULT_DATA, DEFAULT_RUN, load, load_configs, load_reference
from yieldbal.scoring import reasons

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
args, _ = parser.parse_known_args()
st.set_page_config(page_title="Mass-Balance & Yield", page_icon="♻️", layout="wide")


@st.cache_data
def read_run(run_path: str, data_path: str):
    run = Path(run_path)
    scores = pd.read_parquet(run/"scores.parquet")
    monthly_path = run/"monthly_checks.parquet"
    monthly = pd.read_parquet(monthly_path) if monthly_path.exists() else compute(load(Path(data_path)), load_reference())[0]
    signals = {signal["entity_id"]: signal for signal in
               (json.loads(line) for line in (run/"signals.jsonl").read_text().splitlines())}
    metrics = json.loads((run/"metrics.json").read_text()) if (run/"metrics.json").exists() else {}
    return scores, monthly, signals, metrics


@st.cache_data
def read_product_balance(data_path: str, entity_id: str, case: pd.DataFrame) -> pd.DataFrame:
    _, products = load_configs(Path(data_path)/"yield/stream_configs")
    sales = pd.read_parquet(Path(data_path)/"yield/outputs_monthly.parquet", filters=[("entity_id", "==", entity_id)])
    sales["month"] = pd.to_datetime(sales.month)
    return product_balance(case, sales, products, load_reference())


st.title("Mass-Balance & Yield")
st.caption("Recycling certificates should be backed by products, electricity and plant capacity.")
st.warning("All data is synthetic. Detectors were designed knowing the planted mechanisms. Results do not establish real-world accuracy; configured yields, tariffs and energy limits are assumptions.")
if not (args.run/"scores.parquet").exists():
    st.info("Run `python -m yieldbal.run_eval` from the repository root to create the audit results.")
    st.stop()
try:
    scores, monthly, signals, metrics = read_run(str(args.run), str(args.data))
except (FileNotFoundError, ValueError) as error:
    st.error(f"Unable to load audit evidence: {error}")
    st.stop()

left, middle, right = st.columns(3)
stream = left.selectbox("Waste stream", ["All streams"]+sorted(scores.stream.unique()), key="stream")
state = middle.selectbox("State", ["All states"]+sorted(scores.state.unique()), key="state")
policy = right.selectbox("Inspection queue", ["All entities", "Priority", "Screening including priority"], key="policy")
view = scores.copy()
if stream != "All streams":
    view = view[view.stream.eq(stream)]
if state != "All states":
    view = view[view.state.eq(state)]
if policy != "All entities":
    view = view[view.priority if policy == "Priority" else view.screening]

rank_tab, case_tab, comparison_tab, method_tab = st.tabs(["Inspection list", "Recycler case file", "Stream comparison", "Method & results"])
with rank_tab:
    a, b, c, d = st.columns(4)
    a.metric("Recyclers in view", f"{len(view):,}")
    b.metric("Priority cases", f"{int(view.priority.sum()):,}")
    c.metric("Suspected hollow tonnes", f"{view.suspected_hollow_t.sum():,.0f} t")
    d.metric("Certificates claimed", f"{view.claimed_t.sum():,.0f} t")
    st.caption("Unsupported tonnes are a what-if estimate under configured ceilings. They are not verified fraud or measured waste.")
    if view.empty:
        st.info("No recyclers match these filters.")
    else:
        shown = view.rename_axis("entity_id").reset_index()[["rank", "entity_id", "legal_name", "stream", "state", "score", "tier", "suspected_hollow_t", "bill_coverage"]]
        st.dataframe(shown, width="stretch", hide_index=True,
                     column_config={"score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=1, format="%.3f")})
        st.download_button("Download inspection list", shown.to_csv(index=False), "yield_inspection_list.csv", "text/csv")

with case_tab:
    if view.empty:
        st.info("Adjust the filters to open a recycler case.")
    else:
        entity = st.selectbox("Recycler", list(view.index), key="entity",
                              format_func=lambda entity: f"#{int(scores.at[entity, 'rank'])} · {scores.at[entity, 'legal_name']} · {entity}")
        row = scores.loc[entity]
        case = monthly[monthly.entity_id.eq(entity)].sort_values("month")
        st.subheader(row.legal_name)
        st.caption(f"{row.state} · {row.stream} · {row.process_type} · {row.tier.replace('_', ' ')}")
        chart, detail = st.columns([3, 2])
        with chart:
            fig = go.Figure()
            for column, label, color in (("certificates_t", "Claimed input", "#c25132"),
                                         ("energy_supported_t", "Energy supported", "#287866"),
                                         ("output_supported_t", "Output supported (6-month window)", "#3069a3"),
                                         ("capacity_supported_t", "Monthly capacity", "#8a7d64")):
                fig.add_trace(go.Scatter(x=case.month, y=case[column], name=label, line={"color": color}))
            fig.update_layout(yaxis_title="Tonnes", height=430, margin={"l": 10, "r": 10, "t": 15, "b": 10},
                              legend={"orientation": "h", "y": -0.2})
            st.plotly_chart(fig, width="stretch")
            st.caption("Output support credits up to 2 future sales months to the trailing input window. The last 2 months have incomplete output evidence; absent bills produce gaps.")
        with detail:
            st.markdown("**Reasons to inspect**")
            for reason in signals.get(entity, {}).get("reasons", reasons(row, case)):
                st.write("• "+reason)
            st.markdown("**Evidence references**")
            for evidence in signals.get(entity, {}).get("evidence_refs", [f"operations_monthly:{entity}:{case.iloc[0].month:%Y-%m}"]):
                st.code(evidence)
            contributions = signals.get(entity, {}).get("model_contributions", [])
            if contributions:
                st.caption("Isolation Forest attribution (negative values increase anomaly).")
                st.dataframe(pd.DataFrame(contributions), hide_index=True)
        st.markdown("**What if the recycler lowers its claim?**")
        percent = st.slider("Retain this percentage of each monthly claim", 0, 100, 100, key="claim_percent")
        result = what_if(case, percent/100)
        x, y, z = st.columns(3)
        x.metric("Revised claim", f"{result['claimed_t']:,.0f} t")
        y.metric("Still unexplained", f"{result['unexplained_t']:,.0f} t")
        z.metric("Maximum share passing tonnage checks", f"{result['safe_fraction']:.1%}")
        if result["tonnage_checks_pass"]:
            st.success("The revised claim passes all available tonnage ceilings.")
        else:
            st.info("Reduce the monthly claims further to meet every available tonnage ceiling.")
        if not result["bill_checks_pass"]:
            st.warning("Reducing claims cannot fix inconsistent electricity charges. Verify the original bills separately.")
        if result["unknown_energy_months"]:
            st.caption(f"{result['unknown_energy_months']} months have unknown energy support. Passing available checks does not establish physical processing.")
        with st.expander("Monthly evidence table"):
            st.dataframe(case[["month", "certificates_t", "supported_t", "suspected_hollow_t", "bill_id", "bill_error", "checks_testable"]], hide_index=True, width="stretch")
        with st.expander("Saleable-product balance"):
            st.caption("Each product is compared with its configured expected and lower yields over a 6-month input window plus 2 months of sales lag. Individual yield priors are review evidence; the aggregate saleable lower bound supplies the output ceiling.")
            st.dataframe(read_product_balance(str(args.data), entity, case), hide_index=True, width="stretch")

with comparison_tab:
    comparison = scores.groupby("stream").agg(recyclers=("score", "size"), priority=("priority", "sum"),
                                               claimed_t=("claimed_t", "sum"), unsupported_t=("suspected_hollow_t", "sum"))
    st.dataframe(comparison, width="stretch")
    st.bar_chart(comparison[["claimed_t", "unsupported_t"]])

with method_tab:
    st.markdown("The auditor reads stream/process physics from YAML. It compares claimed input with capacity, connected load, electricity after base load and declared solar, and saleable products over 3- and 6-month windows with 2 months of sales lag. Bills are checked against state/FY tariffs. Missing bills remain unknown; output mix and changes against the first 6 active months add supporting evidence.")
    st.markdown("One-sided residuals use stream/process medians and MAD/IQR. An Isolation Forest sees only direction-aware suspicion features; SHAP describes its attributions. Weighted physical-feature ranks, residual ranks and forest ranks form a score in [0, 1]. Scores rank inspections and are not fraud probabilities. Labels are read only by evaluation.")
    st.markdown("**Priority:** at least 15% average deficit in two physical groups, at least 50% in one group, or a tariff discrepancy above 6%. **Screening:** at least 5% average deficit in one physical group or a tariff discrepancy above 3%. These assumptions are fixed before evaluation.")
    st.markdown("Undeclared solar, incorrect tariff references, campaign inventories and stable atypical yields can still produce false positives. Proportional claim reduction does not resolve bill arithmetic or auxiliary statistical anomalies.")
    if metrics:
        st.json({"pooled": metrics["pooled"], "false_positives": metrics["false_positives"], "acceptance": metrics["acceptance"]}, expanded=False)
    st.caption("Evaluation output was reviewed during development for correctness. See the package README for changes, actual held-out results and ablations.")
