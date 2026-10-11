"""Real saved-result summaries; no combined risk score or label access."""
from html import escape

import pandas as pd
import streamlit as st

from ui.auditors import missing_inputs
from ui.settings import Settings

USE_CASES = (
    ("plastic", "01", "Plastic", "Phantom Plastic", "Trace certificates back to physical recycling.",
     "Electricity, capacity, trading patterns and brand exposure.", "recyclers screened"),
    ("biomedical", "02", "Biomedical", "Biomedical Waste Gap", "Follow hospital waste from pickup to treatment.",
     "Barcode collections, GPS stops and incinerator logs.", "healthcare facilities screened"),
    ("chemical", "03", "Chemical", "Mass-Balance & Yield", "Match claimed recovery to what a plant can support.",
     "E-waste, batteries, tyres and used oil; products, bills and capacity.", "recovery plants screened"),
)


@st.cache_data(ttl=60, show_spinner=False)
def summarize(config: Settings) -> dict:
    result = {}
    for name, *_ in USE_CASES:
        item = {"ready": False, "count": None, "flagged": None, "note": "Results unavailable"}
        if not missing_inputs(name, config):
            try:
                run = getattr(config, name)
                if name == "plastic":
                    scores = pd.read_parquet(run/"recycler_scores.parquet", columns=["ens_pct"])
                    item.update(count=len(scores), flagged=int((scores.ens_pct >= .90).sum()),
                                note="in the top 10% inspection list")
                elif name == "biomedical":
                    scores = pd.read_parquet(run/"facility_scores.parquet", columns=["tier"])
                    plants = pd.read_parquet(run/"cbwtf_scores.parquet", columns=["score"])
                    item.update(count=len(scores), flagged=int(scores.tier.eq("priority").sum()),
                                note="priority facilities", plants=len(plants))
                else:
                    scores = pd.read_parquet(run/"scores.parquet", columns=["priority", "stream"])
                    item.update(count=len(scores), flagged=int(scores.priority.sum()),
                                note="priority recovery plants", streams=scores.stream.nunique())
                item["ready"] = item["count"] > 0
            except (OSError, ValueError, KeyError):
                item["note"] = "Results could not be read"
        result[name] = item
    return result


def render_overview(config: Settings):
    summary = summarize(config)
    st.html('''<section class="audit-shell overview-hero" data-theme="audit">
      <div class="audit-hero">
        <div class="audit-hero-content">
          <div class="hero-copy">
            <h1>Waste audits,<br><span>in one place.</span></h1>
            <p>Inspect plastic certificates, biomedical collections and chemical recovery. Move from a ranked case to the evidence behind it.</p>
          </div>
          <div class="hero-number"><span>03</span><p>Auditors in this workspace</p></div>
        </div>
      </div>
    </section>''')
    st.html('''<div class="audit-shell overview-heading" data-theme="audit">
      <h2>Choose an audit</h2><p>Each queue uses its own evidence and inspection policy.</p>
    </div>''')
    columns = st.columns(3, gap="small")
    for column, (name, number, title, agent, description, coverage, unit) in zip(columns, USE_CASES):
        item = summary[name]
        count = f"{item['count']:,}" if item["count"] is not None else "Unavailable"
        detail = f"<strong>{item['flagged']:,}</strong> {escape(item['note'])}" if item["flagged"] is not None else escape(item["note"])
        with column:
            st.html(f'''<article class="audit-shell audit-card audit-card-border use-case" data-theme="audit">
              <div class="audit-card-body">
                <div class="use-case-top"><span class="use-case-number">{number}</span><span class="agent-name">{escape(agent)}</span></div>
                <h3 class="audit-card-title">{title}</h3>
                <p class="use-case-description">{escape(description)}</p>
                <div class="audit-stats"><div class="audit-stat">
                  <div class="audit-stat-value">{count}</div>
                  <div class="audit-stat-title">{unit}</div>
                  <div class="audit-stat-desc">{detail}</div>
                </div></div>
                <p class="use-case-coverage">{escape(coverage)}</p>
              </div>
            </article>''')
            if st.button(f"Open {title.lower()} audit", key=f"open_{name}", width="stretch"):
                st.switch_page(f"ui/pages/{name}.py")
    st.html('''<div class="audit-shell overview-disclosure" data-theme="audit">
      <div class="audit-alert audit-alert-outline" role="note">
        <span class="audit-badge audit-badge-outline">Synthetic data</span>
        <p>These are generated datasets with planted fraud patterns. Scores support inspection; they do not establish real-world accuracy or prove fraud.</p>
      </div>
    </div>''')
    st.html('''<div class="audit-shell evidence-path" data-theme="audit">
      <h2>From a signal to its source.</h2>
      <div class="evidence-steps">
        <div><span>01</span><h3>Inspect the queue</h3><p>Review ranked recyclers, facilities or treatment plants.</p></div>
        <div><span>02</span><h3>Open the case</h3><p>Read the reasons, monthly checks and model contributions.</p></div>
        <div><span>03</span><h3>Review the evidence</h3><p>Trace findings to bills, pickups, sales and treatment records.</p></div>
      </div>
    </div>''')
