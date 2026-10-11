"""Render the existing dashboards with explicit paths inside one app session."""
from __future__ import annotations

import streamlit as st

from ui.settings import Settings

REQUIRED = {
    "plastic": {
        "run": ("recycler_scores.parquet", "features.parquet", "brand_exposure.parquet"),
        "data": ("registry/entities.parquet", "plastic/brands.parquet", "plastic/feedstock_invoices.parquet"),
        "command": "python -m phantom.pipeline --data data/synthetic --out phantom/outputs/seed42",
    },
    "biomedical": {
        "run": ("facility_scores.parquet", "cbwtf_scores.parquet", "top_facility_pickups.parquet", "flagged_pickups.parquet", "signals.jsonl"),
        "data": ("bmw/facilities.parquet", "bmw/facility_changes.parquet", "bmw/cbwtf_daily_treatment.parquet", "bmw/pickups.parquet"),
        "command": "python -m biomed.run_eval --data data/synthetic --out biomed/outputs/seed42",
    },
    "chemical": {
        "run": ("scores.parquet", "signals.jsonl"),
        "data": ("registry/entities.parquet", "registry/tariff_reference.parquet", "yield/operations_monthly.parquet", "yield/electricity_bills.parquet", "yield/outputs_monthly.parquet", "yield/stream_configs"),
        "command": "python -m yieldbal.run_eval --data data/synthetic --out yieldbal/outputs/seed42",
    },
}


def missing_inputs(name: str, config: Settings) -> list[str]:
    run = getattr(config, name)
    spec = REQUIRED[name]
    return [f"{kind}/{file}" for kind, directory in (("run", run), ("data", config.data))
            for file in spec[kind] if not (directory/file).exists()]


def render_auditor(name: str, config: Settings):
    number = {"plastic": "01", "biomedical": "02", "chemical": "03"}[name]
    st.html(f'<div class="audit-shell section-number" data-theme="audit">{number}</div>')
    missing = missing_inputs(name, config)
    if missing:
        st.title({"plastic": "Phantom Plastic", "biomedical": "Biomedical Waste Gap", "chemical": "Chemical recovery"}[name])
        st.info("This auditor's results or source data are unavailable. The other auditors are still accessible from the sidebar.")
        with st.expander("Set up this auditor", expanded=True):
            st.write("Run from the repository root, then reload this page:")
            st.code(REQUIRED[name]["command"], language="bash")
            st.caption("Missing files: "+", ".join(missing))
        return
    if name == "plastic":
        from phantom.app import render
    elif name == "biomedical":
        from biomed.app import render
    else:
        from yieldbal.app import render
    try:
        render(getattr(config, name), config.data, integrated=True)
    except (FileNotFoundError, OSError) as error:
        st.error(f"Unable to read {name} audit evidence: {error}")
