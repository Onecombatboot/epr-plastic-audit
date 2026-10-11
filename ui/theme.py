"""Swiss typography, shared navigation and local CSS; no remote assets."""
from html import escape
from pathlib import Path

import plotly.io as pio
import streamlit as st

from ui.settings import Settings


def apply_theme():
    st.html(Path(__file__).parent/"assets/dashboard.css")
    pio.templates["audit_swiss"] = {
        "layout": {
            "font": {"family": "Helvetica Neue, Helvetica, Arial, sans-serif", "color": "#181818"},
            "paper_bgcolor": "#FFFFFF", "plot_bgcolor": "#FFFFFF",
            "colorway": ["#002FA7", "#525252", "#999999", "#CCCCCC"],
            "xaxis": {"gridcolor": "#E2E2E2", "zerolinecolor": "#E2E2E2"},
            "yaxis": {"gridcolor": "#E2E2E2", "zerolinecolor": "#E2E2E2"},
        }
    }
    pio.templates.default = "audit_swiss"


def sidebar(pages, config: Settings, current_page: str):
    with st.sidebar:
        st.html('''<div class="audit-shell" data-theme="audit">
          <div class="workspace-brand">Waste<span>Audit.</span></div>
          <p class="workspace-caption">Evidence for every inspection.</p>
          <div class="nav-label">Workspace</div>
        </div>''')
        for number, page in enumerate(pages):
            label = page.title if number == 0 else f"{number:02d}  {page.title}"
            with st.container(key="active_navigation" if page.title == current_page else f"navigation_{number}"):
                st.page_link(page, label=label)
        st.html('''<div class="audit-shell sidebar-note" data-theme="audit">
          <div class="sidebar-rule"></div>
          <span class="audit-badge audit-badge-outline">Synthetic data</span>
          <p>Inspection ranks are not fraud probabilities. Review the source evidence before drawing conclusions.</p>
        </div>''')
        with st.expander("Data sources"):
            st.caption("The workspace reads saved audit results. Each use case keeps its own scoring policy.")
            st.write("Dataset", config.data.name)
            for label, path in (("Plastic", config.plastic), ("Biomedical", config.biomedical), ("Chemical", config.chemical)):
                st.write(label, path.name)


def masthead(title: str):
    st.html(f'''<div class="audit-shell masthead" data-theme="audit">
      <span>Waste compliance <span class="masthead-divider">/</span> {escape(title)}</span>
      <span class="masthead-end">Three auditors. One workspace.</span>
    </div>''')
