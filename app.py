"""Unified waste-audit workspace. Start with: streamlit run app.py."""
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.settings import settings
from ui.theme import apply_theme, masthead, sidebar

st.set_page_config(page_title="TraceWAI", layout="wide", initial_sidebar_state="auto")
apply_theme()
config = settings()
pages = [
    st.Page("ui/pages/overview.py", title="Overview", default=True),
    st.Page("ui/pages/plastic.py", title="Plastic", url_path="plastic"),
    st.Page("ui/pages/biomedical.py", title="Biomedical", url_path="biomedical"),
    st.Page("ui/pages/chemical.py", title="Chemical", url_path="chemical"),
]
legacy_pages = [
    st.Page("ui/pages/plastic_legacy.py", title="Plastic", url_path="phantom_plastic"),
    st.Page("ui/pages/biomedical_legacy.py", title="Biomedical", url_path="biomedical_waste"),
]
page = st.navigation(pages + legacy_pages, position="hidden")
sidebar(pages, config, page.title)
masthead(page.title)
page.run()
