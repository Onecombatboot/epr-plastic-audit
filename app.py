"""National Waste Compliance Audit Portal.

Unified Inspector Dashboard uniting two autonomous audit agents:
1. Phantom Plastic: Unsupervised detection of hollow EPR plastic recycling certificates
2. Biomedical Waste Gap: Detection of hospital waste dumping and treatment plant fraud
"""
import streamlit as st

st.set_page_config(
    page_title="National Waste Compliance Audit",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

phantom_page = st.Page(
    "phantom/app.py",
    title="Phantom Plastic (EPR Fraud)",
    icon="♻️",
    default=True,
    url_path="phantom_plastic"
)

biomed_page = st.Page(
    "biomed/app.py",
    title="Biomedical Waste Gap",
    icon="🏥",
    url_path="biomedical_waste"
)

pg = st.navigation({
    "Audit Agents": [phantom_page, biomed_page]
})

st.sidebar.markdown("---")
st.sidebar.caption("⚖️ **National Waste Compliance Audit**")
st.sidebar.caption("Built for WeMakeDevs Environmental Hacks (Waste & Energy Track)")

pg.run()
