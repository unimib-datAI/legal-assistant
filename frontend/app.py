"""
Central entry point for the Legal Assistant frontend.

Run with (from the project root):
    streamlit run frontend/app.py
"""
import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="Legal Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = st.navigation(
    {
        "Graph Construction": [
            st.Page("kg/graph_init.py", title="Graph Initialization", icon="🗄️"),
            st.Page("kg/aske_pipeline.py", title="ASKE", icon="🔍"),
            st.Page("kg/case_law_parser.py", title="Case Law Document Parser", icon="⚖️"),
        ],
    }
)
pages.run()
