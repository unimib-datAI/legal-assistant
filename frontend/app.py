"""
Central entry point for the Legal Assistant frontend.

Requires the package to be installed (``pip install -e .``), then run from the project
root::

    streamlit run frontend/app.py
"""
import streamlit as st
from dotenv import load_dotenv

from legal_assistant.logging_setup import configure_logging

load_dotenv()
configure_logging()

st.set_page_config(
    page_title="Legal Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = st.navigation(
    {
        "Assistant": [
            st.Page("evaluation/evaluation.py", title="Evaluation", icon="📊"),
        ],
        "Graph Construction": [
            st.Page("kg/graph_init.py", title="Graph Initialization", icon="🗄️"),
            st.Page("kg/aske_pipeline.py", title="ASKE", icon="🔍"),
            st.Page("kg/case_law_parser.py", title="Case Law Document Parser", icon="⚖️"),
        ],
    }
)
pages.run()
