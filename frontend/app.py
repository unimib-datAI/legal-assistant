"""
Central entry point for the Legal Assistant frontend.

Run with (from the project root):
    streamlit run frontend/app.py
"""
import sys
from pathlib import Path

import streamlit as st

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="Legal Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = st.navigation(
    {
        "Graph Construction": [
            st.Page("graph_construction.py", title="Graph Initialization", icon="🗄️"),
            st.Page("aske_pipeline.py", title="ASKE Pipeline", icon="🔍"),
            st.Page("case_law_parser.py", title="Case Law Parser", icon="⚖️"),
        ],
        "RAG": [
            st.Page("rag_chat.py", title="Chat", icon="💬"),
        ],
    }
)
pages.run()
