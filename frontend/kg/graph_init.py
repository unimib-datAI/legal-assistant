"""
Graph Initialization page.

Downloads EU regulation documents from EUR-Lex, loads them into Neo4j, and generates
embeddings. All of that lives in ``legal_assistant.pipelines.graph_build`` — this page
only collects the parameters and streams the log output.
"""
import streamlit as st
from utils.streamlit_log_handler import stream_logs

from legal_assistant.pipelines.graph_build import DEFAULT_CELEX_IDS, build_graph
from legal_assistant.validation.gate import GraphValidationError

st.title("Graph Initialization")
st.caption(
    "Downloads EU regulation documents from EUR-Lex, loads them into Neo4j, "
    "and generates paragraph embeddings."
)

# ── form ──────────────────────────────────────────────────────────────────────

celex_input = st.text_area(
    "CELEX IDs (one per line)",
    value="\n".join(DEFAULT_CELEX_IDS),
    height=120,
    help="GDPR · AI Act · Data Act · Data Governance Act",
)
clear_db = st.checkbox("Clear existing database before loading", value=True)

if st.button("Run Graph Initialization", type="primary"):
    celex_ids = [c.strip() for c in celex_input.splitlines() if c.strip()]
    if not celex_ids:
        st.error("Enter at least one CELEX ID.")
        st.stop()

    try:
        with stream_logs(), st.spinner("Initializing graph — this may take several minutes…"):
            result = build_graph(celex_ids, clear_db=clear_db)
        st.success(f"Graph initialized — {len(result.celex_ids)} document(s) loaded.")
    except GraphValidationError as exc:
        st.error(
            "Graph validation failed — **nothing was written and the database was not "
            "cleared.** The parsed graph does not faithfully represent the source document:"
        )
        st.code(exc.report(), language="text")
    except Exception as exc:
        st.error(f"Error: {exc}")
