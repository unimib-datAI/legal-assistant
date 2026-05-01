"""
Graph Initialization page.

Loads EU regulation documents into Neo4j and generates paragraph embeddings.
"""
import logging
from utils.streamlit_log_handler import StreamlitLogHandler
from langchain_huggingface import HuggingFaceEmbeddings
from service.graph.graph import Neo4jGraph
from service.graph.graph_loader import GraphLoader
from service.scraper.eurlex_document_utils import EurlexDocumentUtils
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
import config  # noqa: E402

st.title("Graph Initialization")
st.caption(
    "Downloads EU regulation documents from EUR-Lex, loads them into Neo4j, "
    "and generates paragraph embeddings."
)

# ── form ──────────────────────────────────────────────────────────────────────

celex_input = st.text_area(
    "CELEX IDs (one per line)",
    value="32016R0679\n32024R1689\n32023R2854\n32022R0868",
    height=120,
    help="GDPR · AI Act · Data Act · Data Governance Act",
)
clear_db = st.checkbox("Clear existing database before loading", value=True)

if st.button("Run Graph Initialization", type="primary"):
    celex_ids = [c.strip() for c in celex_input.splitlines() if c.strip()]
    if not celex_ids:
        st.error("Enter at least one CELEX ID.")
        st.stop()

    log_area = st.empty()
    handler = StreamlitLogHandler(log_area)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    try:
        with st.spinner("Initializing graph — this may take several minutes…"):
            graph = Neo4jGraph(config.NEO4J_URI, config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
            graph.verify_connection()

            if clear_db:
                graph.clear_database()

            eurlex_utils = EurlexDocumentUtils()
            loader = GraphLoader(graph)
            documents_config = [eurlex_utils.build_document_config(c) for c in celex_ids]
            loader.load_all_documents(documents_config)

            bge_embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-large-en-v1.5",
                encode_kwargs={"normalize_embeddings": True},
            )
            dimensions = graph.generate_text_embeddings(
                embed_fn=bge_embeddings.embed_documents,
                embedding_dim=1536,
                node_name="Paragraph",
            )
            graph.create_vector_index("Paragraph", "Paragraph", dimensions)
            graph.close()

        st.success(f"Graph initialized — {len(celex_ids)} document(s) loaded.")
    except Exception as exc:
        st.error(f"Error: {exc}")
    finally:
        root_logger.removeHandler(handler)
