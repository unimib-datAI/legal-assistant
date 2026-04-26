"""
RAG Chat page.

Answers questions grounded in GDPR, AI Act, Data Act, and Data Governance Act.
"""
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.title("RAG Chat")
st.caption(
    "Ask questions about EU digital regulation. "
    "Answers are grounded in GDPR, AI Act, Data Act, and Data Governance Act."
)


@st.cache_resource(show_spinner="Loading RAG pipeline…")
def _get_rag():
    from rag_pipeline import RAGPipeline
    return RAGPipeline()


try:
    rag = _get_rag()
except Exception as exc:
    st.error(f"Failed to initialize RAG pipeline: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render conversation history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"- `{src}`")

# Chat input
question = st.chat_input("Ask a question about EU digital regulation…")
if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant paragraphs…"):
            try:
                result = rag.query(question)
                answer = result["answer"]
                sources = result.get("sources", [])
            except Exception as exc:
                answer = f"Error: {exc}"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("Sources", expanded=False):
                for src in sources:
                    st.markdown(f"- `{src}`")

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

if st.session_state["messages"]:
    if st.button("Clear conversation", key="clear_chat"):
        st.session_state["messages"] = []
        st.rerun()
